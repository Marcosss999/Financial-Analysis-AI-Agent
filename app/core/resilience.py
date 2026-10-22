"""
稳定性基础设施 (Resilience Primitives)
职责：
1. CircuitBreaker  —— 熔断器：连续失败达到阈值后快速失败，冷却后半开探测，避免故障上游拖垮整条分析链路。
2. retry_async     —— 带单次超时的指数退避重试(含抖动)，仅用于幂等的 LLM 调用。
3. TokenBucket     —— 令牌桶限流：按客户端维度限制请求速率，防止 LLM 调用被刷爆。
4. IdempotencyCache—— 幂等缓存：相同 Idempotency-Key 的重复提交只执行一次，并发重复请求共享同一次执行结果。

全部为纯 Python 标准库实现，时钟与 sleep 可注入，便于单元测试。
"""

import asyncio
import random
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


class CircuitOpenError(RuntimeError):
    """熔断器处于打开状态，调用被直接拒绝"""


class CircuitBreaker:
    """
    三态熔断器：
      CLOSED    正常放行，统计连续失败次数；
      OPEN      连续失败 >= failure_threshold 后打开，recovery_seconds 内全部快速失败；
      HALF_OPEN 冷却结束后只放行一个探测请求，成功则关闭，失败则重新打开。
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._probe_in_flight = False
        self._state = self.CLOSED

    @property
    def state(self) -> str:
        if self._state == self.OPEN and self._opened_at is not None \
                and self._clock() - self._opened_at >= self.recovery_seconds:
            return self.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        st = self.state
        if st == self.CLOSED:
            return True
        if st == self.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True  # 只放行一个探测请求
            self._state = self.HALF_OPEN
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._probe_in_flight = False
        self._opened_at = None
        self._state = self.CLOSED

    def record_failure(self) -> None:
        self._probe_in_flight = False
        if self._state == self.HALF_OPEN or self.state == self.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = self.OPEN
        self._opened_at = self._clock()

    def snapshot(self) -> Dict[str, Any]:
        return {"state": self.state, "consecutive_failures": self._failures}


async def retry_async(
    fn: Callable[[], Awaitable[Any]],
    retries: int = 2,
    base_delay: float = 0.5,
    timeout: Optional[float] = None,
    max_delay: float = 8.0,
    jitter: float = 0.2,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
) -> Tuple[Any, int]:
    """
    执行 fn，失败按 base_delay * 2^n (+/- jitter) 指数退避重试，单次调用受 timeout 约束。
    :return: (结果, 实际重试次数)；全部失败则抛出最后一次异常。
    """
    attempt = 0
    while True:
        try:
            coro = fn()
            result = await (asyncio.wait_for(coro, timeout) if timeout else coro)
            return result, attempt
        except asyncio.CancelledError:
            raise  # 被取消(如整体超时)必须向上传播，不能吞掉
        except Exception as e:  # noqa: BLE001 —— 重试层需要捕获所有可恢复异常
            if attempt >= retries:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay *= 1.0 + random.uniform(-jitter, jitter)
            if on_retry:
                on_retry(attempt + 1, e)
            await sleep(max(0.0, delay))
            attempt += 1


class TokenBucket:
    """按 key(如客户端 IP)维度的令牌桶：每分钟补充 rate_per_minute 个令牌，桶容量为 burst"""

    def __init__(self, rate_per_minute: int = 30, burst: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.rate_per_sec = rate_per_minute / 60.0
        self.capacity = float(burst if burst is not None else rate_per_minute)
        self._clock = clock
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_ts)

    def allow(self, key: str) -> Tuple[bool, float]:
        """:return: (是否放行, 被拒绝时建议等待的秒数)"""
        now = self._clock()
        tokens, last = self._buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate_per_sec)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return True, 0.0
        self._buckets[key] = (tokens, now)
        wait = (1.0 - tokens) / self.rate_per_sec if self.rate_per_sec > 0 else 60.0
        return False, wait


class IdempotencyCache:
    """
    幂等执行缓存：
      - 同一个 key 在 TTL 内重复提交，直接返回第一次的结果，不重复调用 LLM；
      - 同一个 key 并发到达时，后到的请求等待第一次执行完成并共享其结果(而不是并发再跑一遍)；
      - 第一次执行失败则不缓存失败，允许客户端用同一个 key 重试。
    """

    def __init__(self, ttl_seconds: float = 600.0, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl_seconds
        self._clock = clock
        self._done: Dict[str, Tuple[float, Any]] = {}
        self._inflight: Dict[str, "asyncio.Future[Any]"] = {}

    def _purge(self) -> None:
        now = self._clock()
        for k in [k for k, (ts, _) in self._done.items() if now - ts > self.ttl]:
            self._done.pop(k, None)

    async def run(self, key: str, factory: Callable[[], Awaitable[Any]]) -> Tuple[Any, bool]:
        """:return: (结果, 是否命中幂等缓存/共享了他人的执行)"""
        self._purge()
        if key in self._done:
            return self._done[key][1], True
        if key in self._inflight:
            return await asyncio.shield(self._inflight[key]), True

        fut: "asyncio.Future[Any]" = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            result = await factory()
        except BaseException as e:
            self._inflight.pop(key, None)
            if not fut.done():
                fut.set_exception(e)
                fut.exception()  # 标记已取回，避免 “exception was never retrieved” 告警
            raise
        self._done[key] = (self._clock(), result)
        self._inflight.pop(key, None)
        fut.set_result(result)
        return result, False

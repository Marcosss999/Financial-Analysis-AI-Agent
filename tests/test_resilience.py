import asyncio

from app.core.middleware import RateLimitMiddleware
from app.core.resilience import CircuitBreaker, IdempotencyCache, TokenBucket, retry_async
from tests.helpers import run


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_opens_then_half_open_then_closes():
    clk = Clock()
    b = CircuitBreaker(failure_threshold=3, recovery_seconds=30, clock=clk)
    for _ in range(2):
        b.record_failure()
    assert b.state == "closed" and b.allow()
    b.record_failure()
    assert b.state == "open" and not b.allow()          # 打开：快速失败
    clk.t = 31
    assert b.state == "half_open"
    assert b.allow() and not b.allow()                   # 半开：只放行一个探测请求
    b.record_success()
    assert b.state == "closed" and b.allow()


def test_breaker_probe_failure_reopens():
    clk = Clock()
    b = CircuitBreaker(failure_threshold=1, recovery_seconds=10, clock=clk)
    b.record_failure()
    clk.t = 11
    assert b.allow()
    b.record_failure()
    assert b.state == "open" and not b.allow()
    clk.t = 22
    assert b.state == "half_open"


def test_retry_succeeds_after_failures_with_exponential_backoff():
    calls, sleeps = {"n": 0}, []

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    async def fake_sleep(d):
        sleeps.append(d)

    res, retries = run(retry_async(flaky, retries=2, base_delay=1.0, jitter=0.0, sleep=fake_sleep))
    assert (res, retries, calls["n"]) == ("ok", 2, 3)
    assert sleeps == [1.0, 2.0]


def test_retry_exhausted_raises_last_error():
    async def always_fail():
        raise ValueError("x")

    async def no_sleep(_):
        pass

    try:
        run(retry_async(always_fail, retries=2, sleep=no_sleep))
    except ValueError:
        return
    raise AssertionError("应抛出异常")


def test_retry_timeout_counts_as_failure():
    async def slow():
        await asyncio.sleep(1)

    async def no_sleep(_):
        pass

    try:
        run(retry_async(slow, retries=1, timeout=0.01, sleep=no_sleep))
    except asyncio.TimeoutError:
        return
    raise AssertionError("应超时")


def test_token_bucket_burst_and_refill():
    clk = Clock()
    tb = TokenBucket(rate_per_minute=60, burst=2, clock=clk)   # 1 token/s
    assert tb.allow("a")[0] and tb.allow("a")[0]
    ok, wait = tb.allow("a")
    assert not ok and 0 < wait <= 1.0
    assert tb.allow("b")[0]                                    # 不同客户端互不影响
    clk.t = 1.0
    assert tb.allow("a")[0]


def test_idempotency_runs_once_sequentially_and_concurrently():
    cache, count = IdempotencyCache(ttl_seconds=60), {"n": 0}

    async def work():
        count["n"] += 1
        await asyncio.sleep(0.02)
        return {"v": count["n"]}

    async def main():
        r = await asyncio.gather(*[cache.run("k", work) for _ in range(5)])
        again = await cache.run("k", work)
        return r, again

    results, again = run(main())
    assert count["n"] == 1                                     # 5 个并发 + 1 个后续，只执行 1 次
    assert all(r[0] == {"v": 1} for r in results) and again == ({"v": 1}, True)
    assert sum(1 for r in results if not r[1]) == 1            # 只有第一个是“真正执行”


def test_idempotency_does_not_cache_failures():
    cache, state = IdempotencyCache(), {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("first fails")
        return "ok"

    async def main():
        try:
            await cache.run("k", flaky)
        except RuntimeError:
            pass
        return await cache.run("k", flaky)

    assert run(main()) == ("ok", False)


def test_rate_limit_middleware_returns_429_with_retry_after():
    sent, called = [], {"n": 0}

    async def app(scope, receive, send):
        called["n"] += 1

    mw = RateLimitMiddleware(app, TokenBucket(rate_per_minute=60, burst=1))
    scope = {"type": "http", "method": "POST", "path": "/finance/analyze", "client": ("1.2.3.4", 1)}

    async def send(msg):
        sent.append(msg)

    async def main():
        await mw(scope, None, send)
        await mw(scope, None, send)
        # 非受保护路径不受限
        await mw({**scope, "path": "/health", "method": "GET"}, None, send)

    run(main())
    assert called["n"] == 2
    assert sent[0]["status"] == 429
    assert any(k == b"retry-after" for k, _ in sent[0]["headers"])

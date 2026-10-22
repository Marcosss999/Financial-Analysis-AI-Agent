"""
可观测性 (Observability)：轻量级链路追踪，思路类似 Langfuse 的 trace / span 模型。
  - 每次分析生成一个 RunTrace(trace_id = run_id)；
  - 每个图节点对应一个 span：耗时、状态、异常、该节点内的 LLM 调用次数 / 降级次数 / 重试次数 / token；
  - 通过 ContextVar 把 trace 透传到 llm_provider，业务代码无需层层传参；并行节点各自拥有独立的 span 上下文。
trace 会随响应返回，并落库到 SQLite，可通过 /finance/runs/{run_id} 复盘任意一次分析。
"""

import contextvars
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("fin.trace")

_current_trace: contextvars.ContextVar[Optional["RunTrace"]] = contextvars.ContextVar("fin_trace", default=None)
_current_span: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("fin_span", default=None)


class RunTrace:
    def __init__(self, run_id: Optional[str] = None, ticker: str = ""):
        self.run_id = run_id or uuid.uuid4().hex[:16]
        self.ticker = ticker
        self.t0 = time.perf_counter()
        self.spans: List[Dict[str, Any]] = []
        self.llm = {"calls": 0, "retries": 0, "fallbacks": 0, "failures": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "latency_ms_total": 0.0}
        self.degraded_nodes: List[str] = []

    def start_span(self, name: str) -> Dict[str, Any]:
        span = {"name": name, "start_ms": round((time.perf_counter() - self.t0) * 1000, 1),
                "duration_ms": None, "status": "running", "error": None,
                "llm_calls": 0, "llm_fallbacks": 0, "llm_retries": 0}
        self.spans.append(span)
        return span

    def end_span(self, span: Dict[str, Any], status: str, error: Optional[str] = None) -> None:
        span["duration_ms"] = round((time.perf_counter() - self.t0) * 1000 - span["start_ms"], 1)
        span["status"] = status
        span["error"] = error
        if status != "ok" and span["name"] not in self.degraded_nodes:
            self.degraded_nodes.append(span["name"])
        logger.info(json.dumps({"trace_id": self.run_id, "span": span["name"], "status": status,
                                "duration_ms": span["duration_ms"]}, ensure_ascii=False))

    def record_llm(self, *, latency_ms: float, retries: int, fallback: bool, failed: bool,
                   prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.llm["calls"] += 1
        self.llm["retries"] += retries
        self.llm["fallbacks"] += int(fallback)
        self.llm["failures"] += int(failed)
        self.llm["prompt_tokens"] += prompt_tokens
        self.llm["completion_tokens"] += completion_tokens
        self.llm["latency_ms_total"] = round(self.llm["latency_ms_total"] + latency_ms, 1)
        span = _current_span.get()
        if span is not None:
            span["llm_calls"] += 1
            span["llm_fallbacks"] += int(fallback)
            span["llm_retries"] += retries

    def to_dict(self) -> Dict[str, Any]:
        spans = sorted(self.spans, key=lambda s: s["start_ms"])
        return {
            "run_id": self.run_id,
            "ticker": self.ticker,
            "total_ms": round((time.perf_counter() - self.t0) * 1000, 1),
            "spans": spans,
            "llm": dict(self.llm),
            "degraded_nodes": list(self.degraded_nodes),
        }


def start_trace(ticker: str = "", run_id: Optional[str] = None) -> RunTrace:
    trace = RunTrace(run_id=run_id, ticker=ticker)
    _current_trace.set(trace)
    return trace


def get_trace() -> Optional[RunTrace]:
    return _current_trace.get()


def traced_node(name: str, fn: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
                degrade_to: Optional[Callable[[Dict[str, Any], BaseException], Dict[str, Any]]] = None):
    """
    给图节点加上 span 记录。
    degrade_to: 提供时，节点抛异常不会中断整条图，而是返回降级结果，并在 trace 里标记该节点降级；
                不提供则异常继续向上抛(关键节点失败应当让整次分析失败，而不是产出错误结论)。
    """
    async def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        trace = _current_trace.get()
        span = trace.start_span(name) if trace else None
        token = _current_span.set(span)
        try:
            out = await fn(state)
            if trace:
                trace.end_span(span, "ok")
            return out
        except Exception as e:  # noqa: BLE001
            if trace:
                trace.end_span(span, "error", f"{type(e).__name__}: {e}")
            if degrade_to is None:
                raise
            logger.warning("node %s failed, degraded: %s", name, e)
            return degrade_to(state, e)
        finally:
            _current_span.reset(token)

    wrapper.__name__ = f"traced_{name}"
    return wrapper

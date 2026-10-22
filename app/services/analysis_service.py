"""
分析服务层 (Analysis Service)
把“构造初始状态 → 召回长期记忆 → 带超时地运行状态图 → 汇总响应 → 写入记忆库”这条链路集中在一处，
同步接口 (/analyze) 与流式接口 (/analyze/stream) 共用同一份实现，避免两套代码各自漂移
(旧版本两处各写一遍，正是 SSE 丢辩论历史、读错指标字段名的根源)。
路由层只负责 HTTP 协议细节，业务逻辑全部在这里，因此可以脱离 FastAPI 单独测试。
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.core.observability import start_trace
from app.core.resilience import IdempotencyCache
from app.graph.financial_graph import financial_graph
from app.graph.state_utils import merge_state
from app.memory.store import MemoryStore
from app.models.schemas import AnalysisRequest, AnalysisResponse
from app.tools.market_data import DataUnavailableError

logger = logging.getLogger(__name__)

NODE_DESC = {
    "init_data_node": "底数据加载与K线提取",
    "macro_node": "宏观经济与政策周期研判",
    "fundamental_node": "基本面三表穿透与杜邦/DCF计算",
    "technical_node": "量化技术形态与动量计算",
    "experts_join_node": "三位专家结论汇合",
    "bull_node": "多头首席辩手陈词",
    "bear_node": "空头首席辩手反驳",
    "arbiter_node": "投委会多空分歧仲裁",
    "risk_node": "合规风控硬沙箱与一票否决审核",
    "reporter_node": "机构级投研研报排版归档",
}


class AnalysisTimeoutError(RuntimeError):
    pass


def build_initial_state(req: AnalysisRequest, run_id: str, history_runs: list) -> Dict[str, Any]:
    return {
        "ticker": req.ticker.strip().upper(),
        "target_name": req.target_name,
        "custom_query": req.custom_query,
        "enable_debate": req.enable_debate,
        "strategy_style": req.strategy_style,
        "fundamental_weight": req.fundamental_weight,
        "technical_weight": req.technical_weight,
        "macro_weight": req.macro_weight,
        "capital_flow_weight": req.capital_flow_weight,
        "execution_logs": [],
        "debate_turns": [],
        "degraded_nodes": [],
        "run_id": run_id,
        "history_runs": history_runs,
    }


def build_data_quality(state: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    quote, fin = state.get("quote"), state.get("financials")
    llm = trace.get("llm", {})
    mode = llm_provider.status()["mode"]
    fallback_calls = llm.get("fallbacks", 0)
    divergences = []
    for key, label in (("fundamental_output", "基本面"), ("technical_output", "技术面"), ("macro_output", "宏观")):
        o = state.get(key)
        if o is not None and o.llm_score is not None and o.quant_score is not None \
                and abs(o.llm_score - o.quant_score) >= settings.decision.LLM_DIVERGENCE_THRESHOLD:
            divergences.append(f"{label}: 量化{o.quant_score} vs LLM{o.llm_score}")
    intel = state.get("market_intel")
    macro = state.get("macro_output")
    idx_src = (macro.details or {}).get("index_data_source") if macro is not None else None
    return {
        "macro_index_source": idx_src or "unavailable",
        "decision_basis": "weights × deterministic quant scores (+ bounded debate adjustment); LLM only narrates",
        "llm_quant_divergence": divergences,
        "missing_dimensions": [k for k, v in (state.get("dimension_scores") or {}).items() if v is None],
        "capital_flow_source": getattr(intel, "data_source", "unknown"),
        "quote_source": getattr(quote, "data_source", "unknown"),
        "financials_source": getattr(fin, "data_source", "unknown"),
        "estimated_fields": list(getattr(fin, "estimated_fields", []) or []),
        "llm_mode": mode,
        "llm_calls": llm.get("calls", 0),
        "llm_fallback_calls": fallback_calls,
        "degraded_nodes": trace.get("degraded_nodes", []),
        "is_demo_data": (getattr(quote, "data_source", "real") == "simulated"
                         or getattr(fin, "data_source", "real") != "real"
                         or (idx_src is not None and idx_src != "real")),
        "risk_skipped_checks": list(getattr(state.get("risk_audit"), "skipped_checks", []) or []),
        "parallel_experts": settings.eng.PARALLEL_EXPERTS,
    }


def build_response(state: Dict[str, Any], run_id: str, session_id: str, elapsed: float,
                   trace: Dict[str, Any]) -> AnalysisResponse:
    arbiter, audit = state.get("arbiter_verdict"), state.get("risk_audit")
    quote, macro = state.get("quote"), state.get("macro_output")
    fund, tech = state.get("fundamental_output"), state.get("technical_output")
    dcf, var_res = state.get("dcf_result"), state.get("var_result")
    memory = {}
    hist = state.get("history_runs") or []
    if hist and arbiter is not None:
        last = hist[0]
        memory = {
            "previous_run_id": last.get("run_id"),
            "previous_verdict": last.get("verdict"),
            "previous_consensus_score": last.get("consensus_score"),
            "consensus_score_delta": (round(arbiter.consensus_score - last["consensus_score"], 2)
                                      if isinstance(last.get("consensus_score"), (int, float)) else None),
            "history_count": len(hist),
        }
    return AnalysisResponse(
        ticker=state.get("ticker", ""),
        target_name=state.get("target_name", state.get("ticker", "")),
        current_price=quote.current_price if quote else 0.0,
        consensus_score=arbiter.consensus_score if arbiter else 0.0,
        verdict=arbiter.final_verdict if arbiter else "无结论",
        target_position=arbiter.target_position_pct if arbiter else "0%",
        risk_status=audit.compliance_status if audit else "UNKNOWN",
        is_vetoed=audit.is_risk_violated if audit else False,
        # 专家缺失(节点降级)时返回 0 分而不是编一个 7.0，调用方可结合 data_quality.degraded_nodes 判断
        macro_score=macro.score if macro else 0.0,
        fundamental_score=fund.score if fund else 0.0,
        technical_score=tech.score if tech else 0.0,
        dcf_margin_of_safety=dcf.margin_of_safety_pct if dcf else 0.0,
        var_95_drawdown=var_res.var_percentage if var_res else 0.0,
        full_report_md=state.get("final_report_md", ""),
        debate_history=state.get("debate_turns", []),
        arbiter_verdict=arbiter,
        score_breakdown=list(state.get("score_breakdown") or []),
        audit_details=audit,
        market_intel=state.get("market_intel"),
        strategy_style=state.get("strategy_style", "balanced"),
        weights_used={
            "fundamental": state.get("fundamental_weight", 0.40),
            "technical": state.get("technical_weight", 0.35),
            "macro": state.get("macro_weight", 0.15),
            "capital_flow": state.get("capital_flow_weight", 0.10),
        },
        execution_time_seconds=elapsed,
        run_id=run_id,
        session_id=session_id,
        data_quality=build_data_quality(state, trace),
        trace=trace,
        memory=memory,
    )


def response_payload(resp: AnalysisResponse, state: Dict[str, Any]) -> Dict[str, Any]:
    """响应 + 前端图表用到的扩展字段(杜邦 / DCF / 技术指标)。同步与流式接口共用，字段名只在这里写一遍"""
    data = resp.model_dump()
    dupont, dcf = state.get("dupont_result"), state.get("dcf_result")
    ind, var_res = state.get("indicators_result"), state.get("var_result")
    if dupont:
        data["dupont_roe"] = dupont.calculated_roe
        data["net_margin"] = dupont.net_profit_margin
    if dcf:
        data["dcf_value"] = dcf.intrinsic_value_per_share
    if ind:
        data.update(rsi_14=ind.rsi_14, macd_bar=ind.macd_bar, trend_signal=ind.trend_signal,
                    support_level=ind.support_level, resistance_level=ind.resistance_level)
    if var_res:
        data["var_absolute_loss"] = var_res.var_absolute_loss
    return data


class AnalysisService:
    def __init__(self, store: MemoryStore, graph=None, idempotency: Optional[IdempotencyCache] = None):
        self.store = store
        self.graph = graph or financial_graph
        self.idem = idempotency or IdempotencyCache(settings.eng.IDEMPOTENCY_TTL_SECONDS)

    async def stream(self, req: AnalysisRequest, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """逐节点产出事件：started → node_update* → completed；失败产出 error 事件"""
        t0 = time.time()
        run_id = uuid.uuid4().hex[:16]
        session_id = session_id or uuid.uuid4().hex[:12]
        ticker = req.ticker.strip().upper()
        trace = start_trace(ticker=ticker, run_id=run_id)

        try:
            history = await asyncio.to_thread(self.store.recent_runs, ticker, 3)
        except Exception as e:  # 记忆库故障不应阻断分析
            logger.warning("memory recall failed: %s", e)
            history = []

        state = build_initial_state(req, run_id, history)
        yield {"event": "started", "ticker": ticker, "run_id": run_id, "session_id": session_id,
               "recalled_history": len(history),
               "message": f"启动标的 {ticker} 投委会多Agent全流程分析..."}

        deadline = asyncio.get_running_loop().time() + settings.eng.ANALYSIS_TIMEOUT_SECONDS
        it = self.graph.astream(dict(state)).__aiter__()
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise AnalysisTimeoutError(f"分析超过 {settings.eng.ANALYSIS_TIMEOUT_SECONDS:.0f}s 总超时")
                try:
                    chunk = await asyncio.wait_for(it.__anext__(), remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise AnalysisTimeoutError(f"分析超过 {settings.eng.ANALYSIS_TIMEOUT_SECONDS:.0f}s 总超时")
                for node, upd in chunk.items():
                    merge_state(state, upd or {})
                    payload = {"event": "node_update", "node": node, "description": NODE_DESC.get(node, node),
                               "step": (upd or {}).get("current_step", "")}
                    for k_src, k_dst in (("bull_score", "bull_score"), ("bull_argument", "bull_argument"),
                                         ("bear_score", "bear_score"), ("bear_argument", "bear_argument"),
                                         ("current_debate_round", "debate_round")):
                        if upd and k_src in upd:
                            payload[k_dst] = upd[k_src]
                    if upd and upd.get("arbiter_verdict") is not None:
                        payload["verdict"] = upd["arbiter_verdict"].final_verdict
                        payload["consensus_score"] = upd["arbiter_verdict"].consensus_score
                    yield payload
        except Exception as e:  # noqa: BLE001
            logger.exception("analysis failed run_id=%s", run_id)
            yield {"event": "error", "run_id": run_id, "error_type": type(e).__name__,
                   "message": f"分析失败: {e}", "trace": trace.to_dict()}
            return
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass

        state["execution_logs"] = sorted(state.get("execution_logs", []), key=lambda x: x.get("timestamp", ""))
        elapsed = round(time.time() - t0, 2)
        trace_dict = trace.to_dict()
        resp = build_response(state, run_id, session_id, elapsed, trace_dict)

        arb = resp.arbiter_verdict
        summary = (f"{resp.verdict}｜共识分 {resp.consensus_score}｜宏观 {resp.macro_score} 基本面 {resp.fundamental_score} "
                   f"技术 {resp.technical_score}｜DCF 安全边际 {resp.dcf_margin_of_safety}%｜VaR95 {resp.var_95_drawdown}%"
                   f"｜风控 {resp.risk_status}" + (f"｜{arb.arbitration_summary[:200]}" if arb else ""))
        try:
            await asyncio.to_thread(self.store.save_run, {
                "run_id": run_id, "session_id": session_id, "ticker": ticker, "target_name": resp.target_name,
                "created_at": time.time(), "strategy_style": resp.strategy_style, "verdict": resp.verdict,
                "consensus_score": resp.consensus_score, "position": resp.target_position,
                "risk_status": resp.risk_status, "is_vetoed": resp.is_vetoed, "data_quality": resp.data_quality,
                "summary": summary, "report_md": resp.full_report_md, "trace": trace_dict,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning("memory save failed: %s", e)

        yield {"event": "completed", "elapsed": elapsed, "data": response_payload(resp, state)}

    async def run(self, req: AnalysisRequest, session_id: Optional[str] = None) -> Dict[str, Any]:
        """同步模式：消费 stream，返回 completed 事件里的 data；失败则抛异常"""
        async for ev in self.stream(req, session_id):
            if ev["event"] == "completed":
                return ev["data"]
            if ev["event"] == "error":
                if ev.get("error_type") == "AnalysisTimeoutError":
                    raise AnalysisTimeoutError(ev["message"])
                if ev.get("error_type") == "DataUnavailableError":
                    raise DataUnavailableError(ev["message"])
                raise RuntimeError(ev["message"])
        raise RuntimeError("分析未产生结果")

    async def run_idempotent(self, key: str, req: AnalysisRequest, session_id: Optional[str] = None):
        """相同 Idempotency-Key 的重复提交只执行一次。返回 (data, 是否命中缓存)"""
        return await self.idem.run(f"{key}:{req.model_dump_json()}", lambda: self.run(req, session_id))

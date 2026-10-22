import asyncio

from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.graph.financial_graph import financial_graph
from app.memory.store import MemoryStore
from app.services.analysis_service import AnalysisService, AnalysisTimeoutError
from app.tools.market_data import DataUnavailableError
from tests.helpers import override, req, run


def _svc():
    return AnalysisService(MemoryStore(":memory:"))


async def _events(svc, request):
    return [ev async for ev in svc.stream(request)]


def test_stream_event_sequence_and_completed_payload_is_complete():
    evs = run(_events(_svc(), req()))
    kinds = [e["event"] for e in evs]
    assert kinds[0] == "started" and kinds[-1] == "completed" and "error" not in kinds
    nodes = [e["node"] for e in evs if e["event"] == "node_update"]
    assert nodes[0] == "init_data_node" and "risk_node" in nodes and nodes[-1] == "reporter_node"
    data = evs[-1]["data"]
    # 回归 1：旧版 SSE 用 dict.update 累积状态，会覆盖 reducer 字段，辩论历史只剩最后一轮
    assert len(data["debate_history"]) == max(1, len([n for n in nodes if n == "bull_node"]))
    # 回归 2：旧版读的是不存在的 state["quant_indicators"]，RSI/MACD 等字段永远缺失
    assert "rsi_14" in data and "macd_bar" in data and "dcf_value" in data and "dupont_roe" in data


def test_sync_and_stream_return_the_same_shape():
    svc = _svc()
    sync = run(svc.run(req()))
    streamed = run(_events(svc, req()))[-1]["data"]
    assert set(sync) == set(streamed)
    assert sync["consensus_score"] == streamed["consensus_score"]


def test_response_carries_data_quality_and_trace():
    data = run(_svc().run(req()))
    dq = data["data_quality"]
    assert dq["is_demo_data"] and dq["quote_source"] == "simulated" and dq["llm_mode"] == "mock"
    assert dq["llm_calls"] > 0 and dq["llm_fallback_calls"] == dq["llm_calls"]      # 离线时每次 LLM 调用都是降级
    assert "regulatory_scrutiny" in dq["risk_skipped_checks"]
    spans = {s["name"]: s for s in data["trace"]["spans"]}
    assert {"init_data_node", "macro_node", "fundamental_node", "technical_node", "risk_node", "reporter_node"} <= set(spans)
    assert all(s["status"] == "ok" and s["duration_ms"] is not None for s in spans.values())
    assert data["full_report_md"].startswith("> ⚠️ **数据质量提示**")
    assert data["run_id"] and data["session_id"]


def test_second_analysis_recalls_first_from_memory_and_persists_trace():
    svc = _svc()
    first = run(svc.run(req(strategy_style="balanced")))
    assert first["memory"] == {} and "首次分析" in first["full_report_md"]
    second = run(svc.run(req(strategy_style="momentum")))
    m = second["memory"]
    assert m["previous_run_id"] == first["run_id"] and m["history_count"] == 1
    assert "历史记忆对比" in second["full_report_md"] and "首次分析" not in second["full_report_md"]
    saved = svc.store.get_run(second["run_id"])
    assert saved["trace"]["spans"] and saved["ticker"] == "600519" and "# " in saved["report_md"] or saved["report_md"]
    assert len(svc.store.list_runs("600519")) == 2


def test_memory_is_scoped_per_ticker():
    svc = _svc()
    run(svc.run(req(ticker="600519")))
    other = run(svc.run(req(ticker="000001")))
    assert other["memory"] == {}


def test_idempotent_submission_executes_graph_once():
    svc = _svc()
    count = {"n": 0}
    orig = svc.graph.astream

    def counting(state):
        count["n"] += 1
        return orig(state)

    svc.graph = type("G", (), {"astream": staticmethod(counting)})()

    async def main():
        r = await asyncio.gather(*[svc.run_idempotent("key-1", req()) for _ in range(4)])
        return r

    results = run(main())
    assert count["n"] == 1 and sum(1 for _, replay in results if replay) == 3
    assert len({d["run_id"] for d, _ in results}) == 1


def test_total_timeout_aborts_and_reports_error():
    orig = llm_provider.chat

    async def very_slow(*a, **k):
        await asyncio.sleep(5)
        return "x"

    llm_provider.chat = very_slow
    try:
        with override(settings.eng, ANALYSIS_TIMEOUT_SECONDS=0.3):
            try:
                run(_svc().run(req()))
            except AnalysisTimeoutError:
                return
    finally:
        llm_provider.chat = orig
    raise AssertionError("应触发总超时")


def test_strict_real_data_surfaces_as_data_unavailable():
    with override(settings.eng, STRICT_REAL_DATA=True):
        try:
            run(_svc().run(req()))
        except DataUnavailableError:
            return
    raise AssertionError("严格模式下离线应报 DataUnavailableError")


def test_memory_failure_does_not_break_analysis():
    svc = _svc()
    svc.store.recent_runs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down"))
    svc.store.save_run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down"))
    data = run(svc.run(req()))
    assert data["verdict"]

import asyncio
import time
from contextlib import contextmanager

from app.agents.macro_agent import macro_agent
from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.core.observability import start_trace
from app.graph.financial_graph import build_financial_graph
from app.graph.state_utils import merge_state
from tests.helpers import make_fin, make_quote, override, run

EXPERTS = {"macro_node", "fundamental_node", "technical_node"}


def _state(**kw):
    return {"ticker": "600519", "enable_debate": True, "strategy_style": "balanced", "execution_logs": [],
            "debate_turns": [], "degraded_nodes": [], **kw}


async def _collect(graph, state):
    start_trace(state["ticker"])
    order, final = [], dict(state)
    async for chunk in graph.astream(dict(state)):
        for node, upd in chunk.items():
            order.append(node)
            merge_state(final, upd)
    return order, final


@contextmanager
def slow_llm(delay):
    orig = llm_provider.chat

    async def slow(*a, **k):
        await asyncio.sleep(delay)
        return await orig(*a, **k)

    llm_provider.chat = slow
    try:
        yield
    finally:
        llm_provider.chat = orig


def test_parallel_topology_fans_out_and_joins_before_debate():
    order, final = run(_collect(build_financial_graph(parallel=True), _state()))
    assert order[0] == "init_data_node"
    assert set(order[1:4]) == EXPERTS                         # 三个专家紧跟 init，同一超步
    assert order[4] == "experts_join_node" and order[5] == "bull_node"
    assert final["current_step"] == "reporter_node" or final["final_report_md"]


def test_serial_topology_keeps_fixed_order():
    order, _ = run(_collect(build_financial_graph(parallel=False), _state()))
    assert order[:5] == ["init_data_node", "macro_node", "fundamental_node", "technical_node", "experts_join_node"]


def test_parallel_is_faster_than_serial_when_llm_has_latency():
    async def timed(parallel):
        g = build_financial_graph(parallel=parallel)
        t = time.perf_counter()
        await _collect(g, _state(enable_debate=False))
        return time.perf_counter() - t

    with slow_llm(0.15):
        serial = run(timed(False))
        parallel = run(timed(True))
    # 专家阶段 3 次 LLM 调用：串行约 0.45s，并行约 0.15s
    assert parallel < serial - 0.2, (parallel, serial)


def test_parallel_and_serial_produce_the_same_scores():
    _, p = run(_collect(build_financial_graph(parallel=True), _state()))
    _, s = run(_collect(build_financial_graph(parallel=False), _state()))
    for k in ("macro_output", "fundamental_output", "technical_output"):
        assert p[k].score == s[k].score
    assert p["arbiter_verdict"].consensus_score == s["arbiter_verdict"].consensus_score


def test_disable_debate_skips_debate_loop_entirely():
    # 回归：旧版本关闭辩论后仲裁节点会因“未收敛且轮次不足”跳回多头节点
    order, final = run(_collect(build_financial_graph(), _state(enable_debate=False)))
    assert "bull_node" not in order and "bear_node" not in order
    assert order.count("arbiter_node") == 1 and not final["debate_turns"]
    # 关闭辩论：最终分 = 权重 × 各维度量化评分的加权基础分，不做任何辩论调整
    v = final["arbiter_verdict"]
    assert v.debate_adjustment == 0.0 and v.consensus_score == round(final["base_score"], 2)


def test_debate_round_cap_is_enforced():
    with override(settings.debate, CONVERGENCE_DELTA=0.0, MAX_DEBATE_ROUNDS=3):
        order, final = run(_collect(build_financial_graph(), _state()))
    assert order.count("bull_node") <= 3 and len(final["debate_turns"]) <= 3


def test_debate_history_is_accumulated_by_reducer():
    with override(settings.debate, CONVERGENCE_DELTA=0.0, MAX_DEBATE_ROUNDS=2):
        _, final = run(_collect(build_financial_graph(), _state()))
    assert [t.round_index for t in final["debate_turns"]] == [1, 2]


def test_hard_veto_overrides_verdict_through_full_graph():
    st = _state(quote=make_quote(source="real"), financials=make_fin(debt=88.0))
    _, final = run(_collect(build_financial_graph(), st))
    assert final["risk_audit"].compliance_status == "VETOED"
    assert "一票否决" in final["arbiter_verdict"].final_verdict and final["arbiter_verdict"].pre_veto_verdict


def test_expert_failure_degrades_instead_of_crashing_and_is_reported():
    orig = macro_agent.run

    async def boom(state):
        raise RuntimeError("macro exploded")

    macro_agent.run = boom
    try:
        order, final = run(_collect(build_financial_graph(), _state()))
    finally:
        macro_agent.run = orig
    assert final["macro_output"] is None and "macro_node" in final["degraded_nodes"]
    assert final["arbiter_verdict"] is not None and "数据质量提示" in final["final_report_md"]
    assert any(l.get("status") == "degraded" for l in final["execution_logs"])


def test_concurrent_experts_do_not_write_conflicting_keys():
    # 并行时若专家再写 quote/current_step 这类无 reducer 的共享 key，LangGraph 会抛 InvalidUpdateError
    from app.graph.financial_graph import _only_own_keys
    assert _only_own_keys({"quote": 1, "financials": 2, "current_step": "x", "macro_output": 3}) == {"macro_output": 3}


def test_merge_state_uses_reducers_not_overwrite():
    s = {"execution_logs": [{"a": 1}], "debate_turns": [], "x": 1}
    merge_state(s, {"execution_logs": [{"b": 2}], "x": 2})
    merge_state(s, {"execution_logs": [{"c": 3}]})
    assert [list(l)[0] for l in s["execution_logs"]] == ["a", "b", "c"] and s["x"] == 2


def test_router_never_returns_to_debate_when_debate_disabled():
    from app.graph.financial_graph import route_after_arbiter
    s = {"enable_debate": False, "is_debate_converged": False, "current_debate_round": 0, "max_debate_rounds": 2}
    assert route_after_arbiter(s) == "risk_node"
    assert route_after_arbiter({**s, "enable_debate": True}) == "bull_node"
    assert route_after_arbiter({**s, "enable_debate": True, "current_debate_round": 2}) == "risk_node"
    assert route_after_arbiter({**s, "enable_debate": True, "is_debate_converged": True}) == "risk_node"

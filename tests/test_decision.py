"""确定性决策：用户权重必须驱动结论，与 LLM 是否在线无关"""
import json

from app.core import llm_provider as lp
from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.core.observability import start_trace
from app.graph.financial_graph import build_financial_graph
from app.graph.state_utils import merge_state
from app.models.schemas import QuantIndicatorsResult
from app.tools.quant_indicators import compute_atr, compute_mfi, compute_obv_slope, compute_realized_vol
from app.tools.scoring import (
    aggregate, atr_risk_sizing, debate_adjustment, interp, score_market_environment, score_technical,
    verdict_from_score,
)
from tests.helpers import override, run


async def _run(debate=False, **weights):
    state = {"ticker": "600519", "enable_debate": debate, "strategy_style": "custom", "execution_logs": [],
             "debate_turns": [], "degraded_nodes": [], **weights}
    start_trace("600519")
    final = dict(state)
    async for chunk in build_financial_graph().astream(dict(state)):
        for _, upd in chunk.items():
            merge_state(final, upd)
    return final


W = dict(fundamental_weight=0.0, technical_weight=0.0, macro_weight=0.0, capital_flow_weight=0.0)


def _only(dim):
    key = {"fundamental": "fundamental_weight", "technical": "technical_weight", "macro": "macro_weight",
           "capital_flow": "capital_flow_weight"}[dim]
    return {**W, key: 1.0}


def test_weights_drive_the_decision_offline():
    f = run(_run(**_only("fundamental")))
    t = run(_run(**_only("technical")))
    assert f["base_score"] == f["fundamental_output"].quant_score
    assert t["base_score"] == t["technical_output"].quant_score
    assert f["base_score"] != t["base_score"]                      # 换权重 → 换分数
    assert f["arbiter_verdict"].consensus_score == f["base_score"]  # 关闭辩论时最终分就是基础分


class _FakeLive:
    """把 llm_provider.chat 换成“真实在线”的假模型：返回荒谬的评分，验证决策不跟着它走"""
    def __init__(self, score):
        self.score = score

    async def chat(self, messages, **kw):
        lp._LAST_CALL_LIVE.set(True)
        return json.dumps({"stance": "x", "score": self.score, "conviction_score": self.score,
                           "summary": "LLM 在线解读", "details": {}, "key_points": ["k"]}, ensure_ascii=False)


def _with_fake_live(score, fn):
    old = llm_provider.chat
    llm_provider.chat = _FakeLive(score).chat
    try:
        return fn()
    finally:
        llm_provider.chat = old


def test_absurd_live_llm_score_does_not_change_decision():
    offline = run(_run(**_only("technical")))
    live_hi = _with_fake_live(10.0, lambda: run(_run(**_only("technical"))))
    live_lo = _with_fake_live(0.0, lambda: run(_run(**_only("technical"))))
    for live in (live_hi, live_lo):
        assert live["base_score"] == offline["base_score"]
        assert live["arbiter_verdict"].consensus_score == offline["arbiter_verdict"].consensus_score
        o = live["technical_output"]
        assert o.quant_score == offline["technical_output"].quant_score
        assert o.summary == "LLM 在线解读" and o.score_source == "quant+llm_narrative"
    assert live_hi["technical_output"].llm_score == 10.0 and live_lo["technical_output"].llm_score == 0.0


def test_live_llm_weight_change_still_changes_decision():
    a = _with_fake_live(5.0, lambda: run(_run(**_only("fundamental"))))
    b = _with_fake_live(5.0, lambda: run(_run(**_only("technical"))))
    assert a["base_score"] != b["base_score"]


def test_offline_expert_text_is_template_not_llm():
    final = run(_run(**_only("fundamental")))
    for k in ("fundamental_output", "technical_output", "macro_output"):
        o = final[k]
        assert o.llm_score is None and o.score_source == "quant+template_narrative"


def test_debate_adjustment_is_bounded_and_weights_dominate():
    with override(settings.decision, DEBATE_MAX_ADJUST=0.8):
        assert debate_adjustment(10, 0) == 0.8 and debate_adjustment(0, 10) == -0.8
        assert debate_adjustment(5, 5) == 0.0
    final = run(_run(debate=True, **_only("technical")))
    v = final["arbiter_verdict"]
    assert abs(v.debate_adjustment) <= settings.decision.DEBATE_MAX_ADJUST
    assert abs(v.consensus_score - v.base_score) <= settings.decision.DEBATE_MAX_ADJUST + 0.01


def test_missing_dimension_is_dropped_and_weights_renormalized():
    # 离线时资金流数据源不可用 → 该维度不参与加权，其余权重按比例放大，而不是编一个默认分
    final = run(_run(fundamental_weight=0.4, technical_weight=0.4, macro_weight=0.1, capital_flow_weight=0.1))
    assert final["dimension_scores"]["capital_flow"] is None
    eff = final["effective_weights"]
    assert "capital_flow" not in eff and abs(sum(eff.values()) - 1.0) < 1e-9
    flow_row = [b for b in final["score_breakdown"] if b["dimension"] == "capital_flow"][0]
    assert flow_row["score"] is None and flow_row["contribution"] == 0.0


def test_aggregate_arithmetic():
    r = aggregate({"fundamental": 8.0, "technical": 4.0, "macro": None, "capital_flow": None},
                  {"fundamental": 0.3, "technical": 0.1, "macro": 0.3, "capital_flow": 0.3})
    assert abs(r["effective_weights"]["fundamental"] - 0.75) < 1e-9
    assert r["base_score"] == round(8 * 0.75 + 4 * 0.25, 2) and set(r["missing"]) == {"macro", "capital_flow"}
    assert aggregate({k: None for k in ("fundamental", "technical", "macro", "capital_flow")}, {})["base_score"] == 5.0


def test_interp_is_monotonic_and_clamped():
    pts = [(0, 1), (10, 5), (20, 9)]
    assert interp(-5, pts) == 1 and interp(99, pts) == 9 and interp(5, pts) == 3
    vals = [interp(x, pts) for x in range(-5, 30)]
    assert vals == sorted(vals)


def test_verdict_bands_are_monotonic():
    order = ["建议规避", "中性持有", "逢低增配", "强烈买入"]
    seen = [next(i for i, k in enumerate(order) if k in verdict_from_score(x / 10)[0]) for x in range(0, 101)]
    assert seen == sorted(seen) and seen[0] == 0 and seen[-1] == 3


def _ind(**kw):
    base = dict(macd_dif=0.0, macd_dea=0.0, macd_bar=0.0, boll_upper=110.0, boll_middle=100.0, boll_lower=90.0,
                trend_signal='t', support_level=95.0, resistance_level=105.0, last_close=100.0, ma20=100.0, ma60=100.0, rsi_14=50.0, realized_vol_20=20.0, vol_regime=1.0,
                volume_ratio=1.0, obv_slope=0.0, mfi_14=50.0, ret_5d=0.0, ret_20d=0.0, atr_14=2.0)
    base.update(kw)
    return QuantIndicatorsResult(**base)


def test_technical_score_reacts_to_new_indicators_in_expected_direction():
    n = score_technical(_ind())[0]
    assert 4.5 <= n <= 5.5                                                   # 中性输入 ≈ 5
    assert score_technical(_ind(realized_vol_20=70, vol_regime=2.5))[0] < n  # 高且放大的波动 → 扣分
    assert score_technical(_ind(volume_ratio=2.0, ret_5d=3.0))[0] > score_technical(_ind(volume_ratio=2.0, ret_5d=-3.0))[0]
    assert score_technical(_ind(obv_slope=0.3))[0] > score_technical(_ind(obv_slope=-0.3))[0]
    assert score_technical(_ind(rsi_14=88, ret_5d=14))[0] < n                # 短期过热 → 回归倾向
    assert 0 <= score_technical(_ind(rsi_14=99, ret_5d=99, realized_vol_20=999))[0] <= 10


def test_market_environment_score_uptrend_beats_downtrend():
    up = score_market_environment(_ind(ma20=105, ma60=100, last_close=108, ret_20d=6, macd_dif=1, macd_dea=0.5))[0]
    dn = score_market_environment(_ind(ma20=95, ma60=100, last_close=92, ret_20d=-6, macd_dif=-1, macd_dea=-0.5))[0]
    assert up > dn


def test_atr_stop_and_position_cap():
    r = atr_risk_sizing(_ind(last_close=100, atr_14=2.0))
    assert r["atr_stop_price"] == 96.0 and r["atr_stop_pct"] == 4.0 and r["risk_cap_position_pct"] == 25.0
    assert atr_risk_sizing(_ind(atr_14=0))["atr_stop_price"] is None


def test_indicator_helpers_hand_computed():
    h, l, c = [10, 11, 12, 13], [9, 10, 11, 12], [9.5, 10.5, 11.5, 12.5]
    assert compute_atr(h, l, c, period=2) > 0
    assert compute_realized_vol([100.0] * 30, 20) == 0.0
    assert compute_obv_slope([1, 2, 3, 4, 5], [10, 10, 10, 10, 10], lookback=2) > 0
    hh, ll, cc = list(range(11, 19)), list(range(9, 17)), list(range(10, 18))
    assert compute_mfi(hh, ll, cc, [100] * 8, period=4) == 100.0     # 只有正向资金流


def test_mixed_weights_produce_the_exact_weighted_average():
    final = run(_run(fundamental_weight=0.7, technical_weight=0.3, macro_weight=0.0, capital_flow_weight=0.0))
    f, t = final["fundamental_output"].quant_score, final["technical_output"].quant_score
    assert final["base_score"] == round(0.7 * f + 0.3 * t, 2)
    flipped = run(_run(fundamental_weight=0.3, technical_weight=0.7, macro_weight=0.0, capital_flow_weight=0.0))
    assert flipped["base_score"] == round(0.3 * f + 0.7 * t, 2) and flipped["base_score"] != final["base_score"]


def test_simulated_index_is_surfaced_in_data_quality_and_banner():
    from app.services.analysis_service import build_data_quality
    from app.graph.financial_graph import render_data_banner
    final = run(_run(**_only("macro")))
    dq = build_data_quality(final, {"llm": {}, "degraded_nodes": []})
    assert dq["macro_index_source"] == "simulated" and dq["is_demo_data"]
    assert "沪深300" in render_data_banner(final)

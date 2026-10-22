import numpy as np

import app.tools.market_data as md
from app.agents.risk_guardrail import risk_guardrail_agent
from app.core.config import settings
from app.models.schemas import ArbiterVerdict, MonteCarloVaRResult
from app.tools.market_data import DataUnavailableError, market_data_provider
from app.tools.risk_calculator import risk_calculator_engine
from tests.helpers import make_fin, make_quote, override, run


def _arbiter():
    return ArbiterVerdict(final_verdict="强烈买入", consensus_score=8.5, target_position_pct="60%",
                          arbitration_summary="s", entry_tactics="t")


def _var(pct):
    return MonteCarloVaRResult(simulations=10000, confidence_level=0.95, horizon_days=5, var_percentage=pct,
                               var_absolute_loss=1.0, worst_drawdown=pct * 2, risk_category="R1")


def _audit(fin, name="测试公司", var_pct=5.0, quote=None):
    state = {"ticker": "600000", "target_name": name, "quote": quote or make_quote(), "financials": fin,
             "var_result": _var(var_pct), "arbiter_verdict": _arbiter()}
    return run(risk_guardrail_agent.run(state)), state


def test_var_deterministic_and_does_not_touch_global_rng():
    q = make_quote()
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    v1 = risk_calculator_engine.calculate_monte_carlo_var(q)
    v2 = risk_calculator_engine.calculate_monte_carlo_var(q)
    assert np.random.rand() == expected          # 全局随机状态没被污染
    assert v1.var_percentage == v2.var_percentage and v1.simulations == 10000


def test_synthetic_data_differs_by_ticker_and_is_flagged():
    a = market_data_provider._get_synthetic_market_quote("600519")
    b = market_data_provider._get_synthetic_market_quote("000001")
    assert a.data_source == b.data_source == "simulated"
    va = risk_calculator_engine.calculate_monte_carlo_var(a).var_percentage
    vb = risk_calculator_engine.calculate_monte_carlo_var(b).var_percentage
    assert va != vb                               # 旧版固定正弦波会让所有标的 VaR 相同
    again = market_data_provider._get_synthetic_market_quote("600519")
    assert [k.close for k in a.kline_history] == [k.close for k in again.kline_history]   # 同标的可复现


def test_offline_fallback_marks_simulated_and_strict_mode_raises():
    q = market_data_provider.get_market_quote("600519")
    f = market_data_provider.get_financial_statements("600519")
    assert q.data_source == "simulated" and f.data_source == "simulated" and f.estimated_fields
    with override(settings.eng, STRICT_REAL_DATA=True):
        for call in (market_data_provider.get_market_quote, market_data_provider.get_financial_statements):
            try:
                call("600519")
            except DataUnavailableError:
                continue
            raise AssertionError("严格模式下应抛 DataUnavailableError")


def test_veto_on_high_debt_keeps_pre_veto_verdict_and_does_not_mutate_input():
    out, state = _audit(make_fin(debt=80.0))
    assert out["risk_audit"].compliance_status == "VETOED" and out["is_risk_violated"]
    new = out["arbiter_verdict"]
    assert "一票否决" in new.final_verdict and new.pre_veto_verdict == "强烈买入" and new.pre_veto_position == "60%"
    assert state["arbiter_verdict"].final_verdict == "强烈买入"        # 上游对象没有被原地篡改


def test_financial_institution_uses_wider_debt_limit():
    ok, _ = _audit(make_fin(debt=90.0), name="某某银行")
    assert not ok["is_risk_violated"]
    bad, _ = _audit(make_fin(debt=95.0), name="某某银行")
    assert bad["is_risk_violated"]


def test_estimated_ebit_never_triggers_interest_coverage_veto():
    est, _ = _audit(make_fin(debt=40.0, ebit=1.0, interest=5.0, estimated=["ebit", "interest_expense"]))
    assert not est["is_risk_violated"] and "interest_coverage" in est["risk_audit"].skipped_checks
    real, _ = _audit(make_fin(debt=40.0, ebit=1.0, interest=5.0))       # 披露值 → 覆盖倍数 0.2x → 应否决
    assert real["is_risk_violated"]


def test_var_threshold_veto_and_warning_band():
    assert _audit(make_fin(), var_pct=25.0)[0]["risk_audit"].compliance_status == "VETOED"
    assert _audit(make_fin(), var_pct=14.0)[0]["risk_audit"].compliance_status == "WARNING"
    assert _audit(make_fin(), var_pct=3.0)[0]["risk_audit"].compliance_status == "APPROVED"


def test_regulatory_check_is_honestly_marked_unchecked_and_demo_data_warned():
    out, _ = _audit(make_fin(source="simulated", estimated=["ebit"]), quote=make_quote(source="simulated"))
    audit = out["risk_audit"]
    assert "未检查" in audit.audit_checklist["regulatory_scrutiny"] and "regulatory_scrutiny" in audit.skipped_checks
    assert "模拟数据" in audit.audit_checklist["data_quality"]

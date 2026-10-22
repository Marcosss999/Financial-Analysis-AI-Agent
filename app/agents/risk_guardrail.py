"""
合规风控硬拦截沙箱与一票否决节点 (Risk Guardrail & Hard Constraint Sandbox)
职责：
1. 穿透执行确定性风控硬规则校验 (资产负债率、利息保障倍数、蒙特卡洛 95% VaR 极端下行风险)。
2. 拥有一票否决权 (Veto Power)：若触碰财务破产或极端波动红线，强制拦截买入建议并熔断归档。
3. 规范输出合规审核清单 (Audit Checklist) 与风控备忘录，保障投研建议严守监管与资管底线。
"""

from datetime import datetime
from typing import Dict, Any, List
from app.core.config import settings
from app.agents.base_agent import BaseSpecialistAgent
from app.models.agent_state import FinancialAgentState
from app.models.schemas import RiskAuditResult, MonteCarloVaRResult
from app.tools.market_data import market_data_provider
from app.tools.risk_calculator import risk_calculator_engine


class RiskGuardrailAgent(BaseSpecialistAgent):
    """合规风控与硬拦截沙箱官 Agent"""

    def __init__(self):
        super().__init__(
            name="Risk_Guardrail_Officer",
            role_description="公募/私募基金首席风控合规官，具备独立一票否决权，严守穿透合规与下行安全底线",
            temperature=0.1,
        )

    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        """执行合规穿透审核与硬规则沙箱拦截"""
        ticker = state.get("ticker", "600519")
        target_name = state.get("target_name") or f"标的-{ticker}"
        quote = state.get("quote") or market_data_provider.get_market_quote(ticker)
        financials = state.get("financials") or market_data_provider.get_financial_statements(ticker)

        # 1. 运行确定性 10,000 次蒙特卡洛 VaR 计算
        var_res: MonteCarloVaRResult = state.get("var_result") or risk_calculator_engine.calculate_monte_carlo_var(quote)

        # 2. 确定性硬规则沙箱审查 (Hard Constraints Verification)
        checklist: Dict[str, str] = {}
        veto_reasons: List[str] = []
        is_violated = False

        # 规则 (1): 资产负债率检查 (标准实体企业红线 75%；商业银行等持牌金融机构因吸收存款记为负债，红线适应性设为 94%)
        debt_ratio = financials.debt_to_asset_ratio
        is_financial_institution = any(k in target_name for k in ["银行", "证券", "保险", "信托"])
        max_debt = (
            settings.risk.FINANCIAL_INSTITUTION_MAX_DEBT_PCT
            if is_financial_institution
            else settings.risk.MAX_DEBT_TO_ASSET_RATIO * 100.0
        )
        skipped: List[str] = []
        est = set(getattr(financials, "estimated_fields", []) or [])
        
        if debt_ratio > max_debt:
            inst_note = " (金融持牌机构特例口径)" if is_financial_institution else ""
            checklist["debt_to_asset"] = f"未通过 (当前 {debt_ratio}% > 红线 {max_debt}%{inst_note})"
            veto_reasons.append(f"资产负债率高企达 {debt_ratio}%，超过资管合规风控红线 ({max_debt}%)")
            is_violated = True
        else:
            checklist["debt_to_asset"] = f"通过 (当前 {debt_ratio}% <= 红线 {max_debt}%)"

        # 规则 (2): 利息保障倍数检查 (非金融企业底线 1.5x；持牌金融机构计息负债利息支出为经营成本，免受工业标准限制)
        if not is_financial_institution and ({"ebit", "interest_expense"} & est):
            # EBIT 与利息费用在数据源里是按净利润/负债乘经验系数推算的，不是财报披露值，
            # 拿估算值去做“一票否决”会产生虚假的硬拦截，所以只记录、不判定
            checklist["interest_coverage"] = "未检查 (EBIT/利息费用为估算值，非财报披露，不作为否决依据)"
            skipped.append("interest_coverage")
        elif not is_financial_institution:
            interest = max(0.1, financials.interest_expense)
            interest_coverage = financials.ebit / interest
            min_coverage = settings.risk.MIN_INTEREST_COVERAGE
            if interest_coverage < min_coverage:
                checklist["interest_coverage"] = f"未通过 (当前 {interest_coverage:.1f}x < 红线 {min_coverage}x)"
                veto_reasons.append(f"利息保障倍数仅为 {interest_coverage:.1f}x，偿债覆盖垫严重不足")
                is_violated = True
            else:
                checklist["interest_coverage"] = f"通过 (当前 {interest_coverage:.1f}x >= 红线 {min_coverage}x)"
        else:
            checklist["interest_coverage"] = "通过 (持牌金融机构按净息差/巴塞尔协议监管，工业利息保障倍数豁免)"

        # 规则 (3): 蒙特卡洛单周 95% VaR 风险价值损失检查 (阈值默认 18%)
        max_var = settings.risk.MAX_ACCEPTABLE_VAR_95 * 100.0
        if var_res.var_percentage > max_var:
            checklist["var_95_drawdown"] = f"未通过 (单周预期在险损失 {var_res.var_percentage}% > 红线 {max_var}%)"
            veto_reasons.append(f"蒙特卡洛模拟揭示单周 95% VaR 最大回撤为 {var_res.var_percentage}%，下行波动极度高危")
            is_violated = True
        else:
            checklist["var_95_drawdown"] = f"通过 (单周预期在险损失 {var_res.var_percentage}% <= 红线 {max_var}%)"

        # 规则 (4): 重大监管舆情合规检查 —— 目前没有接入监管数据源，如实标记为未检查，而不是写“通过”
        checklist["regulatory_scrutiny"] = "未检查 (尚未接入监管立案/退市风险数据源)"
        skipped.append("regulatory_scrutiny")

        # 数据质量声明：模拟数据上的风控结论只能用于演示
        if getattr(financials, "data_source", "real") != "real" or getattr(quote, "data_source", "real") == "simulated":
            checklist["data_quality"] = "警告：本次风控基于离线模拟数据，结论仅供演示，不可作为投资依据"

        # 3. 裁定风控状态与评级
        if is_violated and settings.risk.ENABLE_HARD_GUARDRAILS:
            compliance_status = "VETOED"
            risk_level = "R5-极高风险 (Hard Constraint Vetoed)"
        elif debt_ratio > settings.risk.WARN_DEBT_PCT or var_res.var_percentage > settings.risk.WARN_VAR_PCT:
            compliance_status = "WARNING"
            risk_level = "R3-中度平衡风险 (Warning Cleared)"
        else:
            compliance_status = "APPROVED"
            risk_level = var_res.risk_category

        audit_res = RiskAuditResult(
            compliance_status=compliance_status,
            is_risk_violated=is_violated,
            risk_level=risk_level,
            audit_checklist=checklist,
            veto_reasons=veto_reasons,
            skipped_checks=skipped,
        )

        # 4. 若触发一票否决，强制篡改下刷投委会终审结果，保卫客户资本
        updates: Dict[str, Any] = {
            "var_result": var_res,
            "risk_audit": audit_res,
            "is_risk_violated": is_violated,
            "current_step": "risk_guardrail_completed",
        }

        if is_violated and settings.risk.ENABLE_HARD_GUARDRAILS:
            arbiter = state.get("arbiter_verdict")
            if arbiter:
                # 不原地修改上游对象：用 model_copy 生成新对象，并保留否决前的原始裁决供审计追溯
                updates["arbiter_verdict"] = arbiter.model_copy(update={
                    "pre_veto_verdict": arbiter.final_verdict,
                    "pre_veto_position": arbiter.target_position_pct,
                    "final_verdict": "【一票否决】合规风控硬性驳回 (Vetoed)",
                    "target_position_pct": "0% (禁止建仓，执行熔断)",
                    "arbitration_summary": f"因触发合规风控硬红线（{'; '.join(veto_reasons)}），风控委员会依据底线原则实施一票否决！",
                })

        log_item = {
            "node": "risk_guardrail",
            "timestamp": datetime.now().isoformat(),
            "status": "warning" if is_violated else "success",
            "summary": f"合规风控沙箱审查完毕: 状态[{compliance_status}], 风险评级[{risk_level}], 红线触发[{is_violated}]",
        }
        updates["execution_logs"] = [log_item]

        return updates


# 便捷单例
risk_guardrail_agent = RiskGuardrailAgent()

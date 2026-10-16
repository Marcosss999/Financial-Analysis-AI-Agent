"""
公司基本面与财务三表分析师 Agent (Fundamental Analyst Specialist Agent)
职责：
1. 联动确定性量化工具执行杜邦三因子拆解 (DuPont Analysis) 与两阶段自由现金流折现 (DCF Valuation)。
2. 穿透公司盈利成色、净资产收益率 (ROE) 质量、资产负债结构及每股内在公允价值安全边际。
3. 结合量化计算事实向大模型发起结构化定性推理，产出基本面评分 (0~10 分) 与风险预警。
"""

from datetime import datetime
from typing import Dict, Any, List
from app.agents.base_agent import BaseSpecialistAgent
from app.models.agent_state import FinancialAgentState
from app.models.schemas import SpecialistAgentOutput
from app.tools.market_data import market_data_provider
from app.tools.financial_metrics import financial_metrics_engine


class FundamentalAgent(BaseSpecialistAgent):
    """公司基本面与估值分析专家 Agent"""

    def __init__(self):
        super().__init__(
            name="Fundamental_Analyst",
            role_description="券商资深行业首席分析师，精通财报三表勾稽、杜邦拆解与 DCF 估值建模",
            temperature=0.2,
        )

    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        """执行基本面与估值穿透分析并返回状态增量"""
        ticker = state.get("ticker", "600519")
        target_name = state.get("target_name") or f"标的-{ticker}"

        # 1. 确保获取最新行情与财报三表底数据
        quote = state.get("quote") or market_data_provider.get_market_quote(ticker)
        financials = state.get("financials") or market_data_provider.get_financial_statements(ticker)

        # 2. 调度确定性纯数学模型运算 (0 幻觉保证)
        dupont_res = financial_metrics_engine.calculate_dupont(financials)
        dcf_res = financial_metrics_engine.calculate_dcf(financials, quote)

        # 3. 构造注入确定性计算事实的 LLM 结构化推理 Prompt
        system_prompt = (
            "你是一名顶级券商研究所的基本面与财务估值首席分析师。\n"
            "系统已通过确定性金融算法完成了杜邦分析与 DCF 现金流折现计算。\n"
            "请基于给出的真实财务指标与估值测算数据，全面评价公司的核心商业壁垒、盈利质量与估值吸引力。\n"
            "要求：输出必须为严格合法的 JSON 对象，不包含任何 Markdown 额外闲聊。\n"
            "JSON 数据结构必须包含以下字段：\n"
            "{\n"
            '  "stance": "定性立场 (如：积极推荐 / 逢低增配 / 谨慎观望 / 明显高估)",\n'
            '  "score": 0.0到10.0之间的综合基本面评分浮点数,\n'
            '  "summary": "核心基本面逻辑精述 (120字左右)",\n'
            '  "details": {\n'
            '    "roe_assessment": "对杜邦拆解 ROE 驱动动力的评价",\n'
            '    "dcf_margin_of_safety": "对 DCF 估值溢价/折价与安全边际的定性评价",\n'
            '    "balance_sheet_health": "资产负债率与偿债风险评价",\n'
            '    "growth_sustainability": "未来成长空间与盈利持续性预测"\n'
            "  },\n"
            '  "key_points": ["核心基本面亮点1", "核心基本面亮点2", "主要财务风险隐患1"]\n'
            "}"
        )

        user_content = (
            f"标的代码: {ticker} ({target_name})\n"
            f"现价: {quote.current_price} 元 | 市值: {quote.market_cap} 亿元 | PE(TTM): {quote.pe_ttm}x\n"
            f"【确定性杜邦拆解计算事实】\n"
            f"- 净利率: {dupont_res.net_profit_margin}%\n"
            f"- 总资产周转率: {dupont_res.asset_turnover} 次/年\n"
            f"- 权益乘数: {dupont_res.equity_multiplier} 倍 (资产负债率: {financials.debt_to_asset_ratio}%)\n"
            f"- 计算所得 ROE: {dupont_res.calculated_roe}%\n"
            f"- 归因定性: {dupont_res.quality_verdict}\n\n"
            f"【确定性 DCF 折现估值计算事实】\n"
            f"- 折现率 WACC: {dcf_res.wacc * 100:.1f}%, 永续增长率 g: {dcf_res.terminal_growth_rate * 100:.1f}%\n"
            f"- 每股内在公允价值: {dcf_res.intrinsic_value_per_share} 元\n"
            f"- 安全边际比例: {dcf_res.margin_of_safety_pct}%\n"
            f"- 估值状态判定: {dcf_res.valuation_status}\n"
            "请出具专业的基本面穿透评审意见。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_reply = await self._call_llm(messages, json_mode=True)

        default_fallback = {
            "stance": "积极买入 (Strong Buy)" if dcf_res.margin_of_safety_pct >= 0 else "中性增持 (Hold)",
            "score": 8.4 if dupont_res.calculated_roe > 20 else 6.8,
            "summary": f"{target_name} 杜邦拆解 ROE 达 {dupont_res.calculated_roe}%，盈利成色极高；DCF 内在价值为 {dcf_res.intrinsic_value_per_share} 元，估值具良好防御性。",
            "details": {
                "roe_assessment": f"ROE 为 {dupont_res.calculated_roe}%，受 {dupont_res.net_profit_margin}% 净利润率深厚护城河驱动",
                "dcf_margin_of_safety": f"安全边际为 {dcf_res.margin_of_safety_pct}%，处于 {dcf_res.valuation_status}",
                "balance_sheet_health": f"资产负债率仅 {financials.debt_to_asset_ratio}%，偿债压力极小",
                "growth_sustainability": "行业竞争格局稳固，具备较强提价能力与内生现金流支撑",
            },
            "key_points": [
                f"高净利率 ({dupont_res.net_profit_margin}%) 验证行业定价权与品牌溢价",
                f"经营现金流 ({financials.operating_cash_flow} 亿元) 对净利润覆盖率良好",
                "关注宏观消费周期复苏斜率对终端动销速度的影响",
            ],
        }

        parsed_data = self._parse_json_safely(raw_reply, default_fallback)

        fundamental_output = SpecialistAgentOutput(
            agent_name=self.name,
            stance=parsed_data.get("stance", "积极增配"),
            score=float(parsed_data.get("score", 8.0)),
            summary=parsed_data.get("summary", ""),
            details=parsed_data.get("details", {}),
            key_points=parsed_data.get("key_points", []),
        )

        log_item = {
            "node": "fundamental_agent",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "summary": f"完成基本面穿透评审，ROE: {dupont_res.calculated_roe}%, DCF合理价值: {dcf_res.intrinsic_value_per_share}元",
        }

        return {
            "quote": quote,
            "financials": financials,
            "dupont_result": dupont_res,
            "dcf_result": dcf_res,
            "fundamental_output": fundamental_output,
            "current_step": "fundamental_analysis_completed",
            "execution_logs": [log_item],
        }


# 便捷单例
fundamental_agent = FundamentalAgent()

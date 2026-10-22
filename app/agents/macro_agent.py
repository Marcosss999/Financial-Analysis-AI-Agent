"""
宏观经济与产业政策分析师 Agent (Macro Analyst Specialist Agent)
职责：
1. 评估货币政策、利率周期、通胀水平及宏观流动性对目标标的估值分母端的影响。
2. 研判国家产业战略、设备更新、财政贴息及外贸进出口对目标行业基本面的推动或压制。
3. 输出结构化宏观评分 (0~10 分)、定性评级立场与关键宏观顺逆风因素。
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List
from app.agents.base_agent import BaseSpecialistAgent
from app.tools.market_data import market_data_provider
from app.tools.quant_indicators import quant_indicators_engine
from app.tools.scoring import score_market_environment
from app.models.agent_state import FinancialAgentState
from app.models.schemas import SpecialistAgentOutput


class MacroAgent(BaseSpecialistAgent):
    """宏观策略与产业政策研究员 Agent"""

    def __init__(self):
        super().__init__(
            name="Macro_Analyst",
            role_description="券商宏观首席分析师，擅长货币/财政政策、经济周期与大类资产配置研判",
            temperature=0.2,
        )

    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        """执行宏观环境穿透分析并返回状态增量"""
        ticker = state.get("ticker", "UNKNOWN")
        target_name = state.get("target_name", f"标的-{ticker}")
        custom_query = state.get("custom_query") or "全面研判宏观流动性与产业政策环境"

        # 可量化的“市场环境”：沪深300 的趋势与波动状态（离线时为标注过的模拟数据）
        idx_bars, idx_src = await asyncio.to_thread(market_data_provider.get_index_bars, 60)
        idx_ind = quant_indicators_engine.calculate_indicators(idx_bars) if idx_bars else None
        if idx_ind is not None:
            quant_score, comps = score_market_environment(idx_ind)
        else:
            quant_score, comps = None, {}

        system_prompt = (
            "你是一名顶级证券研究所的宏观经济与产业策略首席分析师。\n"
            "请针对输入的标的资产与宏观背景，从货币政策导向、长端国债利率、经济周期阶段及产业政策扶持力度进行全面推演。\n"
            "要求：输出必须为严格合法的 JSON 对象，不包含任何 Markdown 额外闲聊。\n"
            "JSON 数据结构必须包含以下字段：\n"
            "{\n"
            '  "stance": "定性立场 (如：中性偏多 / 积极乐观 / 审慎观望 / 显著偏空)",\n'
            '  "score": 0.0到10.0之间的综合宏观评分浮点数,\n'
            '  "summary": "核心宏观定性逻辑综述 (100字左右)",\n'
            '  "details": {\n'
            '    "economic_cycle": "所处经济周期判定 (如复苏中早期)",\n'
            '    "policy_impact": "货币与财政产业政策影响机制",\n'
            '    "interest_rate_outlook": "基准利率与长端国债走势研判",\n'
            '    "industry_tailwinds": ["产业红利1", "政策倾斜2"]\n'
            "  },\n"
            '  "key_points": ["核心宏观支撑论据1", "主要潜在宏观阻力2"]\n'
            "}"
        )

        user_content = (
            f"分析标的代码: {ticker}\n"
            f"标的名称: {target_name}\n"
            f"投研指令: {custom_query}\n"
            + (f"【可量化的市场环境(沪深300，来源:{idx_src})】20日涨跌 {idx_ind.ret_20d}%，MA20/MA60={idx_ind.ma20}/{idx_ind.ma60}，"
               f"20日年化波动率 {idx_ind.realized_vol_20}%。\n" if idx_ind is not None else "")
            + "请基于你所掌握的宏观与产业信息客观研判，正反两方面因素都要写；若信息不足请明确说明，不要预设结论。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_reply, live = await self._call_llm_ex(messages, json_mode=True)

        if quant_score is None:
            # 市场环境数据不可用：宏观维度不参与加权(权重由其余维度按比例承接)，不编造分数
            parsed = self._parse_json_safely(raw_reply, {}) if live else {}
            macro_output = SpecialistAgentOutput(
                agent_name=self.name, stance="数据不足", score=5.0,
                summary=str(parsed.get("summary") or f"{target_name} 宏观/市场环境数据不可用，该维度不参与本次加权。"),
                details=dict(parsed.get("details") or {}), key_points=list(parsed.get("key_points") or []),
                quant_score=None, llm_score=None, score_components={}, score_source="unavailable",
            )
        else:
            fallback = {
                "summary": (
                    f"{target_name} 市场环境量化评分 {quant_score}/10（沪深300，数据来源:{idx_src}）：近20日 {idx_ind.ret_20d}%，"
                    f"MA20 {idx_ind.ma20} vs MA60 {idx_ind.ma60}，20日实现波动率 {idx_ind.realized_vol_20}%。"
                    "政策与周期等定性因素需要大模型在线才能解读。（本段为数据模板生成）"
                ),
                "details": {"market_environment": f"分项：{comps}"},
                "key_points": [],
            }
            macro_output = self.build_output(
                quant_score=quant_score, components=comps, raw_reply=raw_reply, live=live, fallback=fallback,
                extra_details={"index_data_source": idx_src},
            )

        log_item = {
            "node": "macro_agent",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "summary": f"完成宏观与政策面分析，得分: {macro_output.score} ({macro_output.stance})",
        }

        return {
            "macro_output": macro_output,
            "current_step": "macro_analysis_completed",
            "execution_logs": [log_item],
        }


# 便捷单例
macro_agent = MacroAgent()

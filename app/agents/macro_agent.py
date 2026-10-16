"""
宏观经济与产业政策分析师 Agent (Macro Analyst Specialist Agent)
职责：
1. 评估货币政策、利率周期、通胀水平及宏观流动性对目标标的估值分母端的影响。
2. 研判国家产业战略、设备更新、财政贴息及外贸进出口对目标行业基本面的推动或压制。
3. 输出结构化宏观评分 (0~10 分)、定性评级立场与关键宏观顺逆风因素。
"""

from datetime import datetime
from typing import Dict, Any, List
from app.agents.base_agent import BaseSpecialistAgent
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
            "请结合当前宏观流动性宽松周期与中长期新质生产力扶持政策，输出该标的宏观研判结论。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_reply = await self._call_llm(messages, json_mode=True)

        default_fallback = {
            "stance": "中性偏多 (Mildly Bullish)",
            "score": 7.6,
            "summary": f"{target_name} 所处行业受国家产业支持引导明确，基准利率保持流动性合理充裕，宏观分母端估值环境良好。",
            "details": {
                "economic_cycle": "温和复苏期",
                "policy_impact": "产业定向支持与税费减免政策落地预期明确",
                "interest_rate_outlook": "十年期国债低位震荡，贴现率维持低位",
                "industry_tailwinds": ["国家重点支柱产业扶持", "设备更新与高质量转型"],
            },
            "key_points": [
                "国内货币政策维持流动性充裕，无系统性紧缩风险",
                "海外地缘与汇率波动对出口链存在阶段性偶发扰动",
            ],
        }

        parsed_data = self._parse_json_safely(raw_reply, default_fallback)

        # 规范化封装为 SpecialistAgentOutput
        macro_output = SpecialistAgentOutput(
            agent_name=self.name,
            stance=parsed_data.get("stance", "中性"),
            score=float(parsed_data.get("score", 7.0)),
            summary=parsed_data.get("summary", ""),
            details=parsed_data.get("details", {}),
            key_points=parsed_data.get("key_points", []),
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

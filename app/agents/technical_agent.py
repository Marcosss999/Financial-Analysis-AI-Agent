"""
量化技术面与走势形态分析师 Agent (Technical Analyst Specialist Agent)
职责：
1. 联动确定性量化技术指标计算器执行 MACD、Wilder RSI 与 Bollinger Bands 纯数学时序分析。
2. 研判均线多空排列、动量背离、超买超卖状态、箱体震荡与关键支撑/阻力位。
3. 结合量化指标计算事实向大模型发起结构化定性推理，产出技术面评分 (0~10 分)、择时操作建议与止损保护位。
"""

from datetime import datetime
from typing import Dict, Any, List
from app.agents.base_agent import BaseSpecialistAgent
from app.tools.scoring import score_technical
from app.models.agent_state import FinancialAgentState
from app.models.schemas import SpecialistAgentOutput
from app.tools.market_data import market_data_provider
from app.tools.quant_indicators import quant_indicators_engine


class TechnicalAgent(BaseSpecialistAgent):
    """量化技术形态与动量择时专家 Agent"""

    def __init__(self):
        super().__init__(
            name="Technical_Analyst",
            role_description="量化对冲基金首席技术分析师，精通量价形态、均线趋势动量与支撑阻力博弈",
            temperature=0.2,
        )

    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        """执行量化技术形态穿透分析并返回状态增量"""
        ticker = state.get("ticker", "600519")
        target_name = state.get("target_name") or f"标的-{ticker}"

        # 1. 确保获取最新行情与近期日 K 线数据
        quote = state.get("quote") or market_data_provider.get_market_quote(ticker, days=60)

        # 2. 调度确定性技术指标纯数学引擎运算
        indicators_res = quant_indicators_engine.calculate_indicators(quote.kline_history)

        # 3. 构造注入量化指标事实的 LLM 结构化推理 Prompt
        system_prompt = (
            "你是一名顶级量化对冲基金的技术分析与量价形态首席分析师。\n"
            "系统已通过确定性算法计算出了近 60 个交易日的 MACD、RSI 及布林带指标。\n"
            "请基于给出的量化指标事实，研判当前走势所处动量阶段、主力博弈结构与右侧交易信号。\n"
            "要求：输出必须为严格合法的 JSON 对象，不包含任何 Markdown 额外闲聊。\n"
            "JSON 数据结构必须包含以下字段：\n"
            "{\n"
            '  "stance": "定性立场 (如：多头突破 / 多头蓄势 / 震荡观望 / 空头破位)",\n'
            '  "score": 0.0到10.0之间的综合技术面评分浮点数,\n'
            '  "summary": "核心技术面与量价动量综述 (120字左右)",\n'
            '  "details": {\n'
            '    "trend_evaluation": "对均线排列与多空趋势方向的研判",\n'
            '    "momentum_analysis": "MACD 柱状图动能与金叉/死叉形态解读",\n'
            '    "volatility_channel": "布林带带宽扩张/收窄及通道位置判断",\n'
            '    "support_resistance_tactic": "支撑位与阻力位附近的操盘博弈建议"\n'
            "  },\n"
            '  "key_points": ["技术面关键信号1", "技术面关键信号2", "战术止损/止盈参考线"]\n'
            "}"
        )

        user_content = (
            f"标的代码: {ticker} ({target_name})\n"
            f"现价: {quote.current_price} 元 | 当日涨跌幅: {quote.change_pct}% | 换手率: {quote.turnover_rate}%\n"
            f"【确定性量化技术指标计算事实】\n"
            f"- MACD(12,26,9): DIF={indicators_res.macd_dif}, DEA={indicators_res.macd_dea}, 柱状图={indicators_res.macd_bar}\n"
            f"- RSI(14): {indicators_res.rsi_14}\n"
            f"- 布林带(20, 2σ): 上轨={indicators_res.boll_upper}, 中轨={indicators_res.boll_middle}, 下轨={indicators_res.boll_lower}\n"
            f"- 趋势/动量: MA20={indicators_res.ma20}, MA60={indicators_res.ma60}, 近5日涨跌={indicators_res.ret_5d}%, 近20日涨跌={indicators_res.ret_20d}%\n"
            f"- 波动率: ATR(14)={indicators_res.atr_14} ({indicators_res.atr_pct}% 现价), 20日年化实现波动率={indicators_res.realized_vol_20}%, 5日/20日波动比={indicators_res.vol_regime}\n"
            f"- 量能: 量比(当日/前5日均量)={indicators_res.volume_ratio}, OBV 斜率={indicators_res.obv_slope}, MFI(14)={indicators_res.mfi_14}\n"
            f"- 布林 %B={indicators_res.boll_pctb}, 带宽={indicators_res.boll_bandwidth}\n"
            f"- 核心支撑位: {indicators_res.support_level} 元 | 核心阻力位: {indicators_res.resistance_level} 元\n"
            f"- 算法综合信号: {indicators_res.trend_signal}\n"
            "请出具专业的技术形态研判与入场择时策略。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_reply, live = await self._call_llm_ex(messages, json_mode=True)

        quant_score, comps = score_technical(indicators_res)
        fallback = {
            "summary": (
                f"{target_name} 技术面量化评分 {quant_score}/10：算法信号[{indicators_res.trend_signal}]，"
                f"RSI {indicators_res.rsi_14}，MACD 柱 {indicators_res.macd_bar}，近5日 {indicators_res.ret_5d}%，"
                f"20日实现波动率 {indicators_res.realized_vol_20}%（5/20日比 {indicators_res.vol_regime}），"
                f"量比 {indicators_res.volume_ratio}，MFI {indicators_res.mfi_14}。（本段为数据模板生成，非大模型推理）"
            ),
            "details": {
                "trend_evaluation": f"现价 {indicators_res.last_close} vs MA20 {indicators_res.ma20} / MA60 {indicators_res.ma60}",
                "momentum_analysis": f"DIF {indicators_res.macd_dif} / DEA {indicators_res.macd_dea}，RSI {indicators_res.rsi_14}",
                "volatility_channel": f"布林区间 [{indicators_res.boll_lower}, {indicators_res.boll_upper}]，%B {indicators_res.boll_pctb}；ATR {indicators_res.atr_14}",
                "support_resistance_tactic": f"支撑 {indicators_res.support_level} 元，阻力 {indicators_res.resistance_level} 元",
            },
            "key_points": [f"分项得分(相对中性5分的加减)：{comps}"],
        }
        technical_output = self.build_output(
            quant_score=quant_score, components=comps, raw_reply=raw_reply, live=live, fallback=fallback,
        )

        log_item = {
            "node": "technical_agent",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "summary": f"完成量化技术面分析，得分: {technical_output.score} ({technical_output.stance}), 信号: {indicators_res.trend_signal}",
        }

        return {
            "quote": quote,
            "indicators_result": indicators_res,
            "technical_output": technical_output,
            "current_step": "technical_analysis_completed",
            "execution_logs": [log_item],
        }


# 便捷单例
technical_agent = TechnicalAgent()

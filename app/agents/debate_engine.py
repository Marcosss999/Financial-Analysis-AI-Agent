"""
投委会多空辩论与冲突仲裁引擎 (Multi-Agent Debate & Consensus Engine)
职责：
1. 多头辩手 (Bull Advocate Agent)：
   - 首轮立论：结合基本面、技术面、东财分时主力净流入及官方公告，提炼 3 大【进攻阿尔法催化剂 (Catalysts)】。
   - 次轮驳论：对空头抛出的下行风险逐条进行【针锋相对的反驳与防御证明】(Point-by-Point Rebuttal)。
2. 空头辩手 (Bear Critic Agent)：
   - 首轮批判：审视财务杠杆、估值溢价、历史在险损失，指出 3 大【致命阿喀琉斯之踵 (Vulnerabilities)】。
   - 次轮质询：对多头的乐观假设与催化剂进行【精准挑刺与破绽质疑】(Counter-Critique)。
3. 投委会仲裁官 (Investment Committee Arbiter Node)：
   - 提取双方【核心争议焦点 (Battleground)】与【盈亏不对称比 (Risk/Reward)】。
   - 维护辩论轮次计数器与收敛门控，彻底规避 LangGraph 循环状态图死循环。
"""

from datetime import datetime
from typing import Dict, Any, List
from app.core.config import settings
from app.agents.base_agent import BaseSpecialistAgent, to_score
from app.models.agent_state import FinancialAgentState
from app.models.schemas import DebateTurn, ArbiterVerdict
from app.tools.scoring import (
    DIM_LABELS, aggregate, atr_risk_sizing, debate_adjustment, verdict_from_score,
)


def _dim_extremes(state: FinancialAgentState):
    """取已有数据的维度中得分最高/最低者，用于离线模板里的“多头论据/空头论据”，保证文字来自数据而非预设立场"""
    ds = {k: v for k, v in (state.get("dimension_scores") or {}).items() if v is not None}
    if not ds:
        return None, None
    hi = max(ds, key=ds.get)
    lo = min(ds, key=ds.get)
    return (hi, ds[hi]), (lo, ds[lo])


def _breakdown_line(state: FinancialAgentState) -> str:
    bd = state.get("score_breakdown") or []
    parts = []
    for b in bd:
        if b.get("score") is None:
            parts.append(f"{b['label']}:无数据")
        else:
            parts.append(f"{b['label']} {b['score']}分×权重{b['effective_weight']:.0%}")
    return "；".join(parts) if parts else "暂无"


class BullAgent(BaseSpecialistAgent):
    """多头首席进攻辩手 Agent"""

    def __init__(self):
        super().__init__(
            name="Bull_Advocate",
            role_description="投委会资深多头研究员，专注挖掘预期差、成长阿尔法催化剂与估值上行空间",
            temperature=0.3,
        )

    async def argue(self, state: FinancialAgentState) -> Dict[str, Any]:
        """多头提出进攻论述"""
        ticker = state.get("ticker", "UNKNOWN")
        target_name = state.get("target_name") or f"标的-{ticker}"
        macro = state.get("macro_output")
        fund = state.get("fundamental_output")
        tech = state.get("technical_output")
        intel = state.get("market_intel")
        last_bear = state.get("bear_argument") or "无 (首轮立论阶段)"
        current_round = state.get("current_debate_round", 1)

        style = state.get("strategy_style") or "balanced"
        w_fund = state.get("fundamental_weight", 0.40)
        w_tech = state.get("technical_weight", 0.35)
        w_macro = state.get("macro_weight", 0.15)
        w_flow = state.get("capital_flow_weight", 0.10)

        # 测算近 60 日真实股价涨跌幅与动量
        quote = state.get("quote")
        price_gain_60d = 0.0
        if quote and quote.kline_history and len(quote.kline_history) >= 10:
            p_start = quote.kline_history[0].close
            p_end = quote.kline_history[-1].close
            if p_start > 0:
                price_gain_60d = round(((p_end - p_start) / p_start) * 100.0, 1)

        # 多头基础信念 = 用户权重加权后的基础分(确定性)；缺失时才回退到专家分按权重现算
        base_bull = state.get("base_score")
        if base_bull is None:
            parts = [(o.score, w) for o, w in ((fund, w_fund), (tech, w_tech), (macro, w_macro)) if o is not None]
            tw = sum(w for _, w in parts) or 1.0
            base_bull = sum(sc * w for sc, w in parts) / tw if parts else 5.0

        if current_round > 1:
            # 第二轮：吸收空头上一轮的风险判断后做有限修正
            last_bear_s = state.get("bear_score", 5.5)
            dynamic_bull_score = round(max(0.0, min(10.0, base_bull * 0.85 + (10.0 - last_bear_s) * 0.15)), 1)
        else:
            dynamic_bull_score = round(max(0.0, min(10.0, base_bull)), 1)

        intel_summary = (
            f"主力资金流向: {intel.main_capital_flow if intel else '暂无'}; "
            f"官方重大公告: {intel.announcements[0] if (intel and intel.announcements) else '暂无数据'}; "
            f"市场舆情: {intel.news_headlines[0] if (intel and intel.news_headlines) else '暂无数据'}"
        )

        if current_round == 1:
            # 首轮：攻势立论
            system_prompt = (
                "你是投委会的【多头首席辩手 (Bull Advocate)】，当前处于【第1轮：攻势立论阶段】。\n"
                "你的职责是综合宏观、基本面、量化技术指标及东财主力资金抢筹信号，确立鲜明看涨逻辑。\n"
                "请输出严格合法的 JSON 对象：\n"
                "{\n"
                '  "conviction_score": 0.0到10.0之间的看多置信分,\n'
                '  "core_thesis": "多头核心逻辑总纲 (80~100字)",\n'
                '  "catalysts": ["阿尔法上行催化剂1", "估值重估驱动2", "主力资金与筹码优势3"]\n'
                "}"
            )
            user_content = (
                f"标的: {ticker} ({target_name})\n"
                f"策略模式: {style} (基本面 {int(w_fund*100)}%, 技术 {int(w_tech*100)}%, 宏观 {int(w_macro*100)}%, 资金 {int(w_flow*100)}%)\n"
                f"近60日价格动量涨幅: {price_gain_60d}%\n"
                f"全网多源情报事实: {intel_summary}\n"
                f"【量化评分与权重贡献】{_breakdown_line(state)}\n"
                f"基本面与估值: {fund.summary if fund else '(基本面专家无产出)'}\n"
                f"量化技术形态: {tech.summary if tech else '(技术面专家无产出)'}\n"
                f"宏观/市场环境: {macro.summary if macro else '(宏观专家无产出)'}\n"
                "请发表有力的第一轮进攻立论。"
            )
        else:
            # 次轮：交叉驳论与证据防御
            system_prompt = (
                "你是投委会的【多头首席辩手 (Bull Advocate)】，当前处于【第2轮：交叉质询与针锋相对驳论阶段】。\n"
                "空头辩手在上一轮提出了尖锐质疑。你的任务是【针对性逐条反驳空头漏洞】，用扎实论据说明风险已被市场充分计提 (Priced-in)，捍卫多头胜率！\n"
                "请输出严格合法的 JSON 对象：\n"
                "{\n"
                '  "conviction_score": 0.0到10.0之间的看多置信分 (理性微调),\n'
                '  "core_thesis": "次轮防守反击核心陈词 (80~100字)",\n'
                '  "rebuttals": ["针对空头质疑点1的针锋相对反驳", "针对空头质疑点2的反驳解释"],\n'
                '  "catalysts": ["更新后最坚固的2大核心上行催化剂"]\n'
                "}"
            )
            user_content = (
                f"标的: {ticker} ({target_name})\n"
                f"上一轮空头的尖锐质疑如下：\n{last_bear}\n"
                f"全网情报支撑: {intel_summary}\n"
                "请对空头的观点展开点对点驳论，阐明为何其过分悲观。"
            )

        raw, live = await self._call_llm_ex([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])

        hi, lo = _dim_extremes(state)
        default_catalysts = []
        if hi:
            default_catalysts.append(f"最强维度：{DIM_LABELS[hi[0]]}量化评分 {hi[1]}/10")
        default_catalysts.append(f"资金面：{intel.main_capital_flow if intel else '暂无数据'}")
        if fund and fund.score_components:
            default_catalysts.append(f"估值/盈利分项(0~10)：{fund.score_components}")
        if price_gain_60d > 15.0 and style in ["momentum", "short_term"]:
            default_catalysts.insert(0, f"近 60 日涨幅 +{price_gain_60d}%，趋势延续是动量风格下的多头论据")
        elif price_gain_60d < -15.0 and style == "deep_value":
            default_catalysts.insert(0, f"近 60 日回调 {price_gain_60d}%，若估值折价成立则提供逆向安全边际")

        fallback = {
            "conviction_score": dynamic_bull_score,
            "core_thesis": f"{target_name} 多头视角（模板生成，非大模型推理）：权重加权基础信念 {dynamic_bull_score}/10。{_breakdown_line(state)}。",
            "catalysts": default_catalysts[:3],
            "rebuttals": [
                f"对空头风险的回应：基础分由用户设定权重加权得出（{_breakdown_line(state)}）"
            ] if current_round > 1 else [],
        }

        # 只有真实 LLM 在线才采用其信念分与文字；离线用确定性基础信念 + 数据模板
        res = self._parse_json_safely(raw, fallback) if live else fallback
        score = to_score(res.get("conviction_score"), dynamic_bull_score)
        core_thesis = res.get("core_thesis") or fallback["core_thesis"]
        catalysts = res.get("catalysts") or fallback["catalysts"]
        rebuttals = res.get("rebuttals", [])

        # 格式化呈现
        if current_round == 1 or not rebuttals:
            thesis = f"{core_thesis}\n\n**【多头核心进攻催化剂 (Catalysts)】**：\n" + "\n".join([f"- {c}" for c in catalysts])
        else:
            thesis = f"{core_thesis}\n\n**【针对空头质疑的针锋相对反驳 (Rebuttals)】**：\n" + "\n".join([f"- 🛡️ {r}" for r in rebuttals]) + "\n\n**【多头坚守的核心上行驱动】**：\n" + "\n".join([f"- 🚀 {c}" for c in catalysts])

        return {
            "bull_argument": thesis,
            "bull_score": round(score, 1),
            "bull_catalysts": catalysts,
        }

    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        return await self.argue(state)


class BearAgent(BaseSpecialistAgent):
    """空头首席防守批判辩手 Agent"""

    def __init__(self):
        super().__init__(
            name="Bear_Critic",
            role_description="投委会严苛空头风控专家，专注寻找财务水分、周期见顶、黑天鹅与估值泡沫",
            temperature=0.3,
        )

    async def argue(self, state: FinancialAgentState) -> Dict[str, Any]:
        """空头提出防守质疑"""
        ticker = state.get("ticker", "UNKNOWN")
        target_name = state.get("target_name") or f"标的-{ticker}"
        bull_arg = state.get("bull_argument") or "多头暂无发言"
        fund = state.get("fundamental_output")
        quote = state.get("quote")
        financials = state.get("financials")
        var_res = state.get("var_result")
        intel = state.get("market_intel")
        current_round = state.get("current_debate_round", 1)

        debt_ratio = financials.debt_to_asset_ratio if financials else 35.0
        pe = quote.pe_ttm if quote else 22.0
        var_pct = var_res.var_percentage if var_res else 5.0

        # 测算近 60 日真实股价涨跌幅与预期透支
        price_gain_60d = 0.0
        if quote and quote.kline_history and len(quote.kline_history) >= 10:
            p_start = quote.kline_history[0].close
            p_end = quote.kline_history[-1].close
            if p_start > 0:
                price_gain_60d = round(((p_end - p_start) / p_start) * 100.0, 1)

        base_risk = 4.2
        t_out = state.get("technical_output")
        ind = state.get("indicators_result")
        if debt_ratio > 65.0:
            base_risk += 1.8
        elif debt_ratio > 50.0:
            base_risk += 0.8
        if pe > 40.0:
            base_risk += 1.5
        elif pe > 25.0:
            base_risk += 0.6
        if var_pct > 10.0:
            base_risk += 1.2

        if price_gain_60d > 35.0:
            base_risk += 1.6
        elif price_gain_60d > 20.0:
            base_risk += 0.8
        elif price_gain_60d < -25.0:
            base_risk = max(3.6, base_risk - 0.8)

        # 波动率体制与短线风险：高波动、波动放大都增加空头论据
        if ind is not None and ind.realized_vol_20 > 50.0:
            base_risk += 0.8
        if ind is not None and ind.vol_regime > 1.8:
            base_risk += 0.4

        if current_round > 1:
            last_bull_s = state.get("bull_score", 8.0)
            dynamic_bear_score = round(max(0.0, min(10.0, base_risk * 0.75 + (10.0 - last_bull_s) * 0.25)), 1)
        else:
            dynamic_bear_score = round(max(0.0, min(10.0, base_risk)), 1)

        intel_summary = (
            f"主力资金流向: {intel.main_capital_flow if intel else '暂无'}; "
            f"官方重大公告: {intel.announcements[0] if (intel and intel.announcements) else '暂无数据'}; "
            f"市场舆情: {intel.news_headlines[0] if (intel and intel.news_headlines) else '暂无数据'}"
        )

        if current_round == 1:
            system_prompt = (
                "你是投委会的【空头首席辩手 (Bear Critic)】，当前处于【第1轮：防守批判阶段】。\n"
                "你的任务是带着审慎怀疑的眼光，针对多头的乐观预期进行精准穿透批判，揭示估值陷阱、隐形成本与下行风险。\n"
                "请输出严格合法的 JSON 对象：\n"
                "{\n"
                '  "conviction_score": 0.0到10.0之间的看空防守得分 (分值越高代表风险越大),\n'
                '  "core_thesis": "空头防守审慎陈词 (80~100字)",\n'
                '  "vulnerabilities": ["潜在暴雷风险与阿喀琉斯之踵1", "行业周期逆风与估值泡沫2", "盘口博弈风险3"]\n'
                "}"
            )
            user_content = (
                f"标的: {ticker} ({target_name})\n"
                f"近60日价格累积涨幅: {price_gain_60d}%\n"
                f"财务与估值: PE(TTM): {pe}x, 资产负债率: {debt_ratio}%, 波动率VaR: {var_pct}%\n"
                f"【量化评分与权重贡献】{_breakdown_line(state)}\n"
                + (f"短线风险: ATR={ind.atr_14}({ind.atr_pct}%), 20日实现波动率={ind.realized_vol_20}%, 波动比={ind.vol_regime}\n" if ind is not None else "")
                +
                f"情报事实: {intel_summary}\n"
                f"多头首轮陈述: {bull_arg}\n"
                "请发起尖锐的空头第一轮防守批判。"
            )
        else:
            system_prompt = (
                "你是投委会的【空头首席辩手 (Bear Critic)】，当前处于【第2轮：精准穿透质询阶段】。\n"
                "针对多头辩手刚刚发表的立论与辩解，【精准挑刺其乐观假设与逻辑漏洞】，指出主力洗盘背后的潜在出逃风险与上方强压力位！\n"
                "请输出严格合法的 JSON 对象：\n"
                "{\n"
                '  "conviction_score": 0.0到10.0之间的看空防守得分 (分值越高代表风险越大),\n'
                '  "core_thesis": "次轮穿透质询核心陈词 (80~100字)",\n'
                '  "counter_critiques": ["直击多头催化剂的漏洞与过度乐观假设1", "指出技术阻力或基本面隐患2"],\n'
                '  "vulnerabilities": ["坚持的核心下行防守警示"]\n'
                "}"
            )
            user_content = (
                f"标的: {ticker} ({target_name})\n"
                f"多头辩手最新辩解陈词如下：\n{bull_arg}\n"
                f"全网情报事实: {intel_summary}\n"
                "请向多头展开深入质询，指明为何不可掉以轻心。"
            )

        raw, live = await self._call_llm_ex([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])

        hi, lo = _dim_extremes(state)
        default_vulns = []
        if lo:
            default_vulns.append(f"最弱维度：{DIM_LABELS[lo[0]]}量化评分 {lo[1]}/10")
        default_vulns.append(f"估值与杠杆：PE(TTM) {pe}x，资产负债率 {debt_ratio}%，VaR {var_pct}%")
        if ind is not None:
            default_vulns.append(f"短线波动：20日实现波动率 {ind.realized_vol_20}%，5/20日波动比 {ind.vol_regime}，ATR {ind.atr_pct}% 现价")
        if price_gain_60d > 20.0:
            default_vulns.insert(0, f"近 60 日已上涨 +{price_gain_60d}%，需警惕利好兑现后的回吐")

        fallback = {
            "conviction_score": dynamic_bear_score,
            "core_thesis": f"{target_name} 空头视角（模板生成，非大模型推理）：风险分 {dynamic_bear_score}/10，由负债率、估值、VaR、近期涨幅与波动率体制规则化累加。",
            "vulnerabilities": default_vulns[:3],
            "counter_critiques": [
                f"多头论据依赖的基础分来自权重设定：{_breakdown_line(state)}；权重偏向的维度若失真，结论同步失真"
            ] if current_round > 1 else [],
        }

        res = self._parse_json_safely(raw, fallback) if live else fallback
        score = to_score(res.get("conviction_score"), dynamic_bear_score)
        core_thesis = res.get("core_thesis") or fallback["core_thesis"]
        vulns = res.get("vulnerabilities") or fallback["vulnerabilities"]
        critiques = res.get("counter_critiques", [])

        if current_round == 1 or not critiques:
            thesis = f"{core_thesis}\n\n**【空头重点下行防守警示 (Downside Risks)】**：\n" + "\n".join([f"- ⚠️ {v}" for v in vulns])
        else:
            thesis = f"{core_thesis}\n\n**【对多头催化剂的穿透式质询 (Counter-Critiques)】**：\n" + "\n".join([f"- 🎯 {c}" for c in critiques]) + "\n\n**【空头坚持把守的风险红线】**：\n" + "\n".join([f"- ⚠️ {v}" for v in vulns])

        return {
            "bear_argument": thesis,
            "bear_score": round(score, 1),
            "bear_vulnerabilities": vulns,
        }

    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        return await self.argue(state)


class ArbiterAgent(BaseSpecialistAgent):
    """投委会主席与仲裁裁决 Agent"""

    def __init__(self):
        super().__init__(
            name="Investment_Committee_Arbiter",
            role_description="投委会主席兼终审投资决策官，负责调和多空分歧，平衡风险与收益，出具组合配置决策",
            temperature=0.1,
        )

    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        """执行多空辩论裁决并更新共识结论"""
        ticker = state.get("ticker", "UNKNOWN")
        target_name = state.get("target_name") or f"标的-{ticker}"
        bull_arg = state.get("bull_argument", "")
        bull_score = state.get("bull_score", 8.0)
        bear_arg = state.get("bear_argument", "")
        bear_score = state.get("bear_score", 5.0)
        current_round = state.get("current_debate_round", 1)

        style = state.get("strategy_style") or "balanced"
        w_fund = state.get("fundamental_weight", 0.40)
        w_tech = state.get("technical_weight", 0.35)
        w_macro = state.get("macro_weight", 0.15)
        w_flow = state.get("capital_flow_weight", 0.10)

        debate_enabled = state.get("enable_debate", True)
        ind = state.get("indicators_result")

        # ---- 1. 基础分：用户权重 × 各维度量化评分(由 experts_join 计算；缺失时现算) ----
        base_score = state.get("base_score")
        breakdown = state.get("score_breakdown") or []
        if base_score is None:
            scores = {"fundamental": getattr(state.get("fundamental_output"), "quant_score", None),
                      "technical": getattr(state.get("technical_output"), "quant_score", None),
                      "macro": getattr(state.get("macro_output"), "quant_score", None),
                      "capital_flow": None}
            agg = aggregate(scores, {"fundamental": w_fund, "technical": w_tech, "macro": w_macro,
                                     "capital_flow": w_flow})
            base_score, breakdown = agg["base_score"], agg["breakdown"]

        # ---- 2. 辩论只做有界调整；关闭辩论则不调整 ----
        if not debate_enabled:
            adj = 0.0
            bull_score, bear_score = base_score, round(10.0 - base_score, 1)
            experts = [state.get("fundamental_output"), state.get("technical_output"), state.get("macro_output")]
            bull_arg = "(已关闭多空辩论，以下为三位专家的结论摘要)\n" + "\n".join(
                f"- {o.agent_name}: {o.stance} ({o.score}) {o.summary}" for o in experts if o is not None
            )
            bear_arg = ""
        else:
            adj = debate_adjustment(bull_score, bear_score)

        divergence = abs(bull_score - bear_score)
        consensus_score = round(max(0.0, min(10.0, base_score + adj)), 2)
        verdict_choice, position_choice = verdict_from_score(consensus_score)

        # ---- 3. ATR 风险预算：止损位、仓位上限、盈亏比(全部确定性) ----
        sizing = atr_risk_sizing(ind)
        if sizing["risk_cap_position_pct"] is not None:
            position_choice = f"{position_choice} | ATR 风险预算仓位上限 ≈ {sizing['risk_cap_position_pct']}%"
        if sizing["atr_stop_price"] is not None and ind is not None and ind.resistance_level > ind.last_close:
            reward = ind.resistance_level - ind.last_close
            risk = ind.last_close - sizing["atr_stop_price"]
            rr_text = f"盈亏比约 {reward / risk:.1f}:1（至阻力位 {ind.resistance_level} 元 vs 至 ATR 止损 {sizing['atr_stop_price']} 元）"
        else:
            rr_text = "盈亏比未估算（缺少 ATR 或阻力位数据）"
        tactics_det = (
            f"以 {settings.decision.ATR_STOP_MULT:g}×ATR 设止损：{sizing['atr_stop_price']} 元（距现价 {sizing['atr_stop_pct']}%）；"
            f"单笔风险预算 {settings.decision.RISK_PER_TRADE_PCT:g}% 对应仓位上限 {sizing['risk_cap_position_pct']}%。"
            if sizing["atr_stop_price"] is not None else "缺少 ATR 数据，未生成止损建议。"
        )

        style_names = {
            "momentum": "动量主升突破模式",
            "balanced": "机构均衡配置模式",
            "deep_value": "深度价值投资模式",
            "short_term": "短线游资博弈模式",
            "custom": "自定义指标权重模式",
        }
        style_name = style_names.get(style, "机构均衡配置模式")

        hi, lo = _dim_extremes(state)
        contradiction = (f"{DIM_LABELS[hi[0]]}({hi[1]}) vs {DIM_LABELS[lo[0]]}({lo[1]})" if hi and lo and hi[0] != lo[0]
                         else "各维度评分一致，无显著分歧")
        battleground = f"【第{current_round}轮争议焦点】：{contradiction}"

        system_prompt = (
            "你是顶级投资机构投委会主席。结论档位、分数、仓位与止损均已由确定性规则算出，你不需要也不得更改它们。\n"
            "你只需用 2~3 句话解释：为什么多空分歧不改变/小幅改变基础分，并点出最需要跟踪的风险。\n"
            "输出必须为严格合法的 JSON：\n"
            "{\n"
            '  "arbitration_summary": "分歧调和与裁决逻辑 (120字左右)",\n'
            '  "core_contradiction": "本次审议最核心的矛盾 (30字以内)"\n'
            "}"
        )
        user_content = (
            f"标的: {ticker} ({target_name})\n"
            f"策略模式: {style_name}\n"
            f"【确定性评分】基础分 {base_score}（{_breakdown_line(state)}），辩论调整 {adj:+.2f}，最终 {consensus_score} → {verdict_choice}\n"
            f"【多头陈述 (置信分 {bull_score})】: {bull_arg}\n"
            f"【空头陈述 (风险分 {bear_score})】: {bear_arg}\n"
        )

        raw, live = await self._call_llm_ex([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])
        parsed = self._parse_json_safely(raw, {}) if live else {}
        fallback_summary = (
            f"按【{style_name}】权重加权基础分 {base_score}；多空辩论调整 {adj:+.2f}（上限 ±{settings.decision.DEBATE_MAX_ADJUST:g}）；"
            f"最终 {consensus_score} → {verdict_choice}。主要分歧：{contradiction}。（模板生成，非大模型推理）"
        )

        # 档位 / 分数 / 仓位 / 止损均为确定性结果，LLM 只提供解释性文字
        arbiter_verdict = ArbiterVerdict(
            final_verdict=verdict_choice,
            consensus_score=consensus_score,
            target_position_pct=position_choice,
            arbitration_summary=str(parsed.get("arbitration_summary") or fallback_summary),
            core_contradiction=str(parsed.get("core_contradiction") or contradiction),
            risk_reward_ratio=rr_text,
            entry_tactics=tactics_det,
            base_score=base_score,
            debate_adjustment=adj,
            score_breakdown=breakdown,
            atr_stop_price=sizing["atr_stop_price"],
            atr_stop_pct=sizing["atr_stop_pct"],
            risk_cap_position_pct=sizing["risk_cap_position_pct"],
        )

        debate_turn = None if not debate_enabled else DebateTurn(
            round_index=current_round,
            stage_name="首轮立论攻防阶段" if current_round == 1 else "次轮针对性交叉质询阶段",
            bull_thesis=bull_arg,
            bull_score=round(bull_score, 2),
            bull_catalysts=state.get("bull_catalysts", []),
            bear_antithesis=bear_arg,
            bear_score=round(bear_score, 2),
            bear_vulnerabilities=state.get("bear_vulnerabilities", []),
            divergence=round(divergence, 2),
            battleground_issue=battleground,
        )

        # 判断是否满足收敛条件
        is_converged = (
            not debate_enabled
            or divergence <= settings.debate.CONVERGENCE_DELTA * 10
            or current_round >= settings.debate.MAX_DEBATE_ROUNDS
        )

        log_item = {
            "node": "arbiter_agent",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "summary": f"第 {current_round} 轮辩论仲裁完毕: 决议[{arbiter_verdict.final_verdict}], 共识分:{arbiter_verdict.consensus_score}, 核心矛盾:[{arbiter_verdict.core_contradiction}], 建议仓位:{arbiter_verdict.target_position_pct}",
        }

        return {
            "current_debate_round": current_round,
            "is_debate_converged": is_converged,
            "debate_turns": [debate_turn] if debate_turn else [],
            "arbiter_verdict": arbiter_verdict,
            "current_step": "debate_arbitration_completed",
            "execution_logs": [log_item],
        }


# 单例导出
bull_agent = BullAgent()
bear_agent = BearAgent()
arbiter_agent = ArbiterAgent()

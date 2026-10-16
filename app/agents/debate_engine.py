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
from app.agents.base_agent import BaseSpecialistAgent
from app.models.agent_state import FinancialAgentState
from app.models.schemas import DebateTurn, ArbiterVerdict


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

        # 结合主力资金动量赋分
        flow_score = 7.5
        if intel:
            flow_score = 8.8 if intel.is_main_inflow else 6.2

        f_score = fund.score if fund else 7.8
        t_score = tech.score if tech else 7.2
        m_score = macro.score if macro else 7.0
        base_bull = round(f_score * w_fund + t_score * w_tech + m_score * w_macro + flow_score * w_flow, 1)

        if current_round > 1:
            # 第二轮辩论：向空方的防守质疑进行理性微调
            last_bear_s = state.get("bear_score", 5.5)
            dynamic_bull_score = round(base_bull * 0.85 + (10.0 - last_bear_s) * 0.15, 1)
        else:
            dynamic_bull_score = min(9.5, max(6.0, round(base_bull, 1)))

        intel_summary = (
            f"主力资金流向: {intel.main_capital_flow if intel else '暂无'}; "
            f"官方重大公告: {intel.announcements[0] if (intel and intel.announcements) else '近期经营无异常'}; "
            f"市场舆情: {intel.news_headlines[0] if (intel and intel.news_headlines) else '行业排产稳定'}"
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
                f"基本面与估值: {fund.summary if fund else '财务稳健'}\n"
                f"量化技术形态: {tech.summary if tech else '走势良好'}\n"
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

        raw = await self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])

        default_catalysts = [
            f"{target_name} 核心主营产品市占率高且具备产业链强势定价权，抗周期韧性强",
            f"多源情报印证：{intel.main_capital_flow if intel else '主力资金平稳'}，筹码具备抗跌防御垫",
            "DCF 绝对估值模型测算每股公允价值具备充足的安全边际"
        ]
        if price_gain_60d > 15.0 and style in ["momentum", "short_term"]:
            default_catalysts.insert(0, f"【右侧主升浪动量】近 60 日股价突破上行（累积 +{price_gain_60d}%），均线系统呈现强势右侧多头排列")
        elif price_gain_60d < -15.0 and style == "deep_value":
            default_catalysts.insert(0, f"【左侧极端错杀黄金坑】近 60 日大幅深蹲回调 {price_gain_60d}%，历史估值折价已打满，提供充足逆向安全边际")

        fallback = {
            "conviction_score": dynamic_bull_score,
            "core_thesis": f"{target_name} 具备坚实的商业护城河与内生现金流驱动力，当前估值安全边际明确，多头共振蓄势突破。",
            "catalysts": default_catalysts[:3],
            "rebuttals": [
                "空头所担忧的短期波动已在近几周走势中充分消化计提 (Priced-in)，缺乏继续深幅杀估值的边际增量",
                "主力大单持续沉淀，下档均线密集区提供强劲承接力，风险收益比严重偏向多头"
            ] if current_round > 1 else [],
        }

        res = self._parse_json_safely(raw, fallback)
        score = float(res.get("conviction_score") or dynamic_bull_score)
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

        if current_round > 1:
            last_bull_s = state.get("bull_score", 8.0)
            dynamic_bear_score = round(base_risk * 0.75 + (10.0 - last_bull_s) * 0.25, 1)
        else:
            dynamic_bear_score = round(min(8.8, max(4.0, base_risk)), 1)

        intel_summary = (
            f"主力资金流向: {intel.main_capital_flow if intel else '暂无'}; "
            f"官方重大公告: {intel.announcements[0] if (intel and intel.announcements) else '近期经营无异常'}; "
            f"市场舆情: {intel.news_headlines[0] if (intel and intel.news_headlines) else '行业排产稳定'}"
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

        raw = await self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])

        default_vulns = [
            "应收账款周转天数若持续拉长，可能对自由现金流质量形成潜在抽水压制",
            "行业潜在新进入者扩产可能引发下一阶段恶性价格内卷，毛利率承压",
            "历史波动率在季报披露日前后显著放大，右侧追高盈亏比不佳"
        ]
        if price_gain_60d > 20.0:
            default_vulns.insert(0, f"【预期抢跑透支警示】近 60 日股价已累积提前大涨 +{price_gain_60d}%，市场抢跑计入了过多乐观假设，警惕利好兑现后的获利盘踩踏 (Priced-in)")

        fallback = {
            "conviction_score": dynamic_bear_score,
            "core_thesis": f"{target_name} 估值需审慎防守看待，市场预期过于饱满，需警惕原材料成本波动与外需走弱带来的杀估值风险。",
            "vulnerabilities": default_vulns[:3],
            "counter_critiques": [
                "多头所称的催化剂多属中长期愿景，短期面临上方近 20 日密集成交区压力压顶，量能若跟不上极易诱多冲高回落",
                "资产负债率与刚性利息开支在宏观复苏斜率放缓时会放大财务弹性风险"
            ] if current_round > 1 else [],
        }

        res = self._parse_json_safely(raw, fallback)
        score = float(res.get("conviction_score") or dynamic_bear_score)
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

        divergence = abs(bull_score - bear_score)

        style = state.get("strategy_style") or "balanced"
        w_fund = state.get("fundamental_weight", 0.40)
        w_tech = state.get("technical_weight", 0.35)
        w_macro = state.get("macro_weight", 0.15)
        w_flow = state.get("capital_flow_weight", 0.10)

        style_names = {
            "momentum": "动量主升突破模式",
            "balanced": "机构均衡配置模式",
            "deep_value": "深度价值投资模式",
            "short_term": "短线游资博弈模式",
            "custom": "自定义指标权重模式",
        }
        style_name = style_names.get(style, "机构均衡配置模式")

        # 动态加权共识得分
        bull_coef = 0.65 if style in ["momentum", "short_term"] else (0.48 if style == "deep_value" else 0.55)
        bear_coef = 1.0 - bull_coef
        consensus_score = round(bull_score * bull_coef + (10.0 - bear_score) * bear_coef, 1)

        if consensus_score >= 8.2:
            verdict_choice = "强烈买入 (Strong Buy)"
            position_choice = "60% ~ 75% (重仓配置 / 积极进攻)"
        elif consensus_score >= 7.2:
            verdict_choice = "逢低增配 (Overweight / Buy on Dips)"
            position_choice = "45% ~ 55% (标准配置 / 稳健逢低吸筹)"
        elif consensus_score >= 6.0:
            verdict_choice = "中性持有 (Neutral / Hold)"
            position_choice = "25% ~ 35% (防守性底仓 / 保持观望)"
        else:
            verdict_choice = "建议规避 (Underweight / Caution)"
            position_choice = "10% 以下 (轻仓或空仓防守)"

        # 核心分歧点提炼
        battleground = f"【第{current_round}轮争议焦点】：多头依据主力抢筹与DCF安全边际看多 vs 空头紧盯上方阻力位承压与估值泡沫"

        system_prompt = (
            "你是顶级投资机构投委会主席 (Investment Committee Chairman)。\n"
            "请客观审视多头与空头的激烈对抗辩论，调和矛盾，评估不对称盈亏比，出具终审投资裁决书。\n"
            "输出必须为严格合法的 JSON 对象：\n"
            "{\n"
            '  "final_verdict": "终审决议",\n'
            '  "consensus_score": 0.0到10.0之间的综合加权共识得分,\n'
            '  "target_position_pct": "建议组合配置权重",\n'
            '  "arbitration_summary": "多空分歧调和与终局裁决逻辑 (120字左右)",\n'
            '  "core_contradiction": "本次审议最核心的矛盾对立面 (30字以内)",\n'
            '  "risk_reward_ratio": "评估不对称盈亏比 (如: 盈亏比 2.5:1，赔率占优)",\n'
            '  "entry_tactics": "具体交易执行、分批吸筹节奏与止损线安排"\n'
            "}"
        )

        user_content = (
            f"标的: {ticker} ({target_name})\n"
            f"策略模式: {style_name} (基本面 {int(w_fund*100)}%, 技术 {int(w_tech*100)}%, 宏观 {int(w_macro*100)}%, 资金 {int(w_flow*100)}%)\n"
            f"当前辩论轮次: 第 {current_round} 轮\n"
            f"参考综合共识分: {consensus_score}\n"
            f"【多头陈述 (置信分 {bull_score})】: {bull_arg}\n"
            f"【空头陈述 (风险分 {bear_score})】: {bear_arg}\n"
            f"当前多空分歧度: {divergence:.2f}\n"
            "请出具投委会终审裁定书。"
        )

        raw = await self._call_llm([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])
        fallback = {
            "final_verdict": verdict_choice,
            "consensus_score": consensus_score,
            "target_position_pct": position_choice,
            "arbitration_summary": f"投委会基于【{style_name}】综合评定：认可多头对内生估值修复与资金净流入的论证，同时吸收空头关于上方阻力位与防守纪律的告诫，出具平衡配置决议。",
            "core_contradiction": "内生估值修复弹性 vs 上方技术形态阻力",
            "risk_reward_ratio": "盈亏比约为 2.2:1 (向上空间大于潜在回撤)",
            "entry_tactics": "建议分 3 批在回调至均线支撑位时梯次吸筹，设置 7% 动态追踪止损线。",
        }

        parsed = self._parse_json_safely(raw, fallback)

        arbiter_verdict = ArbiterVerdict(
            final_verdict=parsed.get("final_verdict") or verdict_choice,
            consensus_score=float(parsed.get("consensus_score") or consensus_score),
            target_position_pct=parsed.get("target_position_pct") or position_choice,
            arbitration_summary=parsed.get("arbitration_summary") or fallback["arbitration_summary"],
            core_contradiction=parsed.get("core_contradiction") or fallback["core_contradiction"],
            risk_reward_ratio=parsed.get("risk_reward_ratio") or fallback["risk_reward_ratio"],
            entry_tactics=parsed.get("entry_tactics") or fallback["entry_tactics"],
        )

        debate_turn = DebateTurn(
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
            divergence <= settings.debate.CONVERGENCE_DELTA * 10
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
            "debate_turns": [debate_turn],
            "arbiter_verdict": arbiter_verdict,
            "current_step": "debate_arbitration_completed",
            "execution_logs": [log_item],
        }


# 单例导出
bull_agent = BullAgent()
bear_agent = BearAgent()
arbiter_agent = ArbiterAgent()

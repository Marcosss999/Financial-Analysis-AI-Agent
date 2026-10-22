"""
LangGraph 多智能体协同状态图编排模块 (Financial Multi-Agent StateGraph)
职责：
1. 构建基于 LangGraph StateGraph 的投委会多智能体有向状态机。
2. 编排【数据准备 -> 宏观/基本面/技术面专家特征抽取 -> 多空辩论博弈 -> 投委会仲裁 -> 风控硬拦截 -> 研报生成】完整拓扑。
3. 设立严密的条件分支路由 (Conditional Edges) 与辩论最大迭代轮次熔断锁，防止状态机无限递归死锁。
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Literal, Optional
from langgraph.graph import StateGraph, START, END
from app.core.config import settings
from app.core.observability import traced_node
from app.models.agent_state import FinancialAgentState
from app.tools.market_data import market_data_provider
from app.tools.market_intel import market_intel_provider
from app.agents.macro_agent import macro_agent
from app.agents.fundamental_agent import fundamental_agent
from app.agents.technical_agent import technical_agent
from app.agents.debate_engine import bull_agent, bear_agent, arbiter_agent
from app.agents.risk_guardrail import risk_guardrail_agent
from app.agents.reporter_agent import reporter_agent
from app.tools.scoring import aggregate, score_capital_flow


# =============================================================================
# 一、 节点处理函数封装 (Node Handlers)
# =============================================================================

async def init_data_node(state: FinancialAgentState) -> Dict[str, Any]:
    """初始化底数据节点：提取行情、财报与全网情报，设定初始步长与计数器"""
    ticker = state.get("ticker", "600519").strip().upper()
    target_name = state.get("target_name")

    # market_data / market_intel 内部是同步 requests 调用(最长数秒)，放到线程池执行，
    # 避免阻塞事件循环而拖慢同进程内其他并发请求(SSE 推送、其他分析任务)
    quote = state.get("quote") or await asyncio.to_thread(market_data_provider.get_market_quote, ticker, 60)
    financials = state.get("financials") or await asyncio.to_thread(market_data_provider.get_financial_statements, ticker)
    real_target_name = target_name or quote.name
    market_intel = state.get("market_intel") or await asyncio.to_thread(
        market_intel_provider.get_market_intel, ticker, real_target_name
    )

    style = state.get("strategy_style") or "balanced"
    style_presets = {
        # 与前端 PRESET_WEIGHTS 保持一致
        "momentum": {"fund": 0.15, "tech": 0.60, "macro": 0.10, "flow": 0.15},
        "balanced": {"fund": 0.40, "tech": 0.35, "macro": 0.15, "flow": 0.10},
        "deep_value": {"fund": 0.65, "tech": 0.15, "macro": 0.15, "flow": 0.05},
        "short_term": {"fund": 0.10, "tech": 0.50, "macro": 0.10, "flow": 0.30},
    }

    # 用户手动传参优先，否则取策略预设
    preset = style_presets.get(style, style_presets["balanced"])
    raw_fund = state.get("fundamental_weight") if state.get("fundamental_weight") is not None else preset["fund"]
    raw_tech = state.get("technical_weight") if state.get("technical_weight") is not None else preset["tech"]
    raw_macro = state.get("macro_weight") if state.get("macro_weight") is not None else preset["macro"]
    raw_flow = state.get("capital_flow_weight") if state.get("capital_flow_weight") is not None else preset["flow"]

    total_w = max(0.01, raw_fund + raw_tech + raw_macro + raw_flow)
    norm_fund = round(raw_fund / total_w, 4)
    norm_tech = round(raw_tech / total_w, 4)
    norm_macro = round(raw_macro / total_w, 4)
    norm_flow = round(max(0.0, 1.0 - norm_fund - norm_tech - norm_macro), 4)

    return {
        "ticker": ticker,
        "target_name": real_target_name,
        "quote": quote,
        "financials": financials,
        "market_intel": market_intel,
        "strategy_style": style,
        "fundamental_weight": norm_fund,
        "technical_weight": norm_tech,
        "macro_weight": norm_macro,
        "capital_flow_weight": norm_flow,
        "indicator_weights": {
            "fundamental": norm_fund,
            "technical": norm_tech,
            "macro": norm_macro,
            "capital_flow": norm_flow,
        },
        "current_debate_round": 0,
        "max_debate_rounds": settings.debate.MAX_DEBATE_ROUNDS,
        "is_debate_converged": False,
        "current_step": "data_preparation_completed",
        "execution_logs": [{
            "node": "init_data",
            "status": "success",
            "summary": f"已成功加载标的 {real_target_name} ({ticker}) 行情、财报与多源情报 (资金: {market_intel.main_capital_flow} | 官方公告: {len(market_intel.announcements)}条 | 媒体要闻: {len(market_intel.news_headlines)}条)，策略: [{style}], 权重: [基本面 {int(norm_fund*100)}% | 技术动量 {int(norm_tech*100)}% | 宏观政策 {int(norm_macro*100)}% | 资金博弈 {int(norm_flow*100)}%]",
        }],
    }


# 三位专家并行时，多个节点在同一个超步里同时写状态。LangGraph 对“没有 reducer 的 key”只允许
# 同一超步内有一个写入者，否则抛 InvalidUpdateError。所以专家节点只能写各自独有的 key：
# quote / financials 在 init 节点已写入、current_step 由 join 节点统一写，这里一律剔除。
_SHARED_KEYS = ("quote", "financials", "current_step")


def _only_own_keys(update: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in update.items() if k not in _SHARED_KEYS}


def _degrade(node: str, output_key: str):
    """专家节点异常时的降级结果：该专家产出置空，记录降级，整条分析继续(下游对缺失的专家有默认处理)"""
    def _fn(state: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
        return {
            output_key: None,
            "degraded_nodes": [node],
            "execution_logs": [{
                "node": node, "timestamp": datetime.now().isoformat(), "status": "degraded",
                "summary": f"节点执行异常，已降级跳过: {type(exc).__name__}: {exc}",
            }],
        }
    return _fn


async def _macro_raw(state: FinancialAgentState) -> Dict[str, Any]:
    """宏观策略专家节点"""
    return _only_own_keys(await macro_agent.run(state))


async def _fundamental_raw(state: FinancialAgentState) -> Dict[str, Any]:
    """基本面与估值专家节点 (内含杜邦拆解与 DCF 模型)"""
    return _only_own_keys(await fundamental_agent.run(state))


async def _technical_raw(state: FinancialAgentState) -> Dict[str, Any]:
    """量化技术形态专家节点 (内含 MACD/RSI/布林带)"""
    return _only_own_keys(await technical_agent.run(state))


macro_node = traced_node("macro_node", _macro_raw, degrade_to=_degrade("macro_node", "macro_output"))
fundamental_node = traced_node("fundamental_node", _fundamental_raw,
                               degrade_to=_degrade("fundamental_node", "fundamental_output"))
technical_node = traced_node("technical_node", _technical_raw,
                             degrade_to=_degrade("technical_node", "technical_output"))


async def experts_join_node(state: FinancialAgentState) -> Dict[str, Any]:
    """
    汇合节点(fan-in)：三位专家全部完成后触发。
    在此把“用户权重 × 各维度量化评分”算成基础分——决策只依赖这里的确定性计算，与 LLM 是否在线无关。
    """
    got = [k for k in ("macro_output", "fundamental_output", "technical_output") if state.get(k) is not None]

    def _q(key: str):
        o = state.get(key)
        return None if o is None else o.quant_score

    flow = score_capital_flow(state.get("market_intel"), state.get("indicators_result"),
                              (state.get("quote").market_cap if state.get("quote") else 0.0))
    scores = {
        "fundamental": _q("fundamental_output"),
        "technical": _q("technical_output"),
        "macro": _q("macro_output"),
        "capital_flow": flow[0] if flow else None,
    }
    w = state.get("indicator_weights") or {}
    agg = aggregate(scores, {"fundamental": w.get("fundamental", 0.0), "technical": w.get("technical", 0.0),
                             "macro": w.get("macro", 0.0), "capital_flow": w.get("capital_flow", 0.0)})
    return {
        "current_step": "experts_analysis_completed",
        "dimension_scores": scores,
        "effective_weights": agg["effective_weights"],
        "base_score": agg["base_score"],
        "score_breakdown": agg["breakdown"],
        "execution_logs": [{
            "node": "experts_join", "timestamp": datetime.now().isoformat(),
            "status": "success" if len(got) == 3 else "warning",
            "summary": (f"专家汇合：{len(got)}/3 位产出有效；基础分 {agg['base_score']}"
                        + (f"；无数据维度 {agg['missing']} 已剔除并重新分配权重" if agg["missing"] else "")),
        }],
    }


async def bull_node(state: FinancialAgentState) -> Dict[str, Any]:
    """多头首席辩手进攻节点：自增当轮辩论计数器并陈述"""
    current_round = state.get("current_debate_round", 0) + 1
    state_temp = dict(state)
    state_temp["current_debate_round"] = current_round
    bull_res = await bull_agent.argue(state_temp)
    bull_res["current_debate_round"] = current_round
    bull_res["current_step"] = f"bull_debate_round_{current_round}"
    return bull_res


async def bear_node(state: FinancialAgentState) -> Dict[str, Any]:
    """空头首席辩手防守节点"""
    bear_res = await bear_agent.argue(state)
    bear_res["current_step"] = f"bear_debate_round_{state.get('current_debate_round', 1)}"
    return bear_res


async def arbiter_node(state: FinancialAgentState) -> Dict[str, Any]:
    """投委会主席仲裁节点"""
    return await arbiter_agent.run(state)


async def risk_node(state: FinancialAgentState) -> Dict[str, Any]:
    """合规风控硬拦截沙箱节点 (内含 10,000 次蒙特卡洛 VaR 与一票否决)"""
    return await risk_guardrail_agent.run(state)


def render_memory_section(state: FinancialAgentState) -> str:
    """长期记忆：与同一标的上一次分析的结论对比"""
    hist = state.get("history_runs") or []
    arb = state.get("arbiter_verdict")
    if not hist or arb is None:
        return "\n\n---\n### 🧠 历史记忆对比\n\n记忆库中暂无该标的的历史分析记录，本次为首次分析。\n"
    last = hist[0]
    ts = datetime.fromtimestamp(last.get("created_at", 0)).strftime("%Y-%m-%d %H:%M")
    prev_score = last.get("consensus_score")
    delta = ""
    if isinstance(prev_score, (int, float)):
        d = round(arb.consensus_score - prev_score, 1)
        delta = f"，共识分变化 **{d:+.1f}** ({prev_score} → {arb.consensus_score})"
    changed = "结论发生变化" if last.get("verdict") != arb.final_verdict else "结论与上次一致"
    return (
        "\n\n---\n### 🧠 历史记忆对比\n\n"
        f"- 上一次分析时间：{ts}（风格：{last.get('strategy_style')}）\n"
        f"- 上次结论：{last.get('verdict')}，仓位 {last.get('position')}，风控 {last.get('risk_status')}\n"
        f"- 本次结论：{arb.final_verdict}，仓位 {arb.target_position_pct}\n"
        f"- 对比：{changed}{delta}。共检索到 {len(hist)} 条同标的历史记录。\n"
    )


def render_data_banner(state: FinancialAgentState) -> str:
    """数据质量横幅：模拟数据 / 估算字段 / 节点降级，必须在报告顶部醒目告知读者"""
    notes: List[str] = []
    quote, fin = state.get("quote"), state.get("financials")
    if quote is not None and getattr(quote, "data_source", "real") != "real":
        notes.append(f"行情数据来源：{quote.data_source}（非实时真实数据）")
    if fin is not None and getattr(fin, "data_source", "real") != "real":
        notes.append("财报数据为离线模拟数据")
    elif fin is not None and getattr(fin, "estimated_fields", None):
        notes.append("财报中 " + "、".join(fin.estimated_fields) + " 为按经验系数推算的估算值，非披露值")
    macro = state.get("macro_output")
    idx_src = (macro.details or {}).get("index_data_source") if macro is not None else None
    if idx_src and idx_src != "real":
        notes.append(f"宏观/市场环境所用沪深300指数数据来源：{idx_src}（非真实数据，该维度评分仅供演示）")
    intel = state.get("market_intel")
    if intel is not None and getattr(intel, "data_source", "real") != "real":
        notes.append("主力资金数据不可用，资金面维度已从加权中剔除")
    if state.get("degraded_nodes"):
        notes.append("以下节点执行异常已降级：" + "、".join(state["degraded_nodes"]))
    if not notes:
        return ""
    return "> ⚠️ **数据质量提示**：" + "；".join(notes) + "。相关结论仅供演示与学习，不构成投资依据。\n\n"


async def _reporter_raw(state: FinancialAgentState) -> Dict[str, Any]:
    """研报终稿排版生成节点：在研报基础上加数据质量横幅与历史记忆对比"""
    out = await reporter_agent.run(state)
    md = out.get("final_report_md") or ""
    out["final_report_md"] = render_data_banner(state) + md + render_memory_section(state)
    return out


reporter_node = traced_node("reporter_node", _reporter_raw)


# =============================================================================
# 二、 条件边路由判定逻辑 (Conditional Edge Routers)
# =============================================================================

def route_after_experts(state: FinancialAgentState) -> Literal["bull_node", "arbiter_node"]:
    """专家汇合后，判断是否启用多空辩论对决"""
    if state.get("enable_debate", True):
        return "bull_node"
    return "arbiter_node"


def route_after_arbiter(state: FinancialAgentState) -> Literal["bull_node", "risk_node"]:
    """
    仲裁后条件路由：
    若未收敛且未达到最大轮次，则循环回多头节点继续辩论；
    否则，硬性跳出循环，流转至合规风控沙箱。
    """
    # 关闭辩论时不得回到辩论环 (旧版本这里漏判：关闭辩论后仲裁节点会因“未收敛且轮次不足”跳回多头节点)
    if not state.get("enable_debate", True):
        return "risk_node"

    is_converged = state.get("is_debate_converged", False)
    current_round = state.get("current_debate_round", 1)
    max_rounds = state.get("max_debate_rounds", settings.debate.MAX_DEBATE_ROUNDS)

    if not is_converged and current_round < max_rounds:
        return "bull_node"
    return "risk_node"


# =============================================================================
# 三、 构建与编译 LangGraph 状态图
# =============================================================================

def build_financial_graph(parallel: Optional[bool] = None):
    """
    组装并编译全流程金融投研状态图。
    parallel=True : init → [宏观 ∥ 基本面 ∥ 技术面] → 汇合节点 (fan-out / fan-in，三位专家同一超步并发执行)
    parallel=False: init → 宏观 → 基本面 → 技术面 → 汇合节点 (串行，用于对比基准与出问题时回退)
    默认读取配置 PARALLEL_EXPERTS。
    """
    if parallel is None:
        parallel = settings.eng.PARALLEL_EXPERTS
    workflow = StateGraph(FinancialAgentState)

    # 1. 注册核心功能节点 (除专家节点已在定义处包装外，其余节点在此包装 trace span)
    workflow.add_node("init_data_node", traced_node("init_data_node", init_data_node))
    workflow.add_node("macro_node", macro_node)
    workflow.add_node("fundamental_node", fundamental_node)
    workflow.add_node("technical_node", technical_node)
    workflow.add_node("experts_join_node", traced_node("experts_join_node", experts_join_node))
    workflow.add_node("bull_node", traced_node("bull_node", bull_node))
    workflow.add_node("bear_node", traced_node("bear_node", bear_node))
    workflow.add_node("arbiter_node", traced_node("arbiter_node", arbiter_node))
    workflow.add_node("risk_node", traced_node("risk_node", risk_node))
    workflow.add_node("reporter_node", reporter_node)

    # 2. 入口与专家分析阶段拓扑
    workflow.add_edge(START, "init_data_node")
    if parallel:
        # 三位专家互不依赖对方产出(只读 init 写入的行情/财报/情报)，因此可以并发
        for expert in ("macro_node", "fundamental_node", "technical_node"):
            workflow.add_edge("init_data_node", expert)
        # 列表形式的 add_edge = 等待全部上游完成后才触发 (join 语义)
        workflow.add_edge(["macro_node", "fundamental_node", "technical_node"], "experts_join_node")
    else:
        workflow.add_edge("init_data_node", "macro_node")
        workflow.add_edge("macro_node", "fundamental_node")
        workflow.add_edge("fundamental_node", "technical_node")
        workflow.add_edge("technical_node", "experts_join_node")

    # 3. 条件分支：进入多空辩论对决或跳过
    workflow.add_conditional_edges(
        "experts_join_node",
        route_after_experts,
        {"bull_node": "bull_node", "arbiter_node": "arbiter_node"},
    )

    # 4. 辩论内部时序边
    workflow.add_edge("bull_node", "bear_node")
    workflow.add_edge("bear_node", "arbiter_node")

    # 5. 辩论收敛与熔断条件边 (带环循环)
    workflow.add_conditional_edges(
        "arbiter_node",
        route_after_arbiter,
        {"bull_node": "bull_node", "risk_node": "risk_node"},
    )

    # 6. 风控穿透与研报生成终局边
    workflow.add_edge("risk_node", "reporter_node")
    workflow.add_edge("reporter_node", END)

    return workflow.compile()


# 全局编译好的状态图实例
financial_graph = build_financial_graph()

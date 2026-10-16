"""
LangGraph 多智能体协同状态图编排模块 (Financial Multi-Agent StateGraph)
职责：
1. 构建基于 LangGraph StateGraph 的投委会多智能体有向状态机。
2. 编排【数据准备 -> 宏观/基本面/技术面专家特征抽取 -> 多空辩论博弈 -> 投委会仲裁 -> 风控硬拦截 -> 研报生成】完整拓扑。
3. 设立严密的条件分支路由 (Conditional Edges) 与辩论最大迭代轮次熔断锁，防止状态机无限递归死锁。
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from app.core.config import settings
from app.models.agent_state import FinancialAgentState
from app.tools.market_data import market_data_provider
from app.tools.market_intel import market_intel_provider
from app.agents.macro_agent import macro_agent
from app.agents.fundamental_agent import fundamental_agent
from app.agents.technical_agent import technical_agent
from app.agents.debate_engine import bull_agent, bear_agent, arbiter_agent
from app.agents.risk_guardrail import risk_guardrail_agent
from app.agents.reporter_agent import reporter_agent


# =============================================================================
# 一、 节点处理函数封装 (Node Handlers)
# =============================================================================

async def init_data_node(state: FinancialAgentState) -> Dict[str, Any]:
    """初始化底数据节点：提取行情、财报与全网情报，设定初始步长与计数器"""
    ticker = state.get("ticker", "600519").strip().upper()
    target_name = state.get("target_name")

    quote = state.get("quote") or market_data_provider.get_market_quote(ticker, days=60)
    financials = state.get("financials") or market_data_provider.get_financial_statements(ticker)
    real_target_name = target_name or quote.name
    market_intel = state.get("market_intel") or market_intel_provider.get_market_intel(ticker, real_target_name)

    style = state.get("strategy_style") or "balanced"
    style_presets = {
        "momentum": {"fund": 0.10, "tech": 0.60, "macro": 0.10, "flow": 0.20},
        "balanced": {"fund": 0.40, "tech": 0.35, "macro": 0.15, "flow": 0.10},
        "deep_value": {"fund": 0.65, "tech": 0.15, "macro": 0.15, "flow": 0.05},
        "short_term": {"fund": 0.05, "tech": 0.70, "macro": 0.05, "flow": 0.20},
    }

    # 用户手动传参优先，否则取策略预设
    preset = style_presets.get(style, style_presets["balanced"])
    raw_fund = state.get("fundamental_weight") if state.get("fundamental_weight") is not None else preset["fund"]
    raw_tech = state.get("technical_weight") if state.get("technical_weight") is not None else preset["tech"]
    raw_macro = state.get("macro_weight") if state.get("macro_weight") is not None else preset["macro"]
    raw_flow = state.get("capital_flow_weight") if state.get("capital_flow_weight") is not None else preset["flow"]

    total_w = max(0.01, raw_fund + raw_tech + raw_macro + raw_flow)
    norm_fund = round(raw_fund / total_w, 2)
    norm_tech = round(raw_tech / total_w, 2)
    norm_macro = round(raw_macro / total_w, 2)
    norm_flow = round(1.0 - norm_fund - norm_tech - norm_macro, 2)
    if norm_flow < 0:
        norm_flow = 0.0

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


async def macro_node(state: FinancialAgentState) -> Dict[str, Any]:
    """宏观策略专家节点"""
    return await macro_agent.run(state)


async def fundamental_node(state: FinancialAgentState) -> Dict[str, Any]:
    """基本面与估值专家节点 (内含杜邦拆解与 DCF 模型)"""
    return await fundamental_agent.run(state)


async def technical_node(state: FinancialAgentState) -> Dict[str, Any]:
    """量化技术形态专家节点 (内含 MACD/RSI/布林带)"""
    return await technical_agent.run(state)


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


async def reporter_node(state: FinancialAgentState) -> Dict[str, Any]:
    """研报终稿排版生成节点"""
    return await reporter_agent.run(state)


# =============================================================================
# 二、 条件边路由判定逻辑 (Conditional Edge Routers)
# =============================================================================

def route_after_technical(state: FinancialAgentState) -> Literal["bull_node", "arbiter_node"]:
    """专家特征就绪后，判断是否启用多空辩论对决"""
    if state.get("enable_debate", True):
        return "bull_node"
    return "arbiter_node"


def route_after_arbiter(state: FinancialAgentState) -> Literal["bull_node", "risk_node"]:
    """
    仲裁后条件路由：
    若未收敛且未达到最大轮次，则循环回多头节点继续辩论；
    否则，硬性跳出循环，流转至合规风控沙箱。
    """
    is_converged = state.get("is_debate_converged", False)
    current_round = state.get("current_debate_round", 1)
    max_rounds = state.get("max_debate_rounds", settings.debate.MAX_DEBATE_ROUNDS)

    if not is_converged and current_round < max_rounds:
        return "bull_node"
    return "risk_node"


# =============================================================================
# 三、 构建与编译 LangGraph 状态图
# =============================================================================

def build_financial_graph():
    """组装并编译全流程金融投研状态图"""
    workflow = StateGraph(FinancialAgentState)

    # 1. 注册核心功能节点
    workflow.add_node("init_data_node", init_data_node)
    workflow.add_node("macro_node", macro_node)
    workflow.add_node("fundamental_node", fundamental_node)
    workflow.add_node("technical_node", technical_node)
    workflow.add_node("bull_node", bull_node)
    workflow.add_node("bear_node", bear_node)
    workflow.add_node("arbiter_node", arbiter_node)
    workflow.add_node("risk_node", risk_node)
    workflow.add_node("reporter_node", reporter_node)

    # 2. 设置入口与专家分析阶段拓扑边
    workflow.add_edge(START, "init_data_node")
    workflow.add_edge("init_data_node", "macro_node")
    workflow.add_edge("macro_node", "fundamental_node")
    workflow.add_edge("fundamental_node", "technical_node")

    # 3. 条件分支：进入多空辩论对决或跳过
    workflow.add_conditional_edges(
        "technical_node",
        route_after_technical,
        {
            "bull_node": "bull_node",
            "arbiter_node": "arbiter_node",
        },
    )

    # 4. 辩论内部时序边
    workflow.add_edge("bull_node", "bear_node")
    workflow.add_edge("bear_node", "arbiter_node")

    # 5. 辩论收敛与熔断条件边 (带环循环)
    workflow.add_conditional_edges(
        "arbiter_node",
        route_after_arbiter,
        {
            "bull_node": "bull_node",
            "risk_node": "risk_node",
        },
    )

    # 6. 风控穿透与研报生成终局边
    workflow.add_edge("risk_node", "reporter_node")
    workflow.add_edge("reporter_node", END)

    return workflow.compile()


# 导出编译好的全局状态图单例
financial_graph = build_financial_graph()

"""
LangGraph 投研状态机共享状态契约模块 (Agent State Definition)
职责：
1. 定义多智能体在 LangGraph 拓扑网络中流转的全局共享状态 (FinancialAgentState)。
2. 利用 Annotated 与 operator.add 实现增量聚合 Reducer（如日志轨迹 execution_logs、多空辩论历史 debate_turns）。
3. 严格维护辩论迭代轮次计数器 (current_debate_round) 与防死循环收敛标记 (is_debate_converged)。
4. 挂载确定性量化指标计算结果、多专家结论、投委会裁决及风控硬拦截状态。
"""

import operator
from typing import Annotated, Dict, Any, List, Optional
from typing_extensions import TypedDict
from app.models.schemas import (
    MarketQuote,
    MarketIntel,
    FinancialStatements,
    DuPontResult,
    DCFValuationResult,
    QuantIndicatorsResult,
    MonteCarloVaRResult,
    SpecialistAgentOutput,
    DebateTurn,
    ArbiterVerdict,
    RiskAuditResult,
)


class FinancialAgentState(TypedDict, total=False):
    """
    LangGraph 全局共享工作流状态契约
    每个节点可返回字典增量更新该状态。
    带有 Annotated[List, operator.add] 标记的字段将在节点输出时自动执行追加（Append），而非覆盖。
    """

    # 1. 基础任务入参
    ticker: str                             # 标的股票代码 (如 600519)
    target_name: str                        # 标的公司简称 (如 贵州茅台)
    custom_query: Optional[str]             # 用户附加分析指令
    enable_debate: bool                     # 是否启用多空辩论委员会
    strategy_style: Optional[str]           # 策略风格模式：balanced, momentum, deep_value, short_term, custom
    fundamental_weight: Optional[float]     # 基本面分析权重 (0.0 ~ 1.0)
    technical_weight: Optional[float]       # 技术动量分析权重 (0.0 ~ 1.0)
    macro_weight: Optional[float]           # 宏观政策分析权重 (0.0 ~ 1.0)
    capital_flow_weight: Optional[float]    # 资金博弈与预期透支权重 (0.0 ~ 1.0)
    indicator_weights: Optional[Dict[str, float]] # 归一化后的四维权重字典

    # 2. 外部金融数据与多源情报 (DataOps, Quant & Market Intel)
    quote: Optional[MarketQuote]            # 实时行情与近期K线
    financials: Optional[FinancialStatements]  # 财务三表结构化数据
    market_intel: Optional[MarketIntel]     # 多源情报：东财分时主力资金、官方重大公告与新浪快讯
    dupont_result: Optional[DuPontResult]   # 杜邦三因子拆解计算结果
    dcf_result: Optional[DCFValuationResult]# DCF 现金流估值结果
    indicators_result: Optional[QuantIndicatorsResult] # MACD/RSI/布林带量化结果
    var_result: Optional[MonteCarloVaRResult]           # 蒙特卡洛 VaR 模拟结果

    # 3. 专业分析师 Agent 产出
    macro_output: Optional[SpecialistAgentOutput]       # 宏观政策分析师结论
    fundamental_output: Optional[SpecialistAgentOutput] # 基本面分析师结论
    technical_output: Optional[SpecialistAgentOutput]   # 量化技术面分析师结论

    # 4. 多空辩论博弈状态 (Debate Engine)
    bull_argument: Optional[str]            # 多头最新论点
    bull_score: float                       # 多头置信分
    bull_catalysts: Optional[List[str]]     # 多头进攻催化剂列表
    bear_argument: Optional[str]            # 空头最新反驳
    bear_score: float                       # 空头置信分
    bear_vulnerabilities: Optional[List[str]] # 空头阿喀琉斯之踵弱点列表
    current_debate_round: int               # 当前已执行辩论轮次
    max_debate_rounds: int                  # 允许最大辩论轮次上限 (熔断保障)
    is_debate_converged: bool               # 是否已提前达成共识收敛
    debate_turns: Annotated[List[DebateTurn], operator.add] # 辩论历史时序 (追加 Reducer)

    # 5. 投委会仲裁与风控拦截 (Committee & Risk)
    arbiter_verdict: Optional[ArbiterVerdict] # 投委会综合裁定
    risk_audit: Optional[RiskAuditResult]    # 合规风控穿透审核
    is_risk_violated: bool                  # 是否触发硬风控红线一票否决
    
    # 6. 研报产出与链路可观测日志
    final_report_md: Optional[str]          # 终版结构化研报 Markdown
    current_step: str                       # 当前执行节点标识
    execution_logs: Annotated[List[Dict[str, Any]], operator.add] # 链路执行日志追踪
    error: Optional[str]                    # 异常错误记录

"""
金融分析数据传输与模型契约契约模块 (Schemas & DTOs)
职责：
1. 定义标准金融行情 (KLine / MarketQuote) 与财报三表基础数据契约。
2. 规范四大确定性量化算法输入输出契约 (杜邦三因子拆解、DCF折现、技术指标、蒙特卡洛 VaR)。
3. 定义专业专家 Agent、多空辩论回合 (DebateTurn)、投委会仲裁与风控拦截审计结构。
4. 规范 FastAPI 外部请求与全流程分析响应 DTO。
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


_A_SHARE = re.compile(r"^(SH|SZ|BJ)?\d{6}$")
_US_HK = re.compile(r"^(US|HK)?[A-Z][A-Z0-9.\-]{0,9}$|^(HK)\d{5}$")


def validate_ticker(raw: str) -> str:
    """
    校验并规范化标的代码。
    A 股必须是 6 位数字(6 开头沪、0/3 开头深、4/8 开头北)；美股/港股为字母代码。
    不合法的代码直接拒绝，避免对不存在的标的生成一份看似正常的研报。
    """
    v = (raw or "").strip().upper()
    if _A_SHARE.match(v):
        digits = v[-6:]
        if digits[0] not in "60348":
            raise ValueError(f"无法识别的 A 股代码: {raw}")
        return v
    if v.isdigit():
        raise ValueError(f"A 股代码应为 6 位数字: {raw}")
    if _US_HK.match(v):
        return v
    raise ValueError(f"非法的标的代码: {raw}")


# =============================================================================
# 一、 基础行情与财报数据契约
# =============================================================================

class KLineBar(BaseModel):
    """单根 K 线数据"""
    date: str = Field(description="交易日期，格式 YYYY-MM-DD")
    open: float = Field(description="开盘价")
    high: float = Field(description="最高价")
    low: float = Field(description="最低价")
    close: float = Field(description="收盘价")
    volume: float = Field(description="成交量(股/手)")
    turnover: float = Field(default=0.0, description="成交金额(元)")


class MarketQuote(BaseModel):
    """标的股票即时行情与基本特征"""
    ticker: str = Field(description="股票代码，如 600519 / 000001 / AAPL")
    name: str = Field(description="证券简称")
    current_price: float = Field(description="最新收盘价/现价")
    change_pct: float = Field(description="涨跌幅 (%)")
    pe_ttm: float = Field(description="市盈率 TTM")
    pb_ratio: float = Field(description="市净率 PB")
    market_cap: float = Field(description="总市值 (亿元)")
    turnover_rate: float = Field(default=1.2, description="换手率 (%)")
    kline_history: List[KLineBar] = Field(default_factory=list, description="近期历史日 K 线时序")
    data_source: str = Field(default="real", description="数据来源：real(实时接口) / simulated(离线模拟，仅供演示)")


class MarketIntel(BaseModel):
    """标的多源情报与舆情（东财主力资金流向 + 官方重大公告 + 新浪舆情快讯）"""
    main_capital_flow: str = Field(default="暂无数据", description="主力资金净流入/流出情况")
    capital_inflow_amount_str: str = Field(default="0.0 万元", description="主力大单+超大单净额展示")
    is_main_inflow: bool = Field(default=False, description="主力是否为净流入")
    announcements: List[str] = Field(default_factory=list, description="东方财富官方重大公告列表")
    news_headlines: List[str] = Field(default_factory=list, description="新浪财经/行业快讯列表")
    flow_net_yuan: Optional[float] = Field(default=None, description="主力净流入金额(元，负为流出)；None 表示数据源不可用")
    data_source: str = Field(default="real", description="资金流数据来源：real / unavailable")


class FinancialStatements(BaseModel):
    """公司财务报表三表关键指标集"""
    ticker: str = Field(description="标的代码")
    period: str = Field(description="财报报告期，如 2024Q3 / 2023Annual")
    revenue: float = Field(description="营业收入 (亿元)")
    revenue_yoy: float = Field(description="营业收入同比增长率 (%)")
    net_profit: float = Field(description="归母净利润 (亿元)")
    net_profit_yoy: float = Field(description="归母净利润同比增长率 (%)")
    gross_margin: float = Field(description="毛利率 (%)")
    net_margin: float = Field(description="销售净利率 (%)")
    total_assets: float = Field(description="总资产 (亿元)")
    total_liabilities: float = Field(description="总负债 (亿元)")
    debt_to_asset_ratio: float = Field(description="资产负债率 (%)")
    operating_cash_flow: float = Field(description="经营活动现金流净额 (亿元)")
    interest_expense: float = Field(default=1.0, description="利息费用 (亿元)")
    ebit: float = Field(description="息税前利润 EBIT (亿元)")
    roe: float = Field(description="净资产收益率 ROE (%)")
    data_source: str = Field(default="real", description="数据来源：real / simulated")
    estimated_fields: List[str] = Field(
        default_factory=list,
        description="并非来自财报原始披露、由其他字段按经验系数推算的字段名(如 ebit、interest_expense)",
    )


# =============================================================================
# 二、 确定性量化与金融算法输出契约
# =============================================================================

class DuPontResult(BaseModel):
    """杜邦三因子拆解分析结果"""
    net_profit_margin: float = Field(description="净利润率 (%) - 反映盈利能力")
    asset_turnover: float = Field(description="总资产周转率 (次) - 反映营运效率")
    equity_multiplier: float = Field(description="权益乘数 - 反映财务杠杆倍数")
    calculated_roe: float = Field(description="计算所得 ROE (%)")
    quality_verdict: str = Field(description="ROE 增长驱动归因判定")


class DCFValuationResult(BaseModel):
    """DCF 自由现金流折现估值输出"""
    wacc: float = Field(description="加权平均资本成本 WACC")
    terminal_growth_rate: float = Field(description="永续增长率 g")
    projected_fcfs: List[float] = Field(description="未来预测期自由现金流 (亿元)")
    terminal_value: float = Field(description="终值 TV (亿元)")
    enterprise_value: float = Field(description="企业价值 EV (亿元)")
    equity_value: float = Field(description="股权价值 (亿元)")
    intrinsic_value_per_share: float = Field(description="每股内在公允价值 (元)")
    current_price: float = Field(description="当前实际市价 (元)")
    margin_of_safety_pct: float = Field(description="安全边际比例 (%) - 正值为折价安全，负值为溢价泡沫")
    valuation_status: str = Field(description="估值定性：显著低估 / 合理区间 / 明显高估")


class QuantIndicatorsResult(BaseModel):
    """技术面指标运算集合"""
    macd_dif: float = Field(description="MACD 差离值 DIF")
    macd_dea: float = Field(description="MACD 讯号线 DEA")
    macd_bar: float = Field(description="MACD 柱状图 Histogram")
    rsi_14: float = Field(description="14周期 RSI 相对强弱指标")
    boll_upper: float = Field(description="布林带上轨")
    boll_middle: float = Field(description="布林带中轨 (MA20)")
    boll_lower: float = Field(description="布林带下轨")
    trend_signal: str = Field(description="综合量化信号：强力买入 / 多头蓄势 / 震荡整理 / 空头破位")
    support_level: float = Field(description="算法推导核心支撑位")
    resistance_level: float = Field(description="算法推导核心阻力位")
    # ---- 短线扩展指标 (日 K 即可计算)：波动率 / 量能 / 短期反转 ----
    last_close: float = Field(default=0.0, description="最新收盘价")
    ma20: float = Field(default=0.0, description="20 日均线")
    ma60: float = Field(default=0.0, description="60 日均线(数据不足时为可用区间均值)")
    atr_14: float = Field(default=0.0, description="14 日平均真实波幅 ATR (Wilder 平滑)")
    atr_pct: float = Field(default=0.0, description="ATR 占现价百分比 (%)")
    realized_vol_20: float = Field(default=0.0, description="20 日已实现波动率，年化 (%)")
    realized_vol_5: float = Field(default=0.0, description="5 日已实现波动率，年化 (%)")
    vol_regime: float = Field(default=1.0, description="波动率状态 = 5 日 RV / 20 日 RV；>1.5 表示波动正在放大")
    volume_ratio: float = Field(default=1.0, description="量比 = 最新成交量 / 前 5 日均量")
    obv_slope: float = Field(default=0.0, description="OBV 近 10 日归一化斜率 (>0 资金持续流入)")
    mfi_14: float = Field(default=50.0, description="14 日资金流量指标 MFI (量价版 RSI)")
    ret_5d: float = Field(default=0.0, description="近 5 日涨跌幅 (%)")
    ret_20d: float = Field(default=0.0, description="近 20 日涨跌幅 (%)")
    boll_pctb: float = Field(default=0.5, description="布林带 %B = (现价-下轨)/(上轨-下轨)")
    boll_bandwidth: float = Field(default=0.0, description="布林带带宽 (上轨-下轨)/中轨")


class MonteCarloVaRResult(BaseModel):
    """蒙特卡洛 VaR 风险价值模拟结果"""
    simulations: int = Field(default=10000, description="模拟随机游走路径条数")
    confidence_level: float = Field(default=0.95, description="置信度")
    horizon_days: int = Field(default=5, description="预测时间范围 (交易日)")
    var_percentage: float = Field(description="95% 置信度下的单周最大预期回撤损失率 (%)")
    var_absolute_loss: float = Field(description="对应单股预期最大损失额 (元)")
    worst_drawdown: float = Field(description="模拟路径中的极端单次最大跌幅 (%)")
    risk_category: str = Field(description="风控评级：低波动 / 中度波动 / 极端高危")


# =============================================================================
# 三、 Agent 决策、辩论与风控仲裁契约
# =============================================================================

class SpecialistAgentOutput(BaseModel):
    """单个专业分析师 Agent 的标准产出"""
    agent_name: str = Field(description="Agent 角色名")
    stance: str = Field(description="定性立场，如：积极推荐 / 中性观望 / 警惕规避")
    score: float = Field(ge=0.0, le=10.0, description="0~10 分的打分")
    summary: str = Field(description="核心定性分析陈述")
    details: Dict[str, Any] = Field(default_factory=dict, description="结构化支撑数据或指标字典")
    key_points: List[str] = Field(default_factory=list, description="核心关键论据清单")
    quant_score: Optional[float] = Field(default=None, description="确定性量化评分(0~10)；决策只使用它，与 LLM 是否在线无关")
    llm_score: Optional[float] = Field(default=None, description="LLM 的主观评分(仅 LLM 在线时有值)，仅作参考与分歧检测")
    score_components: Dict[str, float] = Field(default_factory=dict, description="量化评分的分项构成")
    score_source: str = Field(default="quant", description="score 字段的来源")


class DebateTurn(BaseModel):
    """单轮多空辩论（Bull vs Bear）记录"""
    round_index: int = Field(description="辩论轮次序号 (从 1 开始)")
    stage_name: str = Field(default="立论攻防阶段", description="辩论阶段名称：如首轮攻防立论 / 次轮针锋相对驳论")
    bull_thesis: str = Field(description="多头进攻论述")
    bull_score: float = Field(description="多头置信度得分 (0~10)")
    bull_catalysts: List[str] = Field(default_factory=list, description="多头核心上行催化剂清单")
    bear_antithesis: str = Field(description="空头防守反驳论述")
    bear_score: float = Field(description="空头置信度得分 (0~10)")
    bear_vulnerabilities: List[str] = Field(default_factory=list, description="空头指出的核心破绽与风险清单")
    divergence: float = Field(description="当轮多空分歧差值 abs(bull - bear)")
    battleground_issue: Optional[str] = Field(default=None, description="当轮交锋的核心焦点")


class ArbiterVerdict(BaseModel):
    """投委会仲裁官最终判定"""
    final_verdict: str = Field(description="终审决议：强烈推荐 / 逢低吸纳 / 中性持有 / 减持规避")
    consensus_score: float = Field(description="加权共识综合评分 (0~10)")
    target_position_pct: str = Field(description="投委会建议仓位配置区间，如 45%~55%")
    arbitration_summary: str = Field(description="多空分歧调和与裁决总结")
    core_contradiction: str = Field(default="成长估值弹性 vs 宏观周期防守", description="多空最核心的分歧对立面判定")
    risk_reward_ratio: str = Field(default="盈亏比合理 (约 2:1)", description="不对称风险收益比评估")
    entry_tactics: str = Field(description="建仓与交易执行节奏建议")
    base_score: Optional[float] = Field(default=None, description="四维度按权重加权的基础分(确定性)")
    debate_adjustment: float = Field(default=0.0, description="多空辩论带来的有界调整量")
    score_breakdown: List[Dict[str, Any]] = Field(default_factory=list, description="各维度得分 / 权重 / 贡献明细")
    atr_stop_price: Optional[float] = Field(default=None, description="ATR 止损价参考")
    atr_stop_pct: Optional[float] = Field(default=None, description="止损距离占现价百分比")
    risk_cap_position_pct: Optional[float] = Field(default=None, description="按单笔风险预算反推的仓位上限(%)")
    pre_veto_verdict: Optional[str] = Field(default=None, description="被风控否决前的原始裁决(审计留痕)")
    pre_veto_position: Optional[str] = Field(default=None, description="被风控否决前的原始建议仓位")


class RiskAuditResult(BaseModel):
    """合规风控硬拦截沙箱检验结果"""
    compliance_status: str = Field(description="合规状态：APPROVED(通过) / WARNING(警告通过) / VETOED(一票否决)")
    is_risk_violated: bool = Field(description="是否触发硬性风控红线拦截")
    risk_level: str = Field(description="综合风险评级：R1极低 / R2稳健 / R3平衡 / R4进取 / R5极高")
    audit_checklist: Dict[str, str] = Field(description="各项指标硬约束穿透核验记录")
    veto_reasons: List[str] = Field(default_factory=list, description="触发一票否决或降级告警的违规条款清单")
    skipped_checks: List[str] = Field(default_factory=list, description="因数据不可靠或未接入而未执行的检查项")


# =============================================================================
# 四、 接口层请求与响应契约
# =============================================================================

class AnalysisRequest(BaseModel):
    """投研分析发起请求"""
    ticker: str = Field(default="600519", description="股票代码 (如 600519, 300750, 000001, AAPL)")
    target_name: Optional[str] = Field(default=None, description="公司简称 (可选，留空则自动识别)")
    custom_query: Optional[str] = Field(default=None, description="定制化投研重点问询需求")
    enable_debate: bool = Field(default=True, description="是否启用多空对抗辩论委员会")
    strategy_style: Optional[str] = Field(default="balanced", description="策略风格模式：balanced(机构均衡), momentum(动量趋势突破), deep_value(深度价值), short_term(短线博弈), custom(自定义权重)")
    fundamental_weight: Optional[float] = Field(default=None, description="基本面分析权重 (0.0~1.0)；不传则使用 strategy_style 对应的预设")
    technical_weight: Optional[float] = Field(default=None, description="技术动量分析权重 (0.0~1.0)")
    macro_weight: Optional[float] = Field(default=None, description="宏观政策分析权重 (0.0~1.0)")
    capital_flow_weight: Optional[float] = Field(default=None, description="资金博弈与预期透支权重 (0.0~1.0)")

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str) -> str:
        return validate_ticker(v)

    @field_validator("fundamental_weight", "technical_weight", "macro_weight", "capital_flow_weight")
    @classmethod
    def _validate_weight(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("权重必须在 0.0 ~ 1.0 之间")
        return v


class AnalysisResponse(BaseModel):
    """投研分析最终产物响应契约"""
    ticker: str
    target_name: str
    current_price: float
    consensus_score: float
    verdict: str
    target_position: str
    risk_status: str
    is_vetoed: bool
    macro_score: float
    fundamental_score: float
    technical_score: float
    dcf_margin_of_safety: float
    var_95_drawdown: float
    full_report_md: str
    debate_history: List[DebateTurn] = Field(default_factory=list)
    arbiter_verdict: Optional[ArbiterVerdict] = None
    audit_details: Optional[RiskAuditResult] = None
    market_intel: Optional[MarketIntel] = None
    strategy_style: str = Field(default="balanced", description="执行的策略风格")
    weights_used: Dict[str, float] = Field(default_factory=dict, description="实际生效的归一化权重字典")
    execution_time_seconds: float = Field(default=0.0)
    run_id: str = Field(default="", description="本次分析的唯一 ID，可用于查询 trace 与历史记录")
    session_id: str = Field(default="", description="会话 ID，追问对话时带上即可延续记忆")
    data_quality: Dict[str, Any] = Field(
        default_factory=dict, description="数据质量：行情/财报来源、估算字段、LLM 降级次数、降级节点"
    )
    trace: Dict[str, Any] = Field(default_factory=dict, description="节点级耗时与 LLM 调用统计")
    memory: Dict[str, Any] = Field(default_factory=dict, description="与历史分析对比的长期记忆摘要")
    score_breakdown: List[Dict[str, Any]] = Field(default_factory=list, description="决策贡献表：维度 / 得分 / 权重 / 贡献")


# =============================================================================
# 六、 大模型 API Key 与网关动态配置契约
# =============================================================================

class LLMConfigReq(BaseModel):
    """设置 LLM 配置入参"""
    api_key: str = Field(description="大模型 API Key")
    base_url: str = Field(description="API Base URL, 例如 https://api.deepseek.com")
    model: str = Field(description="模型名称, 例如 deepseek-chat")
    temperature: float = Field(default=0.2, description="采样温度 (0.0 ~ 1.0)")


class LLMConfigTestReq(BaseModel):
    """测试 LLM 连通性入参"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class LLMConfigResp(BaseModel):
    """当前系统 LLM 状态与配置响应"""
    is_configured: bool
    api_key_masked: str
    api_key_raw: str = ""  # 已弃用：出于安全考虑不再回传明文 Key，仅保留字段以兼容旧前端
    base_url: str
    model: str
    temperature: float


# =============================================================================
# 七、 AI 投顾深度多轮追问对话契约
# =============================================================================

class StockChatRequest(BaseModel):
    """针对标的与研报上下文的追问对话入参"""
    ticker: str = Field(description="标的股票代码")
    target_name: Optional[str] = Field(default="", description="标的公司名称")
    question: str = Field(description="用户追问问题")
    report_context: Optional[str] = Field(default="", description="前序研报全文或核心背景")
    history: List[Dict[str, str]] = Field(default_factory=list, description="(兼容旧前端)客户端自带的历史对话；传了 session_id 时以服务端记忆为准")
    session_id: Optional[str] = Field(default=None, description="会话 ID；为空则服务端新建并在响应中返回")


"""
确定性量化评分与决策聚合 (Deterministic Scoring & Decision Aggregation)

设计原则
1. 决策只使用本模块产出的“量化评分”，与 LLM 是否在线无关：用户设定的权重 × 各维度量化评分 = 基础分。
   LLM 负责解读与论证，它给出的主观评分只用于展示以及与量化评分的“分歧检测”，不进入决策。
2. 多空辩论只对基础分做“有界调整”(±DEBATE_MAX_ADJUST)，所以权重永远主导结论，辩论的边际贡献也可以被单独度量。
3. 某一维度没有数据(如离线时拿不到资金流)，该维度不参与加权并把权重按比例分给其余维度，同时在结果里显式标记，
   而不是编一个默认分。
4. 下面的分段阈值是依据常识设定的先验规则，没有对历史数据拟合；它们是否真的有预测力，由 experiments/ 里的回测来检验。
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.models.schemas import (
    DCFValuationResult, DuPontResult, FinancialStatements, MarketIntel, QuantIndicatorsResult,
)

DIMENSIONS = ("fundamental", "technical", "macro", "capital_flow")
DIM_LABELS = {"fundamental": "基本面", "technical": "技术面", "macro": "宏观/市场环境", "capital_flow": "资金面"}


def _clip(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def interp(x: float, points: List[Tuple[float, float]]) -> float:
    """分段线性插值；超出两端取端点值（单调映射，便于解释与测试）"""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


# ---------------------------------------------------------------------------
# 一、 基本面：估值 / 盈利 / 成长 / 杠杆
# ---------------------------------------------------------------------------
def score_fundamental(dupont: DuPontResult, dcf: DCFValuationResult, fin: FinancialStatements,
                      is_financial_institution: bool = False) -> Tuple[float, Dict[str, float]]:
    prof = interp(dupont.calculated_roe, [(0, 1.5), (5, 3), (10, 5), (15, 6.5), (20, 8), (25, 9), (35, 9.5)])
    val = interp(dcf.margin_of_safety_pct, [(-50, 1), (-20, 2.5), (0, 5), (20, 7), (40, 8.5), (60, 9.5)])
    growth = interp(fin.net_profit_yoy, [(-30, 1), (0, 4), (10, 5.5), (20, 7), (40, 8.5)])
    limit = settings.risk.FINANCIAL_INSTITUTION_MAX_DEBT_PCT if is_financial_institution \
        else settings.risk.MAX_DEBT_TO_ASSET_RATIO * 100.0
    lev = interp(fin.debt_to_asset_ratio / limit, [(0.3, 8.5), (0.6, 6.5), (0.85, 4), (1.0, 2), (1.2, 0.5)])
    comp = {"valuation": round(val, 2), "profitability": round(prof, 2), "growth": round(growth, 2),
            "leverage": round(lev, 2)}
    return round(_clip(0.30 * val + 0.30 * prof + 0.20 * growth + 0.20 * lev), 2), comp


# ---------------------------------------------------------------------------
# 二、 技术面 (短线版)：趋势 / 动量-反转 / 量能 / 波动率
# ---------------------------------------------------------------------------
def score_technical(ind: QuantIndicatorsResult) -> Tuple[float, Dict[str, float]]:
    base = 5.0
    # 趋势：现价相对 MA20 的偏离、MA20 与 MA60 的排列、MACD 方向
    d20 = (ind.last_close / ind.ma20 - 1.0) * 100.0 if ind.ma20 > 0 else 0.0
    trend = max(-1.5, min(1.5, d20 * 0.25))
    trend += 0.5 if ind.ma20 > ind.ma60 else -0.5
    if ind.macd_dif > ind.macd_dea and ind.macd_bar > 0:
        trend += 0.7
    elif ind.macd_dif < ind.macd_dea and ind.macd_bar < 0:
        trend -= 0.7
    # 动量 / 短期反转：RSI 两端与近 5 日涨跌幅的极端值在短线上更倾向回归
    rsi = ind.rsi_14
    rsi_adj = interp(rsi, [(15, 1.0), (25, 0.5), (35, -0.3), (45, 0.0), (55, 0.0), (65, 0.6), (75, -0.7), (85, -1.5)])
    r5 = ind.ret_5d
    rev = interp(r5, [(-15, 0.8), (-6, 0.4), (-2, 0.0), (2, 0.0), (6, -0.4), (15, -1.0)])
    momentum = rsi_adj + rev
    # 量能：OBV 斜率、量比与价格方向是否配合、MFI 极端
    vol = max(-0.8, min(0.8, ind.obv_slope * 2.0))
    if ind.volume_ratio > 1.5:
        vol += 0.5 if ind.ret_5d > 0 else -0.5      # 放量上涨加分，放量下跌减分
    elif ind.volume_ratio < 0.6 and ind.ret_5d > 3:
        vol -= 0.3                                    # 缩量上涨：缺乏资金确认
    if ind.mfi_14 >= 85:
        vol -= 0.5
    elif ind.mfi_14 <= 15:
        vol += 0.5
    # 波动率：高波动、且波动正在放大时降低评分（短线风险更大）
    vola = interp(ind.realized_vol_20, [(10, 0.3), (20, 0.3), (35, 0.0), (50, -0.5), (70, -1.0)])
    if ind.vol_regime > 1.8:
        vola -= 0.5
    comp = {"trend": round(trend, 2), "momentum_reversal": round(momentum, 2),
            "volume": round(vol, 2), "volatility": round(vola, 2)}
    return round(_clip(base + trend + momentum + vol + vola), 2), comp


def legacy_technical_score(ind: QuantIndicatorsResult) -> float:
    """旧版“MACD + RSI + 布林带”信号映射成的分数，仅用于 experiments/ 的对照组"""
    golden = ind.macd_dif > ind.macd_dea and ind.macd_bar > 0
    death = ind.macd_dif < ind.macd_dea and ind.macd_bar < 0
    above_mid = ind.last_close >= ind.boll_middle
    if golden and rsi_gt(ind, 50) and above_mid:
        return 5.5 if ind.rsi_14 >= 75 else 8.0
    if golden and not above_mid:
        return 6.5
    if death and (not above_mid or ind.rsi_14 < 45):
        return 4.5 if ind.rsi_14 <= 30 else 2.5
    return 5.0


def rsi_gt(ind: QuantIndicatorsResult, v: float) -> bool:
    return ind.rsi_14 > v


# ---------------------------------------------------------------------------
# 三、 资金面：主力净流入金额相对市值的强度 + 量能确认
# ---------------------------------------------------------------------------
def score_capital_flow(intel: Optional[MarketIntel], ind: Optional[QuantIndicatorsResult],
                       market_cap_yi: float) -> Optional[Tuple[float, Dict[str, float]]]:
    if intel is None or intel.flow_net_yuan is None:
        return None  # 数据源不可用：不参与加权，由聚合函数重新分配权重
    scale = max(market_cap_yi * 1e8 * 0.0005, 1e6)       # 以市值的 0.05% 作为“显著净流入”的尺度
    strength = 4.0 * math.tanh(intel.flow_net_yuan / scale)
    confirm = 0.0
    if ind is not None and ind.volume_ratio > 1.5:
        confirm = 0.5 if intel.flow_net_yuan > 0 else -0.5
    return round(_clip(5.0 + strength + confirm), 2), {"strength": round(strength, 2), "volume_confirm": confirm}


# ---------------------------------------------------------------------------
# 四、 宏观/市场环境：宽基指数的趋势与波动状态 (有数据可算的部分)
# ---------------------------------------------------------------------------
def score_market_environment(idx: QuantIndicatorsResult) -> Tuple[float, Dict[str, float]]:
    trend = max(-1.5, min(1.5, idx.ret_20d * 0.15))
    trend += 0.8 if idx.ma20 > idx.ma60 else -0.8
    trend += 0.5 if idx.last_close > idx.ma60 else -0.5
    if idx.macd_dif > idx.macd_dea:
        trend += 0.4
    else:
        trend -= 0.4
    vola = interp(idx.realized_vol_20, [(8, 0.4), (15, 0.3), (25, -0.3), (35, -0.8)])
    return round(_clip(5.0 + trend + vola), 2), {"index_trend": round(trend, 2), "index_volatility": round(vola, 2)}


# ---------------------------------------------------------------------------
# 五、 聚合：权重 × 维度评分 → 基础分；辩论有界调整；档位与仓位映射
# ---------------------------------------------------------------------------
def aggregate(scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    :return: base_score / effective_weights / breakdown / missing
    缺失维度不参与加权，其权重按比例分给其余维度；全部缺失时返回中性 5.0 并标记。
    """
    avail = {k: v for k, v in scores.items() if v is not None and weights.get(k, 0.0) > 0}
    missing = [k for k in DIMENSIONS if scores.get(k) is None]
    total_w = sum(weights[k] for k in avail)
    if total_w <= 0:
        return {"base_score": 5.0, "effective_weights": {}, "breakdown": [], "missing": list(DIMENSIONS)}
    eff = {k: weights[k] / total_w for k in avail}
    base = sum(avail[k] * eff[k] for k in avail)
    breakdown = []
    for k in DIMENSIONS:
        if k in avail:
            breakdown.append({"dimension": k, "label": DIM_LABELS[k], "score": round(avail[k], 2),
                              "user_weight": round(weights.get(k, 0.0), 4), "effective_weight": round(eff[k], 4),
                              "contribution": round(avail[k] * eff[k], 3)})
        else:
            breakdown.append({"dimension": k, "label": DIM_LABELS[k], "score": None,
                              "user_weight": round(weights.get(k, 0.0), 4), "effective_weight": 0.0,
                              "contribution": 0.0,
                              "note": "无可用数据，已从加权中剔除" if k in missing else "权重为 0"})
    return {"base_score": round(base, 2), "effective_weights": eff, "breakdown": breakdown, "missing": missing}


def debate_adjustment(bull: float, bear: float, max_adj: Optional[float] = None) -> float:
    """多头信念 - 空头信念，线性映射到 [-max_adj, +max_adj]；bear 是“看空信念”，越高越看空"""
    m = settings.decision.DEBATE_MAX_ADJUST if max_adj is None else max_adj
    return round(max(-m, min(m, m * (bull - bear) / 10.0)), 2)


def verdict_from_score(score: float) -> Tuple[str, str]:
    d = settings.decision
    if score >= d.BAND_STRONG_BUY:
        return "强烈买入 (Strong Buy)", "60% ~ 75% (重仓配置 / 积极进攻)"
    if score >= d.BAND_OVERWEIGHT:
        return "逢低增配 (Overweight / Buy on Dips)", "45% ~ 55% (标准配置 / 稳健逢低吸筹)"
    if score >= d.BAND_NEUTRAL:
        return "中性持有 (Neutral / Hold)", "25% ~ 35% (防守性底仓 / 保持观望)"
    return "建议规避 (Underweight / Caution)", "10% 以下 (轻仓或空仓防守)"


def stance_from_score(score: float) -> str:
    if score >= 7.5:
        return "积极看多"
    if score >= 6.3:
        return "偏多"
    if score >= 4.8:
        return "中性"
    if score >= 3.5:
        return "偏空"
    return "明显看空"


def atr_risk_sizing(ind: Optional[QuantIndicatorsResult]) -> Dict[str, Optional[float]]:
    """ATR 止损与仓位上限：止损距离 = k×ATR；仓位上限 = 单笔风险预算 ÷ 止损距离(占现价%)"""
    if ind is None or ind.atr_14 <= 0 or ind.last_close <= 0:
        return {"atr_stop_price": None, "atr_stop_pct": None, "risk_cap_position_pct": None}
    dist = settings.decision.ATR_STOP_MULT * ind.atr_14
    stop_pct = dist / ind.last_close * 100.0
    cap = min(100.0, settings.decision.RISK_PER_TRADE_PCT / stop_pct * 100.0)
    return {"atr_stop_price": round(ind.last_close - dist, 2), "atr_stop_pct": round(stop_pct, 2),
            "risk_cap_position_pct": round(cap, 1)}

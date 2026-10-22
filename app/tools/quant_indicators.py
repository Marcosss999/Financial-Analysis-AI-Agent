"""
确定性量化技术指标计算引擎 (Quant Technical Indicators Engine)
职责：
1. 纯数学算法实现 MACD (平滑异同移动平均线)、Wilder 标准平滑 RSI (相对强弱指标) 与 Bollinger Bands (布林带)。
2. 推导关键成交支撑位与阻力位，结合多指标共振综合判定当前技术形态走势信号。
3. 纯 Python 确定性运算，杜绝大模型在时序 K 线数值推演中的幻觉与错误信号误报。
"""

import math
from typing import List
from app.core.config import settings
from app.models.schemas import KLineBar, QuantIndicatorsResult


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    """样本标准差 (ddof=1)"""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Wilder 平滑 ATR：TR = max(H-L, |H-前收|, |L-前收|)；首个 ATR 取前 period 根 TR 均值，之后递推平滑"""
    if len(closes) < 2:
        return 0.0
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
           for i in range(1, len(closes))]
    if len(trs) <= period:
        return _mean(trs)
    atr = _mean(trs[:period])
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def compute_realized_vol(closes: List[float], window: int) -> float:
    """近 window 个交易日对数收益率的年化波动率 (%)"""
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    rets = rets[-window:]
    return _std(rets) * math.sqrt(252.0) * 100.0


def compute_obv_slope(closes: List[float], volumes: List[float], lookback: int = 10) -> float:
    """OBV 近 lookback 日的归一化斜率 = OBV 变化量 / 同期总成交量，取值 [-1, 1]"""
    if len(closes) < lookback + 1:
        return 0.0
    obv = [0.0]
    for i in range(1, len(closes)):
        step = volumes[i] if closes[i] > closes[i - 1] else (-volumes[i] if closes[i] < closes[i - 1] else 0.0)
        obv.append(obv[-1] + step)
    total = sum(volumes[-lookback:])
    return (obv[-1] - obv[-lookback - 1]) / total if total > 0 else 0.0


def compute_mfi(highs, lows, closes, volumes, period: int = 14) -> float:
    """资金流量指标 MFI：以典型价 x 成交量为资金流，是带成交量的 RSI"""
    if len(closes) < 2:
        return 50.0
    tp = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    pos = neg = 0.0
    for i in range(max(1, len(tp) - period), len(tp)):
        flow = tp[i] * volumes[i]
        if tp[i] > tp[i - 1]:
            pos += flow
        elif tp[i] < tp[i - 1]:
            neg += flow
    if neg == 0:
        return 100.0 if pos > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + pos / neg)


class QuantIndicatorsEngine:
    """确定性量化技术指标计算器"""

    @staticmethod
    def _calculate_ema(series: List[float], period: int) -> List[float]:
        """计算指数移动平均线 (EMA)"""
        if not series:
            return []
        alpha = 2.0 / (period + 1.0)
        ema = [series[0]]
        for price in series[1:]:
            ema.append(alpha * price + (1.0 - alpha) * ema[-1])
        return ema

    @classmethod
    def calculate_indicators(cls, klines: List[KLineBar]) -> QuantIndicatorsResult:
        """
        基于历史 K 线列表计算当期 MACD、RSI 与布林带指标
        :param klines: 按时间递增排序的 K 线列表 (建议不少于 30 根)
        :return: QuantIndicatorsResult
        """
        if not klines or len(klines) < 15:
            # 数据量极小兜底
            return QuantIndicatorsResult(
                macd_dif=0.0,
                macd_dea=0.0,
                macd_bar=0.0,
                rsi_14=50.0,
                boll_upper=100.0,
                boll_middle=95.0,
                boll_lower=90.0,
                trend_signal="数据不足-震荡观望",
                support_level=90.0,
                resistance_level=100.0,
            )

        closes = [bar.close for bar in klines]
        highs = [bar.high for bar in klines]
        lows = [bar.low for bar in klines]
        latest_close = closes[-1]

        # ---------------------------------------------------------------------
        # 1. MACD 计算 (默认 12, 26, 9)
        # ---------------------------------------------------------------------
        fast_period = settings.quant.MACD_FAST
        slow_period = settings.quant.MACD_SLOW
        signal_period = settings.quant.MACD_SIGNAL

        ema_fast = cls._calculate_ema(closes, fast_period)
        ema_slow = cls._calculate_ema(closes, slow_period)
        dif_series = [f - s for f, s in zip(ema_fast, ema_slow)]
        dea_series = cls._calculate_ema(dif_series, signal_period)

        latest_dif = dif_series[-1]
        latest_dea = dea_series[-1]
        latest_macd_bar = (latest_dif - latest_dea) * 2.0

        # ---------------------------------------------------------------------
        # 2. Wilder 标准平滑 RSI-14 计算
        # ---------------------------------------------------------------------
        rsi_period = settings.quant.RSI_PERIOD
        gains: List[float] = []
        losses: List[float] = []

        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            if delta >= 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(delta))

        if len(gains) < rsi_period:
            avg_gain = sum(gains) / max(1, len(gains))
            avg_loss = sum(losses) / max(1, len(losses))
        else:
            avg_gain = sum(gains[:rsi_period]) / rsi_period
            avg_loss = sum(losses[:rsi_period]) / rsi_period
            for g, l in zip(gains[rsi_period:], losses[rsi_period:]):
                avg_gain = (avg_gain * (rsi_period - 1) + g) / rsi_period
                avg_loss = (avg_loss * (rsi_period - 1) + l) / rsi_period

        if avg_loss == 0:
            rsi_14 = 100.0
        else:
            rs = avg_gain / max(0.0001, avg_loss)
            rsi_14 = 100.0 - (100.0 / (1.0 + rs))

        # ---------------------------------------------------------------------
        # 3. 布林带 (BOLL) 计算 (默认 20 周期, 2 倍标准差)
        # ---------------------------------------------------------------------
        boll_period = min(len(closes), settings.quant.BOLL_PERIOD)
        boll_slice = closes[-boll_period:]
        boll_middle = sum(boll_slice) / boll_period

        variance = sum((p - boll_middle) ** 2 for p in boll_slice) / boll_period
        std_dev = math.sqrt(variance)
        boll_upper = boll_middle + settings.quant.BOLL_STD_DEV * std_dev
        boll_lower = boll_middle - settings.quant.BOLL_STD_DEV * std_dev

        # ---------------------------------------------------------------------
        # 4. 支撑位与阻力位测算
        # ---------------------------------------------------------------------
        recent_window = min(len(klines), 30)
        recent_lows = lows[-recent_window:]
        recent_highs = highs[-recent_window:]
        support_level = min(recent_lows)
        resistance_level = max(recent_highs)

        # ---------------------------------------------------------------------
        # 5. 多指标共振综合技术走势研判
        # ---------------------------------------------------------------------
        is_macd_golden = latest_dif > latest_dea and latest_macd_bar > 0
        is_macd_death = latest_dif < latest_dea and latest_macd_bar < 0

        if is_macd_golden and rsi_14 > 50.0 and latest_close >= boll_middle:
            if rsi_14 >= 75.0:
                trend_signal = "强势运行-RSI超买预警 (Caution: Overbought Momentum)"
            else:
                trend_signal = "强力买入-多头共振突破 (Strong Buy: Bullish Convergence)"
        elif is_macd_golden and latest_close < boll_middle:
            trend_signal = "多头蓄势-底部反弹萌芽 (Bullish Accumulation)"
        elif is_macd_death and (latest_close < boll_middle or rsi_14 < 45.0):
            if rsi_14 <= 30.0:
                trend_signal = "超跌严重-酝酿技术反抽 (Oversold Technical Rebound)"
            else:
                trend_signal = "空头破位-下行趋势确立 (Bearish Breakdown)"
        else:
            trend_signal = "震荡整理-等待方向选择 (Neutral Consolidation)"

        # ---------------------------------------------------------------------
        # 6. 短线扩展指标：波动率 / 量能 / 短期反转 (日 K 即可计算)
        # ---------------------------------------------------------------------
        volumes = [bar.volume for bar in klines]
        atr = compute_atr(highs, lows, closes, 14)
        rv20 = compute_realized_vol(closes, 20)
        rv5 = compute_realized_vol(closes, 5)
        prev5 = volumes[-6:-1] if len(volumes) >= 6 else volumes[:-1]
        avg_prev5 = _mean(prev5)
        n = len(closes)
        band = boll_upper - boll_lower

        return QuantIndicatorsResult(
            last_close=round(latest_close, 4),
            ma20=round(_mean(closes[-20:]), 4),
            ma60=round(_mean(closes[-60:]), 4),
            atr_14=round(atr, 4),
            atr_pct=round(atr / latest_close * 100.0, 3) if latest_close > 0 else 0.0,
            realized_vol_20=round(rv20, 2),
            realized_vol_5=round(rv5, 2),
            vol_regime=round(rv5 / rv20, 3) if rv20 > 0 else 1.0,
            volume_ratio=round(volumes[-1] / avg_prev5, 3) if avg_prev5 > 0 else 1.0,
            obv_slope=round(compute_obv_slope(closes, volumes, 10), 4),
            mfi_14=round(compute_mfi(highs, lows, closes, volumes, 14), 2),
            ret_5d=round((closes[-1] / closes[-6] - 1.0) * 100.0, 3) if n >= 6 and closes[-6] > 0 else 0.0,
            ret_20d=round((closes[-1] / closes[-21] - 1.0) * 100.0, 3) if n >= 21 and closes[-21] > 0 else 0.0,
            boll_pctb=round((latest_close - boll_lower) / band, 4) if band > 0 else 0.5,
            boll_bandwidth=round(band / boll_middle, 4) if boll_middle > 0 else 0.0,
            macd_dif=round(latest_dif, 3),
            macd_dea=round(latest_dea, 3),
            macd_bar=round(latest_macd_bar, 3),
            rsi_14=round(rsi_14, 2),
            boll_upper=round(boll_upper, 2),
            boll_middle=round(boll_middle, 2),
            boll_lower=round(boll_lower, 2),
            trend_signal=trend_signal,
            support_level=round(support_level, 2),
            resistance_level=round(resistance_level, 2),
        )


# 便捷单例
quant_indicators_engine = QuantIndicatorsEngine()

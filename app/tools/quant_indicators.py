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

        return QuantIndicatorsResult(
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

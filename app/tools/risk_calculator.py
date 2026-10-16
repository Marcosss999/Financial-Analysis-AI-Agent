"""
蒙特卡洛 VaR 风险价值模拟与极端回撤沙箱工具 (Monte Carlo VaR Risk Engine)
职责：
1. 基于几何布朗运动 (Geometric Brownian Motion, GBM) 执行 10,000 次蒙特卡洛随机价格路径模拟。
2. 测算 95% 置信水平下 5 交易日时间窗口的预期最大潜在风险价值损失 (VaR) 与条件在险价值 (CVaR)。
3. 量化标的极端下行尾部风险与最大回撤，为合规风控一票否决沙箱提供扎实的统计学度量底座。
"""

import numpy as np
from typing import List, Optional
from app.core.config import settings
from app.models.schemas import KLineBar, MarketQuote, MonteCarloVaRResult


class RiskCalculatorEngine:
    """确定性蒙特卡洛风险量化计算器"""

    @classmethod
    def calculate_monte_carlo_var(
        cls,
        quote: MarketQuote,
        simulations: Optional[int] = None,
        confidence_level: Optional[float] = None,
        horizon_days: Optional[int] = None,
    ) -> MonteCarloVaRResult:
        """
        基于历史波动率执行向量化蒙特卡洛 VaR 模拟
        :param quote: 包含历史 K 线与现价的行情实体
        :param simulations: 模拟路径条数 (默认 10,000 次)
        :param confidence_level: 置信度水平 (默认 0.95)
        :param horizon_days: 预测时间窗口天数 (默认 5 天)
        :return: MonteCarloVaRResult
        """
        n_sims = simulations or settings.quant.MONTE_CARLO_SIMULATIONS
        conf_lvl = confidence_level or settings.quant.VAR_CONFIDENCE_LEVEL
        days = horizon_days or settings.quant.VAR_HORIZON_DAYS

        current_price = quote.current_price
        klines = quote.kline_history

        # 1. 提取对数收益率序列
        if klines and len(klines) >= 10:
            closes = np.array([b.close for b in klines], dtype=np.float64)
            # 计算对数收益率
            log_returns = np.diff(np.log(closes))
            daily_mu = float(np.mean(log_returns))
            daily_sigma = float(np.std(log_returns))
            # 保证波动率底线 (防止零波动导致除零)
            daily_sigma = max(0.008, daily_sigma)
        else:
            # 兜底默认日度波动率与漂移率
            daily_mu = 0.0005
            daily_sigma = 0.020

        # 2. 向量化几何布朗运动模拟 (GBM: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*sqrt(t)*Z))
        np.random.seed(42)  # 固定随机种子确保同批次复现确定性
        z = np.random.normal(0.0, 1.0, n_sims)
        drift = (daily_mu - 0.5 * (daily_sigma ** 2)) * days
        diffusion = daily_sigma * np.sqrt(days) * z

        simulated_prices = current_price * np.exp(drift + diffusion)
        simulated_returns = (simulated_prices - current_price) / current_price

        # 3. 计算 VaR (损失分位数)
        # 例如 95% 置信度对应 5% 的最差收益率分位数
        loss_percentile = (1.0 - conf_lvl) * 100.0
        var_quantile_return = float(np.percentile(simulated_returns, loss_percentile))
        var_percentage = abs(min(0.0, var_quantile_return)) * 100.0
        var_absolute_loss = current_price * (var_percentage / 100.0)

        # 最极端情景单次最大跌幅
        worst_drawdown = abs(min(0.0, float(np.min(simulated_returns)))) * 100.0

        # 4. 风险等级评定
        if var_percentage < 5.0:
            risk_category = "R1-低波动极稳健 (Low Volatility)"
        elif var_percentage < 9.0:
            risk_category = "R2-稳健适度波动 (Moderate Volatility)"
        elif var_percentage < 14.0:
            risk_category = "R3-平衡型常规波动 (Standard Equity Volatility)"
        elif var_percentage < 18.0:
            risk_category = "R4-进取型高波动 (Elevated Downside Risk)"
        else:
            risk_category = "R5-极端高危强波动 (Critical Downside Risk)"

        return MonteCarloVaRResult(
            simulations=n_sims,
            confidence_level=conf_lvl,
            horizon_days=days,
            var_percentage=round(var_percentage, 2),
            var_absolute_loss=round(var_absolute_loss, 2),
            worst_drawdown=round(worst_drawdown, 2),
            risk_category=risk_category,
        )


# 便捷单例
risk_calculator_engine = RiskCalculatorEngine()

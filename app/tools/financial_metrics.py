"""
确定性财务建模工具模块 (DuPont Analysis & DCF Valuation Engine)
职责：
1. 杜邦三因子拆解计算 (净利率 x 总资产周转率 x 权益乘数 = ROE)，精准定位盈利质量内生驱动力。
2. DCF (Discounted Cash Flow) 自由现金流折现估值数学模型，测算内在公允价值与安全边际。
3. 作为确定性纯算法沙箱，杜绝大模型在复杂财务公式与估值乘数运算中的数值幻觉。
"""

from typing import Optional, List
from app.core.config import settings
from app.models.schemas import FinancialStatements, MarketQuote, DuPontResult, DCFValuationResult


class FinancialMetricsEngine:
    """确定性财务指标与估值运算引擎"""

    @classmethod
    def calculate_dupont(cls, financials: FinancialStatements) -> DuPontResult:
        """
        执行杜邦三因子拆解分析
        ROE = 净利率 (Net Margin) * 总资产周转率 (Asset Turnover) * 权益乘数 (Equity Multiplier)
        """
        revenue = max(0.001, financials.revenue)
        total_assets = max(0.001, financials.total_assets)
        equity = max(0.001, financials.total_assets - financials.total_liabilities)

        # 1. 销售净利率 (%)
        net_profit_margin = (financials.net_profit / revenue) * 100.0

        # 2. 总资产周转率 (次/年)
        asset_turnover = revenue / total_assets

        # 3. 权益乘数 (倍)
        equity_multiplier = total_assets / equity

        # 4. 杜邦公式推导所得 ROE (%)
        calculated_roe = (net_profit_margin / 100.0) * asset_turnover * equity_multiplier * 100.0

        # 5. 驱动因子归因判定
        if net_profit_margin > 25.0 and equity_multiplier < 2.0:
            quality_verdict = "高壁垒型：依靠极高产品附加值与深厚护城河驱动，财务杠杆稳健，盈利成色极高。"
        elif asset_turnover > 1.2:
            quality_verdict = "高周转型：依靠卓越的供应链管理与高运营周转效率驱动，属于薄利快销的运营先锋。"
        elif equity_multiplier > 3.0:
            quality_verdict = "高杠杆警示型：ROE 过度依赖举债与财务杠杆放大，需密切审视债务偿付与利息压力。"
        else:
            quality_verdict = "均衡稳健型：净利率、资产周转率与财务杠杆各维度指标处于行业均衡分布区间。"

        return DuPontResult(
            net_profit_margin=round(net_profit_margin, 2),
            asset_turnover=round(asset_turnover, 3),
            equity_multiplier=round(equity_multiplier, 2),
            calculated_roe=round(calculated_roe, 2),
            quality_verdict=quality_verdict,
        )

    @classmethod
    def calculate_dcf(
        cls,
        financials: FinancialStatements,
        quote: MarketQuote,
        forecast_years: int = 5,
        wacc: Optional[float] = None,
        terminal_growth: Optional[float] = None,
    ) -> DCFValuationResult:
        """
        执行两阶段自由现金流折现 (DCF) 估值模型
        """
        # 读取超参数配置
        discount_rate = wacc if wacc is not None else settings.quant.DEFAULT_WACC
        g = terminal_growth if terminal_growth is not None else settings.quant.DEFAULT_TERMINAL_GROWTH_RATE

        # 基础现金流基准 (取经营现金流扣除维持性资本开支约 20% 作为保守自由现金流 FCF)
        base_fcf = max(1.0, financials.operating_cash_flow * 0.8)

        # 初始预测期增速，取历史净利增速与 15% 的调和折中，并随年份逐年递减衰退
        initial_growth = min(0.35, max(0.04, financials.net_profit_yoy / 100.0))

        projected_fcfs: List[float] = []
        pv_projected_fcfs = 0.0

        current_fcf = base_fcf
        for t in range(1, forecast_years + 1):
            decayed_growth = initial_growth * (0.85 ** (t - 1))
            current_fcf = current_fcf * (1.0 + decayed_growth)
            projected_fcfs.append(round(current_fcf, 2))
            # 折现到当期现值
            discount_factor = (1.0 + discount_rate) ** t
            pv_projected_fcfs += current_fcf / discount_factor

        # 终值计算 (Gordon 永续增长模型): TV = FCF_n * (1 + g) / (WACC - g)
        fcf_terminal_year = projected_fcfs[-1]
        denominator = max(0.01, discount_rate - g)
        terminal_value = (fcf_terminal_year * (1.0 + g)) / denominator
        pv_terminal_value = terminal_value / ((1.0 + discount_rate) ** forecast_years)

        # 企业价值 EV = 预测期现值总和 + 终值折现
        enterprise_value = pv_projected_fcfs + pv_terminal_value

        # 股权价值 Equity Value = 企业价值 - 净负债
        net_debt = max(0.0, financials.total_liabilities * 0.5)
        equity_value = max(1.0, enterprise_value - net_debt)

        # 推导每股内在价值
        total_shares = max(0.1, quote.market_cap / quote.current_price)
        intrinsic_value_per_share = equity_value / total_shares

        # 计算安全边际 Margin of Safety (%)
        margin_of_safety_pct = (
            (intrinsic_value_per_share - quote.current_price) / quote.current_price
        ) * 100.0

        if margin_of_safety_pct > 15.0:
            valuation_status = "显著低估 (Undervalued) - 具备坚实安全边际"
        elif margin_of_safety_pct < -15.0:
            valuation_status = "明显高估 (Overvalued) - 价格透支未来预期"
        else:
            valuation_status = "合理区间 (Fair Value) - 当前市价充分反映基本面"

        return DCFValuationResult(
            wacc=round(discount_rate, 4),
            terminal_growth_rate=round(g, 4),
            projected_fcfs=projected_fcfs,
            terminal_value=round(terminal_value, 2),
            enterprise_value=round(enterprise_value, 2),
            equity_value=round(equity_value, 2),
            intrinsic_value_per_share=round(intrinsic_value_per_share, 2),
            current_price=round(quote.current_price, 2),
            margin_of_safety_pct=round(margin_of_safety_pct, 2),
            valuation_status=valuation_status,
        )


# 便捷单例
financial_metrics_engine = FinancialMetricsEngine()

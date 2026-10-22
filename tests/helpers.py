import asyncio
from contextlib import contextmanager

from app.core.config import settings
from app.models.schemas import AnalysisRequest, FinancialStatements, MarketQuote, KLineBar


def run(coro):
    return asyncio.run(coro)


@contextmanager
def override(obj, **attrs):
    """临时修改 settings 子对象的字段，退出时还原"""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def make_fin(debt=40.0, estimated=None, source="real", ebit=30.0, interest=2.0) -> FinancialStatements:
    return FinancialStatements(
        ticker="600000", period="2025Q2", revenue=100.0, revenue_yoy=5.0, net_profit=20.0, net_profit_yoy=6.0,
        gross_margin=30.0, net_margin=20.0, total_assets=200.0, total_liabilities=200.0 * debt / 100.0,
        debt_to_asset_ratio=debt, operating_cash_flow=22.0, interest_expense=interest, ebit=ebit, roe=15.0,
        data_source=source, estimated_fields=estimated or [],
    )


def make_quote(source="real", price=50.0) -> MarketQuote:
    bars = [KLineBar(date=f"2026-01-{i+1:02d}", open=price, high=price * 1.01, low=price * 0.99,
                     close=price * (1 + 0.002 * ((-1) ** i)), volume=1000.0) for i in range(30)]
    return MarketQuote(ticker="600000", name="测试标的", current_price=price, change_pct=0.5, pe_ttm=15.0,
                       pb_ratio=2.0, market_cap=500.0, kline_history=bars, data_source=source)


def req(**kw) -> AnalysisRequest:
    return AnalysisRequest(**{"ticker": "600519", **kw})

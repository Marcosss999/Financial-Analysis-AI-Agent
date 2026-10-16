"""
模拟盘 (Paper Trading) 数据结构与契约定义模块
职责：
1. 定义虚拟账户资产总览 (PortfolioSummary)
2. 定义单标的持仓明细 (Position)
3. 定义成交流水记录 (TradeOrder)
4. 定义买入、卖出及 Agent 一键建仓请求入参
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Position(BaseModel):
    """单只股票持仓明细"""
    ticker: str = Field(description="标的代码，如 600519")
    name: str = Field(description="证券简称，如 贵州茅台")
    shares: int = Field(description="持仓股数")
    avg_cost: float = Field(description="买入均价(元)")
    current_price: float = Field(description="最新市场现价(元)")
    market_value: float = Field(description="持仓市值(元)")
    unrealized_pnl: float = Field(description="浮动盈亏额(元)")
    unrealized_pnl_pct: float = Field(description="浮动盈亏比例(%)")
    position_pct: float = Field(default=0.0, description="占账户总资产权重(%)")
    strategy_tag: str = Field(default="自主决策", description="建仓时的策略标签，如：动量趋势突破")
    recommendation: str = Field(default="买入", description="建仓时的投资建议评级")
    created_at: str = Field(description="首次建仓时间")
    updated_at: str = Field(description="最新调仓时间")


class TradeOrder(BaseModel):
    """交易流水与成交记录"""
    order_id: str = Field(description="唯一交易订单号")
    timestamp: str = Field(description="交易成交时间")
    ticker: str = Field(description="标的代码")
    name: str = Field(description="证券简称")
    action: str = Field(description="交易动作: BUY(买入) / SELL(卖出)")
    shares: int = Field(description="成交数量(股)")
    price: float = Field(description="成交价格(元)")
    amount: float = Field(description="成交总金额(元)")
    fee: float = Field(default=0.0, description="交易手续费/经手费(元)")
    realized_pnl: float = Field(default=0.0, description="已实现平仓盈亏(元)")
    strategy_tag: str = Field(default="自主决策", description="决策策略标签")
    reason: str = Field(default="正常建仓", description="交易动因说明")


class PortfolioSummary(BaseModel):
    """模拟账户综合资产概况"""
    initial_capital: float = Field(default=1000000.0, description="初始本金(元)，默认100万")
    cash: float = Field(description="可用现金(元)")
    market_value: float = Field(description="总持仓市值(元)")
    total_assets: float = Field(description="总资产(现金+市值)(元)")
    total_pnl: float = Field(description="累计总盈亏额(元)")
    total_pnl_pct: float = Field(description="累计收益率(%)")
    win_rate: float = Field(default=0.0, description="历史平仓交易胜率(%)")
    total_trades: int = Field(default=0, description="累计成交笔数")
    positions_count: int = Field(default=0, description="当前持仓标的数")
    positions: List[Position] = Field(default_factory=list, description="当前活跃持仓列表")
    recent_trades: List[TradeOrder] = Field(default_factory=list, description="最近成交流水(倒序)")


class BuyOrderRequest(BaseModel):
    """手动买入/加仓请求"""
    ticker: str = Field(description="标的代码")
    shares: int = Field(gt=0, description="买入股数")
    price: Optional[float] = Field(default=None, description="指定成交价，若为空则按市场最新价成交")
    strategy_tag: Optional[str] = Field(default="手动决策", description="策略标签")
    reason: Optional[str] = Field(default="用户手动买入", description="决策理由")


class SellOrderRequest(BaseModel):
    """手动卖出/平仓请求"""
    ticker: str = Field(description="标的代码")
    shares: Optional[int] = Field(default=None, description="卖出股数，为 None 或 >= 持仓则全部平仓")
    price: Optional[float] = Field(default=None, description="指定成交价，若为空则按市场最新价成交")
    reason: Optional[str] = Field(default="用户手动平仓", description="卖出理由")


class AgentAutoBuyRequest(BaseModel):
    """依据 Agent 投研结论一键建仓请求"""
    ticker: str = Field(description="标的代码")
    target_position_pct: Optional[float] = Field(default=None, description="目标仓位比例(如 0.25 即 25%)，若为空则从投委会建议中解析")
    strategy_style: Optional[str] = Field(default="balanced", description="当前策略流派")
    verdict: Optional[str] = Field(default="推荐买入", description="投委会仲裁评级")
    reason: Optional[str] = Field(default="依据 Agent 投委会决策一键建仓", description="建仓理由")

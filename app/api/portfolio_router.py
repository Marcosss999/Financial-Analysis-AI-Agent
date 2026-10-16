"""
模拟盘 (Paper Trading) API 接口路由模块
职责：
1. 提供模拟账户总览与持仓/流水查询 (/finance/portfolio)
2. 提供手动买入加仓 (/finance/portfolio/buy) 与平仓卖出 (/finance/portfolio/sell)
3. 深度联动 Agent：提供依据投委会仲裁决议的一键智能建仓端点 (/finance/portfolio/agent-buy)
4. 提供批量持仓最新行情实时刷新 (/finance/portfolio/refresh-prices)
5. 提供模拟账户一键重置归零 (/finance/portfolio/reset)
"""

from fastapi import APIRouter, HTTPException
from app.models.portfolio_schemas import (
    PortfolioSummary,
    BuyOrderRequest,
    SellOrderRequest,
    AgentAutoBuyRequest,
)
from app.tools.portfolio_manager import portfolio_manager

router = APIRouter(prefix="/finance/portfolio", tags=["Paper Trading Portfolio"])


@router.get("", response_model=PortfolioSummary)
async def get_portfolio_summary(refresh: bool = False):
    """获取模拟盘账户资产总览及全部持仓与历史流水"""
    try:
        return portfolio_manager.get_summary(auto_refresh_prices=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模拟盘数据失败: {str(e)}")


@router.post("/buy")
async def execute_manual_buy(req: BuyOrderRequest):
    """手动买入/加仓"""
    try:
        res = portfolio_manager.buy_stock(
            ticker=req.ticker,
            shares=req.shares,
            price=req.price,
            strategy_tag=req.strategy_tag or "自主交易",
            reason=req.reason or "手动买入",
        )
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("message"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"买入委托执行失败: {str(e)}")


@router.post("/sell")
async def execute_manual_sell(req: SellOrderRequest):
    """手动卖出/平仓"""
    try:
        res = portfolio_manager.sell_stock(
            ticker=req.ticker,
            shares=req.shares,
            price=req.price,
            reason=req.reason or "手动平仓",
        )
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("message"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"卖出委托执行失败: {str(e)}")


@router.post("/agent-buy")
async def execute_agent_auto_buy(req: AgentAutoBuyRequest):
    """
    一键按 Agent 投研结论智能建仓：
    自动根据投委会建议仓位比例、当前可用现金和股票实时价格，折算为一手整倍数建仓。
    """
    try:
        res = portfolio_manager.agent_auto_buy(
            ticker=req.ticker,
            target_position_pct=req.target_position_pct,
            strategy_style=req.strategy_style or "balanced",
            verdict=req.verdict or "推荐买入",
            reason=req.reason or "依据 Agent 投委会决策一键建仓",
        )
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("message"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 自动建仓失败: {str(e)}")


@router.post("/refresh-prices", response_model=PortfolioSummary)
async def refresh_portfolio_prices():
    """实时向行情中心抓取最新收盘价/现价，重新计算全部持仓市值与浮盈"""
    try:
        return portfolio_manager.refresh_all_prices()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"行情刷新失败: {str(e)}")


@router.post("/reset", response_model=PortfolioSummary)
async def reset_paper_portfolio():
    """重置模拟账户资产，清空持仓并恢复 100 万元初始现金"""
    try:
        return portfolio_manager.reset_portfolio()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置模拟账户失败: {str(e)}")

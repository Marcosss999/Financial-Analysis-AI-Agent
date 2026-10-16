"""
模拟盘 (Paper Trading) 核心账户资产管理引擎模块
职责：
1. 维护本地持久化模拟账户 (data/paper_portfolio.json)，保障重启不丢失。
2. 支持真实市价委托撮合、仓位加权成本核算与已实现/浮动盈亏计算。
3. 联动 LiveMarketDataProvider，支持持仓标的批量实时行情刷新。
4. 深度桥接 Agent 投研决议：支持一键将投委会建议仓位自动折算为一手(100股)整倍数下单建仓。
"""

import json
import logging
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.models.portfolio_schemas import (
    Position,
    TradeOrder,
    PortfolioSummary,
)
from app.tools.market_data import market_data_provider

logger = logging.getLogger(__name__)

# 本地数据存储路径
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PORTFOLIO_FILE = DATA_DIR / "paper_portfolio.json"

DEFAULT_INITIAL_CAPITAL = 1000000.0  # 初始本金 100 万元人民币
COMMISSION_RATE = 0.00025            # 经手佣金费率万分之2.5 (模拟真实交易成本)


class PortfolioManager:
    """模拟账户资产与委托撮合管理引擎"""

    def __init__(self):
        self._lock = threading.Lock()
        self._ensure_storage()

    def _ensure_storage(self):
        """确保存储目录与初始数据就绪"""
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not PORTFOLIO_FILE.exists():
            initial_data = {
                "initial_capital": DEFAULT_INITIAL_CAPITAL,
                "cash": DEFAULT_INITIAL_CAPITAL,
                "positions": {},
                "trades": [],
                "closed_trades_count": 0,
                "winning_trades_count": 0,
            }
            with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)

    def _load_data(self) -> Dict[str, Any]:
        """读取账户数据"""
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取模拟盘数据失败: {e}，正在重建默认账户")
            self._ensure_storage()
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

    def _save_data(self, data: Dict[str, Any]):
        """持久化保存账户数据"""
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_summary(self, auto_refresh_prices: bool = False) -> PortfolioSummary:
        """获取当前账户资产总览，可选择是否顺带刷新最新行情"""
        with self._lock:
            data = self._load_data()

        if auto_refresh_prices and data.get("positions"):
            self.refresh_all_prices()
            with self._lock:
                data = self._load_data()

        cash = round(data.get("cash", DEFAULT_INITIAL_CAPITAL), 2)
        initial_cap = round(data.get("initial_capital", DEFAULT_INITIAL_CAPITAL), 2)
        raw_pos = data.get("positions", {})
        raw_trades = data.get("trades", [])

        total_market_val = 0.0
        position_list: List[Position] = []

        for ticker, pos_dict in raw_pos.items():
            shares = pos_dict["shares"]
            curr_p = pos_dict["current_price"]
            avg_c = pos_dict["avg_cost"]
            mv = round(shares * curr_p, 2)
            total_market_val += mv

            unreal_pnl = round((curr_p - avg_c) * shares, 2)
            unreal_pct = round(((curr_p - avg_c) / avg_c * 100.0) if avg_c > 0 else 0.0, 2)

            pos_obj = Position(
                ticker=ticker,
                name=pos_dict.get("name", ticker),
                shares=shares,
                avg_cost=avg_c,
                current_price=curr_p,
                market_value=mv,
                unrealized_pnl=unreal_pnl,
                unrealized_pnl_pct=unreal_pct,
                position_pct=0.0,  # 稍后总资产算出后更新
                strategy_tag=pos_dict.get("strategy_tag", "自主交易"),
                recommendation=pos_dict.get("recommendation", "买入"),
                created_at=pos_dict.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M")),
                updated_at=pos_dict.get("updated_at", datetime.now().strftime("%Y-%m-%d %H:%M")),
            )
            position_list.append(pos_obj)

        total_market_val = round(total_market_val, 2)
        total_assets = round(cash + total_market_val, 2)
        total_pnl = round(total_assets - initial_cap, 2)
        total_pnl_pct = round((total_pnl / initial_cap) * 100.0, 2)

        # 补算各标的在总资产中的权重
        if total_assets > 0:
            for p in position_list:
                p.position_pct = round((p.market_value / total_assets) * 100.0, 1)

        closed_cnt = data.get("closed_trades_count", 0)
        win_cnt = data.get("winning_trades_count", 0)
        win_rate = round((win_cnt / closed_cnt * 100.0) if closed_cnt > 0 else 0.0, 1)

        # 格式化流水记录 (按时间倒序排列)
        trade_orders = [TradeOrder(**t) for t in reversed(raw_trades)]

        return PortfolioSummary(
            initial_capital=initial_cap,
            cash=cash,
            market_value=total_market_val,
            total_assets=total_assets,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            total_trades=len(raw_trades),
            positions_count=len(position_list),
            positions=position_list,
            recent_trades=trade_orders[:50],  # 最多返回最近50笔
        )

    def buy_stock(
        self,
        ticker: str,
        shares: int,
        price: Optional[float] = None,
        strategy_tag: str = "自主交易",
        recommendation: str = "买入",
        reason: str = "用户买入建仓",
    ) -> Dict[str, Any]:
        """买入股票撮合"""
        ticker = ticker.strip().upper()
        if shares <= 0:
            return {"success": False, "message": "买入股数必须大于 0"}

        # 确定买入价格与证券简称
        quote = market_data_provider.get_market_quote(ticker, days=5)
        exec_price = price if (price is not None and price > 0) else quote.current_price
        target_name = quote.name or ticker

        order_amount = round(exec_price * shares, 2)
        fee = round(order_amount * COMMISSION_RATE, 2)
        total_deduction = round(order_amount + fee, 2)

        with self._lock:
            data = self._load_data()
            curr_cash = data.get("cash", 0.0)

            if curr_cash < total_deduction:
                return {
                    "success": False,
                    "message": f"可用资金不足！需要 ¥{total_deduction:,.2f} (含佣金 ¥{fee:,.2f})，当前可用仅 ¥{curr_cash:,.2f}",
                }

            # 扣除现金
            data["cash"] = round(curr_cash - total_deduction, 2)
            positions = data.setdefault("positions", {})

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if ticker in positions:
                # 已有持仓：加权平均计算成本
                old_shares = positions[ticker]["shares"]
                old_cost = positions[ticker]["avg_cost"]
                new_total_shares = old_shares + shares
                new_avg_cost = round(((old_shares * old_cost) + (shares * exec_price)) / new_total_shares, 2)

                positions[ticker]["shares"] = new_total_shares
                positions[ticker]["avg_cost"] = new_avg_cost
                positions[ticker]["current_price"] = exec_price
                positions[ticker]["updated_at"] = now_str
                positions[ticker]["strategy_tag"] = strategy_tag
                positions[ticker]["recommendation"] = recommendation
            else:
                # 首次建仓
                positions[ticker] = {
                    "name": target_name,
                    "shares": shares,
                    "avg_cost": exec_price,
                    "current_price": exec_price,
                    "strategy_tag": strategy_tag,
                    "recommendation": recommendation,
                    "created_at": now_str,
                    "updated_at": now_str,
                }

            order_id = f"ORD_{int(time.time()*1000)}"
            trade_record = {
                "order_id": order_id,
                "timestamp": now_str,
                "ticker": ticker,
                "name": target_name,
                "action": "BUY",
                "shares": shares,
                "price": exec_price,
                "amount": order_amount,
                "fee": fee,
                "realized_pnl": 0.0,
                "strategy_tag": strategy_tag,
                "reason": reason,
            }
            data.setdefault("trades", []).append(trade_record)
            self._save_data(data)

        logger.info(f"模拟买入成功: {ticker} {target_name} {shares}股 @ {exec_price}元, 消耗现金: {total_deduction}元")
        return {
            "success": True,
            "message": f"成功买入 {target_name} ({ticker}) {shares} 股，成交均价: ¥{exec_price:.2f} 元",
            "order": trade_record,
        }

    def sell_stock(
        self,
        ticker: str,
        shares: Optional[int] = None,
        price: Optional[float] = None,
        reason: str = "用户平仓卖出",
    ) -> Dict[str, Any]:
        """卖出股票撮合"""
        ticker = ticker.strip().upper()

        with self._lock:
            data = self._load_data()
            positions = data.get("positions", {})
            if ticker not in positions:
                return {"success": False, "message": f"账户中未持有标的 {ticker}"}

            pos = positions[ticker]
            curr_holding = pos["shares"]
            avg_cost = pos["avg_cost"]
            target_name = pos.get("name", ticker)

            # 若 shares 为 None 或大于等于持仓量，则全部平仓
            sell_shares = curr_holding if (shares is None or shares <= 0 or shares >= curr_holding) else shares

            quote = market_data_provider.get_market_quote(ticker, days=5)
            exec_price = price if (price is not None and price > 0) else quote.current_price

            gross_amount = round(exec_price * sell_shares, 2)
            fee = round(gross_amount * COMMISSION_RATE, 2)
            net_inflow = round(gross_amount - fee, 2)

            # 计算已实现盈亏 (不含已扣买入费，扣除本次卖出佣金)
            realized_pnl = round((exec_price - avg_cost) * sell_shares - fee, 2)

            # 现金回流
            data["cash"] = round(data.get("cash", 0.0) + net_inflow, 2)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if sell_shares >= curr_holding:
                # 全部清仓
                del positions[ticker]
            else:
                # 部分减仓
                pos["shares"] = curr_holding - sell_shares
                pos["current_price"] = exec_price
                pos["updated_at"] = now_str

            # 统计胜率
            data["closed_trades_count"] = data.get("closed_trades_count", 0) + 1
            if realized_pnl > 0:
                data["winning_trades_count"] = data.get("winning_trades_count", 0) + 1

            order_id = f"ORD_{int(time.time()*1000)}"
            trade_record = {
                "order_id": order_id,
                "timestamp": now_str,
                "ticker": ticker,
                "name": target_name,
                "action": "SELL",
                "shares": sell_shares,
                "price": exec_price,
                "amount": gross_amount,
                "fee": fee,
                "realized_pnl": realized_pnl,
                "strategy_tag": pos.get("strategy_tag", "平仓"),
                "reason": reason,
            }
            data.setdefault("trades", []).append(trade_record)
            self._save_data(data)

        logger.info(f"模拟卖出成功: {ticker} {target_name} {sell_shares}股 @ {exec_price}元, 实现盈亏: {realized_pnl}元")
        return {
            "success": True,
            "message": f"成功卖出 {target_name} ({ticker}) {sell_shares} 股，成交均价: ¥{exec_price:.2f} 元，实现盈亏: ¥{realized_pnl:+,.2f} 元",
            "order": trade_record,
        }

    def agent_auto_buy(
        self,
        ticker: str,
        target_position_pct: Optional[float] = None,
        strategy_style: str = "balanced",
        verdict: str = "强烈推荐",
        reason: str = "依据投委会仲裁决议自动建仓",
    ) -> Dict[str, Any]:
        """
        依据 Agent 投研结论一键计算并智能建仓：
        1. 获取标的实时现价与当前账户总资产。
        2. 根据投委会建议的目标仓位比例 (如 25%~35% 中位数 30%)，计算目标拟配置金额。
        3. 对 A 股自动按“一手 = 100 股”整倍数向下取整，至少买入 100 股。
        """
        ticker = ticker.strip().upper()
        summary = self.get_summary(auto_refresh_prices=False)

        # 默认仓位推导：如果未给具体数字，根据评级设置合理仓位
        pos_ratio = target_position_pct
        if pos_ratio is None or pos_ratio <= 0:
            if "强烈推荐" in verdict or "强推" in verdict:
                pos_ratio = 0.30
            elif "逢低吸纳" in verdict or "积极" in verdict:
                pos_ratio = 0.20
            elif "谨慎观望" in verdict:
                pos_ratio = 0.10
            else:
                pos_ratio = 0.15

        pos_ratio = min(0.50, max(0.05, pos_ratio))  # 单票最高50%风控约束

        quote = market_data_provider.get_market_quote(ticker, days=5)
        price = quote.current_price
        if price <= 0:
            return {"success": False, "message": f"标的 {ticker} 行情价格异常 (¥{price})"}

        # 目标采购金额
        target_budget = round(summary.total_assets * pos_ratio, 2)
        # 不能超过可用现金的 98% (预留印花税与佣金)
        available_budget = min(target_budget, summary.cash * 0.98)

        if available_budget < price * 100 and ticker.isdigit():
            # A 股买不起一手
            return {
                "success": False,
                "message": f"可用现金不足以买入一手(100股) {quote.name} (单价 ¥{price:.2f}，买一手需约 ¥{price*100:,.2f}，可用现金 ¥{summary.cash:,.2f})",
            }

        # 计算股数 (A股为纯数字代码，按100股一手取整)
        if ticker.isdigit():
            shares = int(available_budget // (price * 100)) * 100
        else:
            shares = int(available_budget // price)

        if shares <= 0:
            return {"success": False, "message": "测算可买入股数为 0，资金不足或单价过高"}

        style_names = {
            "momentum": "动量趋势突破",
            "balanced": "机构均衡配置",
            "deep_value": "深度逆向价值",
            "short_term": "短线游资量价",
            "custom": "自定义指标矩阵",
        }
        style_desc = style_names.get(strategy_style, strategy_style)
        full_reason = f"【Agent投委会裁决】{verdict} | 建议配置约 {int(pos_ratio*100)}% 仓位 | 策略流派: {style_desc}"

        return self.buy_stock(
            ticker=ticker,
            shares=shares,
            price=price,
            strategy_tag=f"{style_desc} · {verdict}",
            recommendation=verdict,
            reason=full_reason,
        )

    def refresh_all_prices(self) -> PortfolioSummary:
        """从行情源批量拉取最新价格，刷新全部持仓市值与浮动盈亏"""
        with self._lock:
            data = self._load_data()
            positions = data.get("positions", {})
            tickers = list(positions.keys())

        if not tickers:
            return self.get_summary(auto_refresh_prices=False)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for t in tickers:
            try:
                q = market_data_provider.get_market_quote(t, days=5)
                if q and q.current_price > 0:
                    with self._lock:
                        if t in data["positions"]:
                            data["positions"][t]["current_price"] = q.current_price
                            data["positions"][t]["name"] = q.name or data["positions"][t].get("name", t)
                            data["positions"][t]["updated_at"] = now_str
            except Exception as e:
                logger.warning(f"刷新标的 {t} 实时现价失败: {e}")

        with self._lock:
            self._save_data(data)

        return self.get_summary(auto_refresh_prices=False)

    def reset_portfolio(self, initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> PortfolioSummary:
        """清空持仓并重置模拟账户"""
        with self._lock:
            reset_data = {
                "initial_capital": initial_capital,
                "cash": initial_capital,
                "positions": {},
                "trades": [],
                "closed_trades_count": 0,
                "winning_trades_count": 0,
            }
            self._save_data(reset_data)
        logger.info(f"模拟账户已完全重置，初始本金: ¥{initial_capital:,.2f} 元")
        return self.get_summary(auto_refresh_prices=False)


# 全局模拟账户管理器单例
portfolio_manager = PortfolioManager()

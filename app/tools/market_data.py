"""
金融行情与财报数据实时抽取工具模块 (Live Market Data & Financial Statements Provider)
职责：
1. 真实行情直连：对接腾讯财经 (Tencent Finance) 毫秒级行情接口，实时获取 A 股、港股、美股当日真实收盘价/现价、涨跌幅、市盈率 PE(TTM)、市净率 PB、总市值与换手率。
2. 真实时序 K 线：直连历史复权日 K 线接口，获取最近 60 个交易日的真实开高低收成交量 (OHLCV) 数据。
3. 真实财报三表：直连东方财富 (EastMoney) 最新财报核心财务指标库，抓取真实报告期营收、归母净利润、毛利率、资产负债率与加权 ROE。
4. 弹性容灾兜底：当网络中断或输入未知非标代码时，自动平滑回退至高保真合成引擎，保证系统永不阻断崩溃。
"""

import hashlib
import logging
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import requests

from app.core.config import settings
from app.models.schemas import MarketQuote, KLineBar, FinancialStatements

logger = logging.getLogger(__name__)


class DataUnavailableError(RuntimeError):
    """STRICT_REAL_DATA 模式下，真实数据源不可用时抛出，而不是静默降级为模拟数据"""


def _safe_float(val: Any, default: float = 0.0) -> float:
    """安全浮点数转换，过滤非数字字符串"""
    if val is None:
        return default
    try:
        cleaned = str(val).strip().replace(",", "")
        return float(cleaned)
    except Exception:
        return default


def _parse_ticker_symbol(ticker: str) -> Tuple[str, str, str]:
    """
    智能解析标的代码及市场前缀
    :return: (prefix, pure_code, full_symbol)
    例如: 600519 -> ('sh', '600519', 'sh600519')
          002594 -> ('sz', '002594', 'sz002594')
          300750 -> ('sz', '300750', 'sz300750')
          AAPL   -> ('us', 'AAPL', 'usAAPL')
    """
    clean = ticker.strip().upper()
    if clean.startswith(("SH", "SZ", "US", "HK")):
        prefix = clean[:2].lower()
        code = clean[2:]
        return prefix, code, f"{prefix}{code}"

    if clean.isdigit():
        if clean.startswith("6"):
            return "sh", clean, f"sh{clean}"
        elif clean.startswith(("0", "3")):
            return "sz", clean, f"sz{clean}"
        elif clean.startswith(("4", "8")):
            return "bj", clean, f"bj{clean}"

    # 默认美股标的
    return "us", clean, f"us{clean}"


class LiveMarketDataProvider:
    """真实金融数据与容灾降级服务提供器"""

    @classmethod
    def get_market_quote(cls, ticker: str, days: int = 60) -> MarketQuote:
        """
        获取标的真实即时/收盘行情与真实历史日 K 线
        """
        prefix, code, full_symbol = _parse_ticker_symbol(ticker)

        # 1. 尝试从腾讯实时行情网关抽取真实行情
        try:
            quote_url = f"http://qt.gtimg.cn/q={full_symbol}"
            resp = requests.get(quote_url, timeout=3.5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                raw_text = resp.content.decode("gbk", errors="ignore").strip()
                if "~" in raw_text:
                    parts = raw_text.split("~")
                    if len(parts) >= 35:
                        real_name = parts[1].strip() or f"标的-{code}"
                        real_price = _safe_float(parts[3])
                        change_pct = _safe_float(parts[32] if len(parts) > 32 else 0.0)
                        turnover_rate = _safe_float(parts[38] if len(parts) > 38 else 1.2, 1.2)
                        pe_ttm = _safe_float(parts[39] if len(parts) > 39 else 20.0, 20.0)

                        if prefix == "us":
                            # 美股字段解析
                            market_cap = _safe_float(parts[45] if len(parts) > 45 else real_price * 15.0, real_price * 15.0)
                            pb_ratio = _safe_float(parts[54] if len(parts) > 54 else 15.0, 15.0)
                        else:
                            # A股 / 港股字段解析
                            market_cap = _safe_float(parts[45] if len(parts) > 45 else real_price * 10.0, real_price * 10.0)
                            pb_ratio = _safe_float(parts[46] if len(parts) > 46 else 3.0, 3.0)

                        # 2. 尝试抽取真实历史日 K 线
                        kline_history = cls._fetch_real_klines(full_symbol, days=days)
                        source = "real"
                        if not kline_history:
                            if settings.eng.STRICT_REAL_DATA:
                                raise DataUnavailableError(f"{ticker} K 线不可用 (STRICT_REAL_DATA=true)")
                            kline_history = cls._generate_synthetic_klines(real_price, days=days, seed=code)
                            source = "partial_simulated_klines"

                        return MarketQuote(
                            ticker=code,
                            name=real_name,
                            current_price=round(real_price, 2),
                            change_pct=round(change_pct, 2),
                            pe_ttm=round(pe_ttm, 1),
                            pb_ratio=round(pb_ratio, 1),
                            market_cap=round(market_cap, 1),
                            turnover_rate=round(turnover_rate, 2),
                            kline_history=kline_history,
                            data_source=source,
                        )
        except DataUnavailableError:
            raise
        except Exception as e:
            logger.warning(f"[MarketData] 真实行情抽取失败 ({e})，启用弹性保真模拟器。")

        # 3. 失败兜底：使用确定性合成引擎 (严格模式下直接报错)
        if settings.eng.STRICT_REAL_DATA:
            raise DataUnavailableError(f"{ticker} 行情数据源不可用 (STRICT_REAL_DATA=true，拒绝使用模拟数据)")
        return cls._get_synthetic_market_quote(code, days=days)

    @classmethod
    def get_index_bars(cls, days: int = 60, symbol: str = "sh000300") -> Tuple[List[KLineBar], str]:
        """宽基指数(默认沪深300)日 K，用于“宏观/市场环境”评分。返回 (bars, source)；source: real / simulated / unavailable"""
        bars = cls._fetch_real_klines(symbol, days=days)
        if bars:
            return bars, "real"
        if settings.eng.STRICT_REAL_DATA:
            return [], "unavailable"
        return cls._generate_synthetic_klines(4000.0, days=days, seed=f"INDEX-{symbol}"), "simulated"

    @classmethod
    def _fetch_real_klines(cls, full_symbol: str, days: int = 60) -> List[KLineBar]:
        """抓取真实历史日 K 线序列"""
        try:
            kline_url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={full_symbol},day,,,{days},qfq"
            resp = requests.get(kline_url, timeout=3.5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                data = resp.json()
                stock_data = data.get("data", {}).get(full_symbol, {})
                raw_bars = stock_data.get("qfqday", stock_data.get("day", []))
                bars: List[KLineBar] = []
                for b in raw_bars:
                    if len(b) >= 6:
                        # 格式: [日期, 开盘, 收盘, 最高, 最低, 成交量]
                        date_str = str(b[0])
                        open_p = _safe_float(b[1])
                        close_p = _safe_float(b[2])
                        high_p = _safe_float(b[3])
                        low_p = _safe_float(b[4])
                        vol = _safe_float(b[5])
                        bars.append(KLineBar(
                            date=date_str,
                            open=round(open_p, 2),
                            high=round(high_p, 2),
                            low=round(low_p, 2),
                            close=round(close_p, 2),
                            volume=round(vol, 0),
                            turnover=round(vol * close_p, 2),
                        ))
                if bars:
                    bars.sort(key=lambda x: x.date)
                    return bars[-days:]
        except Exception as e:
            logger.debug(f"[MarketData] 真实 K 线解析失败: {e}")
        return []

    @classmethod
    def get_financial_statements(cls, ticker: str) -> FinancialStatements:
        """
        获取标的真实财报核心三表与指标数据
        """
        prefix, code, _ = _parse_ticker_symbol(ticker)

        # 尝试从东方财富获取真实 A 股最新季报/中报/年报指标
        if prefix in ("sh", "sz", "bj"):
            try:
                em_code = f"{prefix.upper()}{code}"
                fin_url = f"http://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/ZYZBAjaxNew?type=0&code={em_code}"
                resp = requests.get(fin_url, timeout=3.5, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    items = resp.json().get("data", [])
                    if items:
                        latest = items[0]
                        period_name = latest.get("REPORT_DATE_NAME", "最新报告期")
                        roe_val = _safe_float(latest.get("ROEJQ", 0))
                        # 金额单位转换为亿元
                        rev_val = round(_safe_float(latest.get("TOTALOPERATEREVE", 0)) / 1e8, 1)
                        net_val = round(_safe_float(latest.get("PARENTNETPROFIT", 0)) / 1e8, 1)
                        debt_ratio = _safe_float(latest.get("ZCFZL", 0))
                        gross_margin = _safe_float(latest.get("XSMLL", 0))

                        # 同比增速计算
                        rev_yoy = 8.5
                        net_yoy = 10.2
                        if len(items) >= 2:
                            prev = items[1]
                            prev_rev = _safe_float(prev.get("TOTALOPERATEREVE", 0))
                            prev_net = _safe_float(prev.get("PARENTNETPROFIT", 0))
                            if prev_rev > 0:
                                rev_yoy = round(((_safe_float(latest.get("TOTALOPERATEREVE", 0)) - prev_rev) / prev_rev) * 100.0, 1)
                            if prev_net > 0:
                                net_yoy = round(((_safe_float(latest.get("PARENTNETPROFIT", 0)) - prev_net) / prev_net) * 100.0, 1)

                        total_assets = round(rev_val * 2.2, 1)
                        total_liab = round(total_assets * (debt_ratio / 100.0), 1)

                        return FinancialStatements(
                            ticker=code,
                            period=period_name,
                            revenue=max(1.0, rev_val),
                            revenue_yoy=rev_yoy,
                            net_profit=max(0.1, net_val),
                            net_profit_yoy=net_yoy,
                            gross_margin=round(gross_margin, 1) if gross_margin else 25.0,
                            net_margin=round(net_val / max(0.1, rev_val) * 100.0, 1),
                            total_assets=max(1.0, total_assets),
                            total_liabilities=max(0.1, total_liab),
                            debt_to_asset_ratio=round(debt_ratio, 1),
                            operating_cash_flow=round(net_val * 1.15, 1),
                            interest_expense=round(max(0.5, total_liab * 0.035), 1),
                            ebit=round(net_val * 1.35, 1),
                            roe=round(roe_val, 1) if roe_val else 15.0,
                            data_source="real",
                            # 以下字段接口未直接提供，按经验系数由营收/净利润推算，风控不得据此一票否决
                            estimated_fields=["total_assets", "total_liabilities", "operating_cash_flow",
                                              "interest_expense", "ebit"],
                        )
            except Exception as e:
                logger.warning(f"[MarketData] 真实财报抽取失败 ({e})，启用保真模拟数据。")

        # 美股或兜底回退生成
        if settings.eng.STRICT_REAL_DATA:
            raise DataUnavailableError(f"{ticker} 财报数据源不可用 (STRICT_REAL_DATA=true，拒绝使用模拟数据)")
        return cls._get_synthetic_financials(code)

    # =========================================================================
    # 弹性兜底：合成数据生成器 (确保离线环境下稳定可测)
    # =========================================================================

    @classmethod
    def _generate_synthetic_klines(cls, current_price: float, days: int = 60, seed: Any = 0) -> List[KLineBar]:
        """
        离线模拟 K 线：以标的代码为种子的随机游走(几何布朗运动式)，从“今天的现价”倒推历史。
        每个标的有不同的波动率，保证离线演示时各标的的技术指标和 VaR 有区分度；仅供演示。
        """
        seed_int = int(hashlib.md5(str(seed).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        daily_sigma = 0.010 + (seed_int % 25) / 1000.0  # 1.0% ~ 3.4% 日波动
        rets = rng.normal(0.0003, daily_sigma, days)
        bars: List[KLineBar] = []
        today = datetime.now()
        close = current_price
        for i in range(days, 0, -1):
            r = float(rets[days - i])
            open_p = close / (1.0 + r)
            span = abs(r) + daily_sigma * 0.5
            high_p = max(open_p, close) * (1.0 + span * 0.5)
            low_p = min(open_p, close) * (1.0 - span * 0.5)
            vol = 15000.0 * (1.0 + abs(r) * 20.0)
            bars.append(KLineBar(
                date=(today - timedelta(days=int(i * 1.45))).strftime("%Y-%m-%d"),
                open=round(open_p, 2),
                high=round(high_p, 2),
                low=round(low_p, 2),
                close=round(close, 2),
                volume=round(vol, 0),
                turnover=round(vol * close, 2),
            ))
            close = open_p
        bars.sort(key=lambda x: x.date)
        return bars

    @classmethod
    def _get_synthetic_market_quote(cls, code: str, days: int = 60) -> MarketQuote:
        h_val = int(hashlib.md5(code.encode("utf-8")).hexdigest()[:8], 16)
        base_price = 20.0 + (h_val % 1800) / 10.0
        return MarketQuote(
            ticker=code,
            name=f"标的资产-{code}",
            current_price=round(base_price, 2),
            change_pct=round(((h_val % 1000) - 480) / 100.0, 2),
            pe_ttm=round(12.0 + (h_val % 450) / 10.0, 1),
            pb_ratio=round(1.2 + (h_val % 80) / 10.0, 1),
            market_cap=round(base_price * 15.0, 1),
            turnover_rate=round(0.5 + (h_val % 40) / 10.0, 2),
            kline_history=cls._generate_synthetic_klines(base_price, days=days, seed=code),
            data_source="simulated",
        )

    @classmethod
    def _get_synthetic_financials(cls, code: str) -> FinancialStatements:
        h_val = int(hashlib.md5(code.encode("utf-8")).hexdigest()[:8], 16)
        revenue = 50.0 + (h_val % 800)
        net_profit = max(2.0, revenue * (0.05 + (h_val % 25) / 100.0))
        debt_ratio = 25.0 + (h_val % 520) / 10.0
        assets = revenue * (1.2 + (h_val % 10) / 10.0)
        liabilities = assets * (debt_ratio / 100.0)
        return FinancialStatements(
            ticker=code,
            period="2024Q3 (模拟报告期)",
            revenue=round(revenue, 1),
            revenue_yoy=round(5.0 + (h_val % 250) / 10.0, 1),
            net_profit=round(net_profit, 1),
            net_profit_yoy=round(4.0 + (h_val % 300) / 10.0, 1),
            gross_margin=round(18.0 + (h_val % 400) / 10.0, 1),
            net_margin=round(net_profit / revenue * 100.0, 1),
            total_assets=round(assets, 1),
            total_liabilities=round(liabilities, 1),
            debt_to_asset_ratio=round(debt_ratio, 1),
            operating_cash_flow=round(net_profit * 1.1, 1),
            interest_expense=round(max(0.5, liabilities * 0.035), 1),
            ebit=round(net_profit * 1.3, 1),
            roe=round((net_profit / max(1.0, assets - liabilities)) * 100.0, 1),
            data_source="simulated",
            estimated_fields=["total_assets", "total_liabilities", "operating_cash_flow", "interest_expense", "ebit"],
        )


# 全局单例
market_data_provider = LiveMarketDataProvider()

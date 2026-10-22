"""
多源金融市场情报与舆情抽取工具模块 (Multi-Source Market Intel Provider)
职责：
1. 真实资金面抓取：直连东方财富分时主力资金流向接口，实时解析主力大单与超大单净买入额。
2. 真实信息面抓取：直连东方财富官方上市公司正式披露公告库，提取近期重大公告（减持、质押、定增、财报等）。
3. 真实媒体舆情抓取：直连新浪财经个股滚动资讯接口，获取市场要闻与盘口动态。
4. 弹性容灾兜底：支持离线及非A股标的高保真合成，保障 Agent 链路 100% 健壮可用。
"""

import hashlib
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple
import requests

from app.models.schemas import MarketIntel

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _normalize_ticker(ticker: str) -> Tuple[str, str, str, str]:
    """
    解析股票代码为腾讯和东财专属标识
    :return: (market, pure_code, tencent_symbol, eastmoney_secid)
    """
    clean = ticker.strip().lower()
    digits = re.sub(r"[^\d]", "", clean)
    if not digits:
        # 非数字代码 (如美股 AAPL)
        return "us", clean.upper(), f"us{clean.upper()}", f"105.{clean.upper()}"

    pure_code = digits.zfill(6)
    if clean.startswith("sh") or pure_code.startswith(("60", "68", "90")):
        market = "sh"
        em_secid = f"1.{pure_code}"
    elif clean.startswith("sz") or pure_code.startswith(("00", "30", "20")):
        market = "sz"
        em_secid = f"0.{pure_code}"
    elif clean.startswith("bj") or pure_code.startswith(("8", "4", "92")):
        market = "bj"
        em_secid = f"0.{pure_code}"
    else:
        market = "sh" if pure_code.startswith("6") else "sz"
        em_secid = f"1.{pure_code}" if market == "sh" else f"0.{pure_code}"

    tencent_symbol = f"{market}{pure_code}"
    return market, pure_code, tencent_symbol, em_secid


class MarketIntelProvider:
    """多源情报事实抽取引擎"""

    @classmethod
    def get_market_intel(cls, ticker: str, target_name: str = "") -> MarketIntel:
        """抽取标的多源资金流向、官方公告与新闻要闻"""
        market, pure_code, tc_symbol, em_secid = _normalize_ticker(ticker)

        # 1. 抓取主力资金流向
        flow_info = cls._fetch_capital_flow(em_secid)

        # 2. 抓取官方公告
        announcements = cls._fetch_announcements(pure_code)

        # 3. 抓取新浪资讯
        news_items = cls._fetch_sina_news(tc_symbol)

        # 4. 组装：数据源不可用时如实说明，不编造"正常/平稳"之类的内容
        if not announcements:
            announcements = ["（公告数据源暂不可用，本次分析未纳入公告信息）"]
        if not news_items:
            news_items = ["（资讯数据源暂不可用，本次分析未纳入新闻舆情）"]

        return MarketIntel(
            main_capital_flow=flow_info["display"],
            capital_inflow_amount_str=flow_info["amount_str"],
            is_main_inflow=flow_info["is_inflow"],
            announcements=announcements[:5],
            news_headlines=news_items[:6],
            flow_net_yuan=flow_info.get("net_yuan"),
            data_source=flow_info.get("source", "real"),
        )

    @classmethod
    def _fetch_capital_flow(cls, em_secid: str, timeout: float = 3.5) -> Dict[str, Any]:
        """从东方财富抽取分时主力资金净额 (大单+超大单)"""
        url = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
        params = {
            "lmt": 0,
            "klt": 1,
            "secid": em_secid,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52",
        }
        headers = dict(DEFAULT_HEADERS, Referer="https://quote.eastmoney.com")
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                klines = ((r.json() or {}).get("data") or {}).get("klines") or []
                if klines:
                    parts = str(klines[-1]).split(",")
                    if len(parts) >= 2 and parts[1] not in (None, "", "-"):
                        net_val = float(parts[1])
                        abs_val = abs(net_val)
                        if abs_val >= 1e8:
                            amt_str = f"{net_val / 1e8:+.2f} 亿元"
                        else:
                            amt_str = f"{net_val / 1e4:+.2f} 万元"

                        is_inflow = net_val >= 0
                        direction_desc = "主力资金积极净流入抢筹" if is_inflow else "主力大单小幅净流出休整/洗盘"
                        return {
                            "amount_str": amt_str,
                            "is_inflow": is_inflow,
                            "display": f"{direction_desc} ({amt_str})",
                            "net_yuan": net_val,
                            "source": "real",
                        }
        except Exception as e:
            logger.debug(f"[MarketIntel] 资金流获取降级: {e}")

        # 数据源不可用：不再编造净流入金额，资金面维度将从加权中剔除
        return {
            "amount_str": "数据不可用",
            "is_inflow": False,
            "display": "主力资金数据暂不可用（已从决策加权中剔除）",
            "net_yuan": None,
            "source": "unavailable",
        }

    @classmethod
    def _fetch_announcements(cls, pure_code: str, count: int = 5, timeout: float = 3.5) -> List[str]:
        """从东方财富官方上市公司披露库获取最新公告"""
        if not pure_code.isdigit():
            return []

        url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
        params = {
            "page_size": str(count),
            "page_index": "1",
            "ann_type": "A",
            "client_source": "web",
            "stock_list": pure_code,
        }
        headers = dict(DEFAULT_HEADERS, Referer="https://data.eastmoney.com")
        ann_list = []
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                items = (r.json() or {}).get("data", {}).get("list", [])
                for it in items[:count]:
                    date_str = str(it.get("notice_date", ""))[:10]
                    title = str(it.get("title", "")).strip()
                    if title:
                        ann_list.append(f"[{date_str}] {title}")
        except Exception as e:
            logger.debug(f"[MarketIntel] 公告抽取降级: {e}")
        return ann_list

    @classmethod
    def _fetch_sina_news(cls, tc_symbol: str, count: int = 6, timeout: float = 3.5) -> List[str]:
        """从新浪财经获取个股滚动资讯快讯"""
        url = f"http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol={tc_symbol}"
        news_list = []
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            if r.status_code == 200:
                r.encoding = "gbk"
                matches = re.findall(r"<a\s+target='_blank'\s+href='[^']*'>([^<]+)</a>", r.text)
                for title in matches:
                    t = title.strip()
                    if t and len(t) >= 6 and not any(k in t for k in ["新浪财经", "行情中心", "股吧", "客户端"]):
                        news_list.append(t)
                        if len(news_list) >= count:
                            break
        except Exception as e:
            logger.debug(f"[MarketIntel] 新浪要闻抽取降级: {e}")
        return news_list


market_intel_provider = MarketIntelProvider()

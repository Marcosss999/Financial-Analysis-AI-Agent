"""
历史日 K 数据加载（实验用）

- fetch_history()：从腾讯 fqkline 接口分段拉取前复权日 K，并缓存成 CSV(experiments/data_cache/)。需要联网，仅应在你自己的机器上运行。
- synthetic_universe()：带明确标签的合成数据，只用于验证实验代码本身有没有写错，绝不能当作真实结果。
"""
import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.models.schemas import KLineBar

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"

# 默认实验股票池：沪深300 里行业分散的一批大盘股(示例，可自行替换为更大样本)
DEFAULT_UNIVERSE = [
    "600519", "000858", "601318", "600036", "000333", "600276", "601899", "600900", "002415", "300750",
    "601166", "600030", "000651", "600887", "601888", "002594", "600309", "601012", "000725", "600028",
    "601398", "600104", "002304", "600585", "601668", "000568", "600809", "603288", "002475", "601088",
]


def _symbol(code: str) -> str:
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _cache_path(code: str) -> Path:
    return CACHE_DIR / f"{code}.csv"


def _read_csv(path: Path) -> List[KLineBar]:
    with open(path, encoding="utf-8") as f:
        return [KLineBar(date=r["date"], open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                         close=float(r["close"]), volume=float(r["volume"])) for r in csv.DictReader(f)]


def _write_csv(path: Path, bars: List[KLineBar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.date, b.open, b.high, b.low, b.close, b.volume])


def fetch_history(code: str, start: str = "2019-01-01", end: str = "2025-12-31",
                  use_cache: bool = True, timeout: float = 8.0) -> List[KLineBar]:
    """返回按日期升序的日 K。缓存命中则不联网；联网失败返回空列表(由调用方决定如何处理，不会静默造数据)"""
    path = _cache_path(code)
    cached = []
    if use_cache and path.exists():
        cached = _read_csv(path)
        filtered = [b for b in cached if start <= b.date <= end]
        if filtered and cached[0].date <= start and (cached[-1].date >= end or cached[-1].date >= "2026-10-21"):
            return filtered

    import requests
    sym, bars_dict, cursor = _symbol(code), {}, end
    for b in cached:
        bars_dict[b.date] = b

    tencent_success = False
    for _ in range(12):                                             # 每次最多约 640 根，向前翻页
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,{start},{cursor},640,qfq"
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                data = resp.json()
                node = (data.get("data") or {}).get(sym, {})
                raw = node.get("qfqday") or node.get("day") or []
                if not raw:
                    break
                for b in raw:
                    if len(b) >= 6:
                        bars_dict[str(b[0])] = KLineBar(date=str(b[0]), open=float(b[1]), close=float(b[2]), high=float(b[3]),
                                                        low=float(b[4]), volume=float(b[5]))
                first = min(bars_dict)
                tencent_success = True
                if first <= start or len(raw) < 600:
                    break
                cursor = first
            else:
                break
        except Exception:
            break

    # 若腾讯接口不可用（例如触发 WAF 501 或翻页不全），回退使用东方财富前复权日 K 接口
    if not tencent_success or min(bars_dict.keys(), default="9999") > start:
        secid = ("1." if code.startswith(("6", "9")) else "0.") + code
        em_url = (f"http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}"
                  f"&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56"
                  f"&klt=101&fqt=1&beg={start.replace('-', '')}&end={end.replace('-', '')}")
        try:
            em_resp = requests.get(em_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}).json()
            klines = (em_resp.get("data") or {}).get("klines", [])
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 6:
                    bars_dict[parts[0]] = KLineBar(
                        date=parts[0],
                        open=float(parts[1]),
                        close=float(parts[2]),
                        high=float(parts[3]),
                        low=float(parts[4]),
                        volume=float(parts[5]),
                    )
        except Exception as e:
            logger.warning("拉取 %s 失败: %s", code, e)

    out = [bars_dict[d] for d in sorted(bars_dict)]
    if out:
        _write_csv(path, out)
    return [b for b in out if start <= b.date <= end]


def load_universe(codes: Optional[List[str]] = None, **kw) -> Dict[str, List[KLineBar]]:
    res = {}
    for c in codes or DEFAULT_UNIVERSE:
        bars = fetch_history(c, **kw)
        if len(bars) >= 200:
            res[c] = bars
    return res


def synthetic_universe(n_tickers: int = 20, n_days: int = 500, seed: int = 0,
                       planted_signal: float = 0.0) -> Dict[str, List[KLineBar]]:
    """
    合成数据(仅用于代码自检)。
    planted_signal=0：收益为独立同分布噪声 → 任何指标的真实 IC 都应≈0（用来确认实验流程不会“凭空产生显著性”）。
    planted_signal>0：让“前5日收益为负”的股票次日期望收益略高(人为植入的反转效应) → 实验流程应能把它测出来。
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, List[KLineBar]] = {}
    for i in range(n_tickers):
        sigma = 0.012 + 0.01 * rng.random()
        rets = rng.normal(0.0, sigma, n_days)
        if planted_signal:
            for t in range(5, n_days):
                past5 = rets[t - 5:t].sum()
                rets[t] += -planted_signal * np.sign(past5) * sigma * 0.5 * min(abs(past5) / (sigma * 2.2), 2.0)
        close, bars = 50.0 + 10 * rng.random(), []
        for t in range(n_days):
            o = close * (1 + rng.normal(0, sigma * 0.2))
            c = close * (1 + rets[t])
            hi, lo = max(o, c) * (1 + abs(rng.normal(0, sigma * 0.3))), min(o, c) * (1 - abs(rng.normal(0, sigma * 0.3)))
            vol = float(1e6 * (1 + 3 * abs(rets[t]) / sigma * 0.3) * (0.8 + 0.4 * rng.random()))
            bars.append(KLineBar(date=f"D{t:04d}", open=round(o, 4), high=round(hi, 4), low=round(lo, 4),
                                 close=round(c, 4), volume=round(vol)))
            close = c
        out[f"SYN{i:03d}"] = bars
    return out

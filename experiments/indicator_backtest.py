"""
实验一：旧技术面(MACD+RSI+布林) vs 新技术面(趋势+动量/反转+量能+波动率) 的预测力对比。

不涉及 LLM。对每只股票、每个交易日 t：
  - 只用 t 日及以前的 K 线计算两套评分（无未来函数）；
  - 前向收益 = 第 t+1 日开盘买入，持有 h 日后收盘卖出（h=1/3/5）。不用 t 日收盘价成交，避免“看到收盘价再按收盘价买”。
统计量：横截面 Spearman IC（日度）+ 块自助法 95% CI、高低分位价差(含成本)、命中率 vs 基准胜率。

用法（在你自己的机器上，需要联网拉取日 K）：
    python -m experiments.indicator_backtest --start 2019-01-01 --end 2025-12-31 --cost-bps 20
自检（合成数据，不联网）：
    python -m experiments.indicator_backtest --synthetic
"""
import argparse
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np

from app.models.schemas import KLineBar
from app.tools.quant_indicators import quant_indicators_engine
from app.tools.scoring import legacy_technical_score, score_technical
from experiments.data_loader import load_universe, synthetic_universe
from experiments.metrics import block_bootstrap_ci, cross_sectional_ic, hit_rate, quantile_spread

HORIZONS = (1, 3, 5)
WINDOW = 80


def collect(universe: Dict[str, List[KLineBar]], step: int = 1) -> Dict[str, dict]:
    """返回 {因子名: {h: {date: [(score, ret), ...]}}}"""
    acc = {name: {h: defaultdict(lambda: ([], [])) for h in HORIZONS} for name in
           ("legacy_macd_rsi_boll", "new_composite", "rev5_only", "low_vol_only", "volume_confirm_only")}
    for code, bars in universe.items():
        n = len(bars)
        for t in range(WINDOW, n - max(HORIZONS) - 1, step):
            ind = quant_indicators_engine.calculate_indicators(bars[t - WINDOW + 1:t + 1])
            feats = {
                "legacy_macd_rsi_boll": legacy_technical_score(ind),
                "new_composite": score_technical(ind)[0],
                "rev5_only": -ind.ret_5d,                    # 单因子：近5日跌得越多得分越高(短期反转)
                "low_vol_only": -ind.realized_vol_20,        # 单因子：波动越低得分越高
                "volume_confirm_only": ind.obv_slope,        # 单因子：OBV 斜率
            }
            entry = bars[t + 1].open
            if entry <= 0:
                continue
            date = bars[t].date
            for h in HORIZONS:
                ret = bars[t + h].close / entry - 1.0
                for name, v in feats.items():
                    s, r = acc[name][h][date]
                    s.append(v)
                    r.append(ret)
    return acc


def summarize(acc, cost_bps: float, signal_threshold: float = 6.3) -> dict:
    out = {}
    for name, per_h in acc.items():
        out[name] = {}
        for h, by_date in per_h.items():
            sc = {d: v[0] for d, v in by_date.items()}
            rt = {d: v[1] for d, v in by_date.items()}
            ics = cross_sectional_ic(sc, rt)
            ic_m, ic_lo, ic_hi = block_bootstrap_ci(ics, block=max(5, h))
            sp = quantile_spread(sc, rt, cost_bps_round_trip=0.0)
            sp_net = quantile_spread(sc, rt, cost_bps_round_trip=cost_bps)
            sp_m, sp_lo, sp_hi = block_bootstrap_ci(sp_net, block=max(5, h))
            flat_s = [x for d in sc for x in sc[d]]
            flat_r = [x for d in rt for x in rt[d]]
            entry = {
                "n_dates": len(ics), "ic_mean": ic_m, "ic_ci95": [ic_lo, ic_hi],
                "ic_ir": float(np.mean(ics) / np.std(ics)) if len(ics) > 2 and np.std(ics) > 0 else float("nan"),
                "spread_gross": float(np.mean(sp)) if sp else float("nan"),
                "spread_net": sp_m, "spread_net_ci95": [sp_lo, sp_hi],
            }
            if name in ("legacy_macd_rsi_boll", "new_composite"):
                entry["hit"] = hit_rate(flat_s, flat_r, signal_threshold)
            out[name][f"{h}d"] = entry
    return out


def _fmt(x, p=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def to_markdown(res: dict, label: str, cost_bps: float) -> str:
    lines = [f"# 技术指标对比实验结果（数据：{label}，成本 {cost_bps:g}bps/往返）", "",
             "| 因子 | 持有期 | 日度IC均值 | IC 95%CI | IC_IR | 高低分位价差(毛) | 价差(净) | 净价差95%CI |",
             "|---|---|---|---|---|---|---|---|"]
    for name, per_h in res.items():
        for h, e in per_h.items():
            lines.append(f"| {name} | {h} | {_fmt(e['ic_mean'])} | [{_fmt(e['ic_ci95'][0])}, {_fmt(e['ic_ci95'][1])}] | "
                         f"{_fmt(e['ic_ir'], 2)} | {_fmt(e['spread_gross'] * 100, 3)}% | {_fmt(e['spread_net'] * 100, 3)}% | "
                         f"[{_fmt(e['spread_net_ci95'][0] * 100, 3)}%, {_fmt(e['spread_net_ci95'][1] * 100, 3)}%] |")
    lines += ["", "命中率（得分≥阈值 的样本，次日起持有 h 日收益>0 的占比 vs 全样本基准胜率）：", ""]
    for name in ("legacy_macd_rsi_boll", "new_composite"):
        for h, e in res[name].items():
            hr = e["hit"]
            lines.append(f"- {name} {h}: 信号数 {hr['n_signal']}，命中率 {_fmt(hr['hit_rate'], 3)} vs 基准 {_fmt(hr['base_rate'], 3)}；"
                         f"信号平均收益 {_fmt(hr['mean_ret_signal'] * 100, 3)}% vs 全样本 {_fmt(hr['mean_ret_all'] * 100, 3)}%")
    lines += ["", "解读规则：IC 的 95%CI 若包含 0，则不能认为该指标有预测力；价差(净)CI 包含 0 同理。",
              "局限：股票池存在幸存者偏差；未处理涨跌停/停牌无法成交；开盘价成交假设理想化；同一批数据上调过阈值则结果会偏乐观，应留出样本外区间复核。"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="使用合成数据自检(不联网，结果不代表真实市场)")
    ap.add_argument("--planted", type=float, default=0.0, help="合成数据中植入的反转效应强度")
    ap.add_argument("--codes", nargs="*")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default="experiments/results_indicator.md")
    a = ap.parse_args()
    if a.synthetic:
        uni, label = synthetic_universe(planted_signal=a.planted), f"合成数据(planted={a.planted}) — 仅自检，非真实市场"
    else:
        uni = load_universe(a.codes, start=a.start, end=a.end)
        label = f"真实日K {a.start}~{a.end}，{len(uni)} 只"
        if not uni:
            raise SystemExit("未获取到任何真实数据：请确认网络可访问腾讯行情接口，或先把 CSV 放入 experiments/data_cache/")
    res = summarize(collect(uni, a.step), a.cost_bps)
    md = to_markdown(res, label, a.cost_bps)
    open(a.out, "w", encoding="utf-8").write(md)
    open(a.out.replace(".md", ".json"), "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=2))
    print(md)


if __name__ == "__main__":
    main()

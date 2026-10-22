"""
实验三：技术面分项消融 + 时间段切分（防止“看着结果调参数”造成的过拟合）

做什么：新技术面 = 5 + 趋势 + 动量/反转 + 量能 + 波动率 四个分项。对每个分项：
  - drop_X ：去掉该分项后的综合分（若 drop_X 的 IC 比 full 高，说明该分项在拖后腿）
  - only_X ：只保留该分项（单独的预测力）
并和旧版 legacy、完整新版 full 比较。

时间段切分（默认）：
  train = 起点 ~ 2022-12-31   —— 允许你看结果、提假设
  valid = 2023-01-01 ~ 2023-12-31 —— 用来确认 train 上的发现是否还成立
  test  = 2024-01-01 ~ 终点     —— 只在最终决定之后看一次；看完不得再据此改动

纪律：只有当某个改动在 valid 与 test 上都改善、且 drop−full 的差值 CI 不含 0，才允许写进评分函数。
否则保持现状，并如实记录“无证据”。

用法：
    python -m experiments.component_ablation --start 2019-01-01 --end 2025-12-31
自检：
    python -m experiments.component_ablation --synthetic
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
from experiments.metrics import block_bootstrap_ci, cross_sectional_ic, quantile_spread

HORIZONS = (1, 3, 5)
WINDOW = 80
COMPS = ("trend", "momentum_reversal", "volume", "volatility")


def collect(universe: Dict[str, List[KLineBar]], step: int = 1):
    """返回 {因子: {h: {date: ([scores],[rets])}}}"""
    names = ["legacy", "full"] + [f"drop_{c}" for c in COMPS] + [f"only_{c}" for c in COMPS]
    acc = {n: {h: defaultdict(lambda: ([], [])) for h in HORIZONS} for n in names}
    for code, bars in universe.items():
        for t in range(WINDOW, len(bars) - max(HORIZONS) - 1, step):
            ind = quant_indicators_engine.calculate_indicators(bars[t - WINDOW + 1:t + 1])
            full, comp = score_technical(ind)
            feats = {"legacy": legacy_technical_score(ind), "full": full}
            for c in COMPS:
                feats[f"drop_{c}"] = 5.0 + sum(v for k, v in comp.items() if k != c)
                feats[f"only_{c}"] = 5.0 + comp[c]
            entry = bars[t + 1].open
            if entry <= 0:
                continue
            date = bars[t].date
            for h in HORIZONS:
                ret = bars[t + h].close / entry - 1.0
                for n, v in feats.items():
                    s, r = acc[n][h][date]
                    s.append(v)
                    r.append(ret)
    return acc


def _slice(acc, lo: str, hi: str):
    return {n: {h: {d: v for d, v in by.items() if lo <= d <= hi} for h, by in per.items()} for n, per in acc.items()}


def _ic_series(by_date):
    return cross_sectional_ic({d: v[0] for d, v in by_date.items()}, {d: v[1] for d, v in by_date.items()})


def summarize(acc, cost_bps: float) -> dict:
    out = {}
    for n, per in acc.items():
        out[n] = {}
        for h, by in per.items():
            ics = _ic_series(by)
            m, lo, hi = block_bootstrap_ci(ics, block=max(5, h))
            sp = quantile_spread({d: v[0] for d, v in by.items()}, {d: v[1] for d, v in by.items()},
                                 cost_bps_round_trip=cost_bps)
            out[n][f"{h}d"] = {"n_dates": len(ics), "ic": m, "ic_ci": [lo, hi],
                               "spread_net": float(np.mean(sp)) if sp else float("nan")}
    return out


def paired_diff(acc, a: str, b: str, h: int):
    """同一批日期上 IC_a − IC_b 的均值与块自助 CI（配对，比各自的 CI 比较更有说服力）"""
    sa, sb = acc[a][h], acc[b][h]
    ds = sorted(set(sa) & set(sb))
    ia = cross_sectional_ic({d: sa[d][0] for d in ds}, {d: sa[d][1] for d in ds})
    ib = cross_sectional_ic({d: sb[d][0] for d in ds}, {d: sb[d][1] for d in ds})
    n = min(len(ia), len(ib))
    return block_bootstrap_ci([x - y for x, y in zip(ia[:n], ib[:n])], block=max(5, h))


def run(universe, cost_bps, bounds):
    acc = collect(universe)
    periods = {"train": (bounds[0], bounds[1]), "valid": (bounds[1], bounds[2]), "test": (bounds[2], bounds[3])}
    res = {}
    for pname, (lo, hi) in periods.items():
        part = _slice(acc, lo, hi)
        res[pname] = {"summary": summarize(part, cost_bps), "paired_drop_minus_full": {}}
        for c in COMPS:
            res[pname]["paired_drop_minus_full"][c] = {f"{h}d": paired_diff(part, f"drop_{c}", "full", h)
                                                       for h in (3, 5)}
    return res


def to_markdown(res, label, cost_bps):
    f = lambda x, p=4: "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"
    L = [f"# 技术面分项消融（{label}，成本 {cost_bps:g}bps）", ""]
    for p, d in res.items():
        L += [f"## {p}", "", "| 因子 | 3d IC [95%CI] | 5d IC [95%CI] | 5d 净价差 |", "|---|---|---|---|"]
        for n, per in d["summary"].items():
            a, b = per["3d"], per["5d"]
            L.append(f"| {n} | {f(a['ic'])} [{f(a['ic_ci'][0])}, {f(a['ic_ci'][1])}] | "
                     f"{f(b['ic'])} [{f(b['ic_ci'][0])}, {f(b['ic_ci'][1])}] | {f(b['spread_net'] * 100, 3)}% |")
        L += ["", "去掉某分项后 IC 的变化（drop - full，配对；CI 含 0 即无证据表明该分项有害/有益）：", ""]
        for c, v in d["paired_drop_minus_full"].items():
            L.append(f"- 去掉 {c}: 3d {f(v['3d'][0])} [{f(v['3d'][1])}, {f(v['3d'][2])}]；"
                     f"5d {f(v['5d'][0])} [{f(v['5d'][1])}, {f(v['5d'][2])}]")
        L.append("")
    L += ["纪律：只有 valid 与 test 都改善、且配对差值 CI 不含 0 的改动才可采纳；test 只看一次。"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--planted", type=float, default=0.0)
    ap.add_argument("--codes", nargs="*")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--valid-start", default="2023-01-01")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    ap.add_argument("--out", default="experiments/results_component.md")
    a = ap.parse_args()
    if a.synthetic:
        uni = synthetic_universe(n_tickers=14, n_days=700, seed=5, planted_signal=a.planted)
        bounds = ("D0000", "D0300", "D0500", "D9999")
        label = f"合成数据 planted={a.planted} — 仅自检"
    else:
        uni = load_universe(a.codes, start=a.start, end=a.end)
        if not uni:
            raise SystemExit("未获取到数据")
        bounds = (a.start, a.valid_start, a.test_start, a.end)
        label = f"真实日K {a.start}~{a.end}，{len(uni)} 只；valid≥{a.valid_start}，test≥{a.test_start}"
    res = run(uni, a.cost_bps, bounds)
    md = to_markdown(res, label, a.cost_bps)
    open(a.out, "w", encoding="utf-8").write(md)
    open(a.out.replace(".md", ".json"), "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=2))
    print(md)


if __name__ == "__main__":
    main()

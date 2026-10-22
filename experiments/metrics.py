"""实验用统计量：Spearman IC、分位数价差、命中率、Bootstrap 置信区间、交易成本。只依赖 numpy。"""
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # 并列取平均秩
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def block_bootstrap_ci(x: Sequence[float], n_boot: int = 2000, block: int = 5, alpha: float = 0.05,
                       seed: int = 0) -> Tuple[float, float, float]:
    """序列均值的块自助法置信区间(块长≥持有期，处理前向收益重叠造成的自相关)。返回 (均值, 下界, 上界)"""
    x = np.asarray([v for v in x if not np.isnan(v)], float)
    n = len(x)
    if n < block * 3:
        return (float(np.mean(x)) if n else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        means[i] = np.concatenate([x[s:s + block] for s in starts])[:n].mean()
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cross_sectional_ic(scores_by_date: Dict[str, List[float]], rets_by_date: Dict[str, List[float]],
                       min_names: int = 8) -> List[float]:
    ics = []
    for d in sorted(scores_by_date):
        s, r = scores_by_date[d], rets_by_date[d]
        if len(s) >= min_names:
            ic = spearman(s, r)
            if not np.isnan(ic):
                ics.append(ic)
    return ics


def quantile_spread(scores_by_date: Dict[str, List[float]], rets_by_date: Dict[str, List[float]],
                    q: float = 0.2, min_names: int = 10, cost_bps_round_trip: float = 0.0) -> List[float]:
    """每个日期：得分最高 q 分位的平均收益 - 最低 q 分位的平均收益。成本按“多头一次往返 + 空头一次往返”粗略扣除"""
    out = []
    for d in sorted(scores_by_date):
        s, r = np.asarray(scores_by_date[d]), np.asarray(rets_by_date[d])
        if len(s) < min_names:
            continue
        k = max(1, int(len(s) * q))
        idx = np.argsort(s, kind="mergesort")
        out.append(r[idx[-k:]].mean() - r[idx[:k]].mean() - 2 * cost_bps_round_trip / 1e4)
    return out


def hit_rate(scores: Sequence[float], rets: Sequence[float], threshold: float) -> Dict[str, float]:
    """得分≥threshold 的样本中次后收益>0 的占比，并与全样本基准胜率比较(避免把“牛市里本来就涨”误当作信号)"""
    s, r = np.asarray(scores, float), np.asarray(rets, float)
    sel = s >= threshold
    return {"n_signal": int(sel.sum()), "hit_rate": float((r[sel] > 0).mean()) if sel.any() else float("nan"),
            "base_rate": float((r > 0).mean()) if len(r) else float("nan"),
            "mean_ret_signal": float(r[sel].mean()) if sel.any() else float("nan"),
            "mean_ret_all": float(r.mean()) if len(r) else float("nan")}

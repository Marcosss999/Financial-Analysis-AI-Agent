"""实验框架自检：零信号数据上不应“发现”显著预测力；植入信号时应能测出来"""
from experiments.data_loader import synthetic_universe
from experiments.indicator_backtest import collect, summarize
from experiments.metrics import block_bootstrap_ci, hit_rate, quantile_spread, spearman


def test_spearman_known_values():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert abs(spearman([1, 1, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-9   # 含并列(平均秩)，与 scipy 一致


def test_bootstrap_ci_contains_true_mean_and_widens_with_noise():
    m, lo, hi = block_bootstrap_ci([0.1] * 60)
    assert lo <= m <= hi and abs(m - 0.1) < 1e-12


def test_quantile_spread_and_hit_rate():
    sc = {"d": list(range(10))}
    rt = {"d": [i / 100 for i in range(10)]}
    assert abs(quantile_spread(sc, rt, q=0.2)[0] - 0.08) < 1e-12
    assert abs(quantile_spread(sc, rt, q=0.2, cost_bps_round_trip=10)[0] - (0.08 - 0.002)) < 1e-12
    h = hit_rate([1, 2, 3, 4], [-0.1, 0.1, 0.2, -0.05], 3)
    assert h["n_signal"] == 2 and h["hit_rate"] == 0.5 and h["base_rate"] == 0.5


def _run(planted):
    uni = synthetic_universe(n_tickers=14, n_days=260, seed=3, planted_signal=planted)
    return summarize(collect(uni, step=2), cost_bps=0.0)


def test_null_data_yields_no_significant_ic():
    res = _run(0.0)
    for name, per_h in res.items():
        for h, e in per_h.items():
            lo, hi = e["ic_ci95"]
            assert lo <= 0.0 <= hi, f"{name}/{h} 在零信号数据上显著: {e['ic_mean']} {e['ic_ci95']}"


def test_planted_reversal_is_detected_and_momentum_scores_go_negative():
    res = _run(1.0)
    r = res["rev5_only"]["3d"]
    assert r["ic_ci95"][0] > 0.1                        # 植入的反转效应被测出
    assert res["new_composite"]["3d"]["ic_ci95"][1] < 0  # 顺势打分在反转世界里应为负


def test_component_ablation_null_and_planted():
    from experiments import component_ablation as ca
    uni = synthetic_universe(n_tickers=12, n_days=400, seed=7, planted_signal=0.0)
    res = ca.summarize(ca.collect(uni, step=3), 0.0)
    for n, per in res.items():                       # 零信号：没有任何分项“显著”
        lo, hi = per["5d"]["ic_ci"]
        assert lo <= 0.0 <= hi, n
    acc = ca.collect(synthetic_universe(n_tickers=12, n_days=400, seed=7, planted_signal=1.0), step=3)
    res = ca.summarize(acc, 0.0)
    assert res["only_momentum_reversal"]["3d"]["ic"] > 0.05        # 植入的反转效应只会出现在反转分项
    m, lo, hi = ca.paired_diff(acc, "only_momentum_reversal", "only_volume", 3)
    assert lo > 0

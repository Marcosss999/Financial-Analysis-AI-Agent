# 实验目录说明

| 脚本 | 回答的问题 | 是否需要 LLM | 是否需要联网 |
|---|---|---|---|
| `indicator_backtest.py` | 新技术面（趋势+动量/反转+量能+波动率）比旧的 MACD+RSI+布林更能预测次日起 1/3/5 日收益吗？ | 否 | 是（拉日K，会缓存成 CSV） |
| `debate_ablation.py` | 多空辩论开启后，最终评分比关闭时更准吗？ | **是**（有费用） | 是 |

## 运行
```bash
# 1) 先自检（不联网，合成数据）：零信号数据应 "没有显著性"，植入反转效应应被测出
python -m experiments.indicator_backtest --synthetic
python -m experiments.indicator_backtest --synthetic --planted 1.0
# 2) 真实数据（在你自己的机器上）
python -m experiments.indicator_backtest --start 2019-01-01 --end 2025-12-31 --cost-bps 20
# 3) 辩论消融（需 .env 里配置真实 Key；样本日期必须晚于模型训练截止日）
python -m experiments.debate_ablation --after 2026-07-01 --dates-step 5          # 先看费用估算
python -m experiments.debate_ablation --after 2026-07-01 --dates-step 5 --yes
```
结果写入 `experiments/results_*.md|json`。**仓库里不含任何真实回测数字**——这些数字必须由你在自己的环境里跑出来，再如实写进简历/报告。

## 怎么读结果
- IC 的 95% 置信区间包含 0 → 不能说该指标有预测力。
- “价差(净)”已扣往返成本；短线策略常见现象是毛价差为正、扣费后为负，这本身就是有价值的结论。
- 命中率要和**全样本基准胜率**比，牛市里随便买也有 55% 胜率。

## 已知局限（写在简历/面试里要主动说）
- 股票池存在幸存者偏差；未处理涨跌停/停牌导致的无法成交；以次日开盘价成交是理想化假设。
- 评分里的分段阈值是依据常识设定的先验，没有在数据上拟合；若你根据回测结果去调阈值，必须留出样本外区间复核，否则会过拟合。
- 辩论消融中，财报是“当前版本”（非点时），所以实验固定只评价技术面；资金流/公告/新闻已屏蔽；LLM 对旧行情可能有记忆污染。
- 仓库自带的 `tests/test_experiments.py` 只证明“实验流程本身没写错”（零信号不出显著性、植入信号能测出），不证明任何指标在真实市场有效。

## 实验三：技术面分项消融（`component_ablation.py`）
```bash
python -m experiments.component_ablation --synthetic                     # 自检
python -m experiments.component_ablation --start 2019-01-01 --end 2025-12-31
```
把数据切成 train(~2022) / valid(2023) / test(2024~)。**train 上提假设，valid 上确认，test 只看一次。** 去掉某分项后 IC 的变化用配对 Bootstrap 给出 CI；只有 valid 与 test 都改善且 CI 不含 0 才允许改评分函数，否则保持现状。

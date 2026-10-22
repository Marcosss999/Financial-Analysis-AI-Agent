"""
实验二：多空辩论开 / 关 的消融实验 —— 辩论是否让最终评分更准？

⚠️ 需要真实 LLM（.env 里配置好 Key），且会产生 API 费用；离线模式下辩论只是规则模板，测它没有意义，脚本会拒绝运行。

设计要点（都是为了不被假结果误导）：
1. 点时(point-in-time)：每个样本只给 LLM 看截至 t 日的 K 线；资金流/公告/新闻一律置空（接口只给“当前”数据，会造成未来函数）；
   财报也只有“当前”版本，所以本实验固定 technical_weight=1，只评价“技术面 + 辩论”，不评价基本面。
2. 模型记忆污染：LLM 可能“记得”历史行情。样本日期必须晚于所用模型的训练截止日（用 --after 指定），否则结论无效。
3. 配对比较：同一 (股票, 日期) 分别跑 debate=on / off，对 IC 差值做块自助法置信区间；CI 含 0 则不能说辩论有帮助。
4. 前向收益：t+1 开盘买入、持有 h 日收盘卖出。
5. 成本估算：运行前打印预计 LLM 调用次数，--yes 才执行。

用法：
    python -m experiments.debate_ablation --after 2026-07-01 --tickers 600519 000858 ... --dates-step 5 --yes
"""
import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.core.llm_provider import llm_provider
from app.core.observability import start_trace
from app.graph.financial_graph import build_financial_graph
from app.graph.state_utils import merge_state
from app.models.schemas import FinancialStatements, MarketIntel, MarketQuote
from experiments.data_loader import DEFAULT_UNIVERSE, fetch_history
from experiments.metrics import block_bootstrap_ci, spearman

CALLS_PER_RUN_OFF, CALLS_PER_RUN_ON = 4, 9   # 估算值：三位专家+仲裁；开启辩论额外多空各2轮+仲裁


async def _score(bars, code: str, t: int, debate: bool) -> float:
    hist = bars[max(0, t - 59):t + 1]
    quote = MarketQuote(ticker=code, name=code, current_price=hist[-1].close, change_pct=0.0, pe_ttm=20.0,
                        pb_ratio=3.0, market_cap=1000.0, kline_history=hist, data_source="real")
    intel = MarketIntel(main_capital_flow="(点时实验：已屏蔽)", capital_inflow_amount_str="n/a", is_main_inflow=False,
                        announcements=[], news_headlines=[], flow_net_yuan=None, data_source="unavailable")
    fin = FinancialStatements(ticker=code, period="点时实验(已屏蔽)", revenue=100.0, revenue_yoy=5.0, net_profit=10.0,
                              net_profit_yoy=5.0, gross_margin=30.0, net_margin=10.0, total_assets=200.0,
                              total_liabilities=80.0, debt_to_asset_ratio=40.0, operating_cash_flow=15.0,
                              free_cash_flow=10.0, roe=15.0, ebit=12.0, interest_expense=2.0)
    state = {"ticker": code, "enable_debate": debate, "strategy_style": "custom", "quote": quote, "financials": fin, "market_intel": intel,
             "fundamental_weight": 0.0, "technical_weight": 1.0, "macro_weight": 0.0, "capital_flow_weight": 0.0,
             "execution_logs": [], "debate_turns": [], "degraded_nodes": []}
    start_trace(code)
    final = dict(state)
    async for chunk in build_financial_graph().astream(dict(state)):
        for _, upd in chunk.items():
            merge_state(final, upd)
    return final["arbiter_verdict"].consensus_score


async def run(args) -> dict:
    if llm_provider.status()["mode"] != "live":
        raise SystemExit("未检测到真实 LLM（离线模式）。辩论消融必须在 LLM 在线时运行，否则测的只是规则模板。")
    tickers = args.tickers or DEFAULT_UNIVERSE[:10]
    # 预热区间：LLM 需要 t 日之前约 60 根 K 线，所以从 after 往前多取 150 个自然日；
    # 但被评估的样本日期只取 >= after 的，保证晚于模型训练截止日。
    warm_start = (datetime.strptime(args.after, "%Y-%m-%d") - timedelta(days=150)).strftime("%Y-%m-%d")
    data = {c: fetch_history(c, start=warm_start, end=args.before) for c in tickers}
    data = {c: b for c, b in data.items() if len(b) >= 70}
    if not data:
        raise SystemExit("没有可用数据：请确认网络与日期区间，样本区间需包含 --after 之后至少 70 个交易日。")
    common = sorted(set.intersection(*[{b.date for b in bars} for bars in data.values()]))
    # 前 60 根作预热；末尾留出 horizon+1 根用于计算前向收益；只评估 after 之后的日期
    usable = common[60:len(common) - (args.horizon + 1)]
    dates = [d for d in usable if d >= args.after][::args.dates_step]
    n_runs = len(data) * len(dates)
    est = n_runs * (CALLS_PER_RUN_ON + CALLS_PER_RUN_OFF)
    print(f"样本：{len(data)} 只 × {len(dates)} 个日期 = {n_runs} 组；预计 LLM 调用 ≈ {est} 次")
    if not args.yes:
        raise SystemExit("确认费用后加 --yes 重新运行。")
    chk_path = Path(args.out).with_suffix(".checkpoint.json")
    rows: Dict[str, Dict[str, list]] = {d: {"on": [], "off": [], "ret": []} for d in dates}
    completed_dates = set()
    if chk_path.exists():
        try:
            with open(chk_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for d, content in saved.get("rows", {}).items():
                    if d in rows and len(content.get("on", [])) >= len(data):
                        rows[d] = content
                        completed_dates.add(d)
                if completed_dates:
                    print(f"[*] 发现历史检查点，已恢复 {len(completed_dates)} 个已完成日期，继续处理剩余日期...")
        except Exception:
            pass

    sem = asyncio.Semaphore(4)  # 控制并发度为4，既加速又不超 DeepSeek 限流

    async def _eval_one(c, bars, d):
        async with sem:
            idx = {b.date: i for i, b in enumerate(bars)}
            t = idx[d]
            on, off = await asyncio.gather(_score(bars, c, t, True), _score(bars, c, t, False))
            ret = bars[t + args.horizon].close / bars[t + 1].open - 1.0
            return c, on, off, ret

    for d_idx, d in enumerate(dates, start=1):
        if d in completed_dates:
            continue
        print(f"[*] 评估进度 [{d_idx}/{len(dates)}] 正在处理截面日期: {d} (标的数: {len(data)})...")
        tasks = [_eval_one(c, bars, d) for c, bars in data.items()]
        results = await asyncio.gather(*tasks)
        for _, on, off, ret in results:
            rows[d]["on"].append(on)
            rows[d]["off"].append(off)
            rows[d]["ret"].append(ret)

        # 实时增量保存检查点
        try:
            with open(chk_path, "w", encoding="utf-8") as f:
                json.dump({"completed_dates": list(completed_dates | {d}), "total_dates": len(dates), "rows": rows}, f, ensure_ascii=False)
        except Exception:
            pass

    # 运行完毕清理临时检查点
    if chk_path.exists():
        try:
            chk_path.unlink()
        except Exception:
            pass
    ic_on = [spearman(r["on"], r["ret"]) for r in rows.values() if len(r["on"]) >= 8]
    ic_off = [spearman(r["off"], r["ret"]) for r in rows.values() if len(r["off"]) >= 8]
    diff = [a - b for a, b in zip(ic_on, ic_off) if not (np.isnan(a) or np.isnan(b))]
    out = {"n_dates": len(diff), "ic_debate_on": block_bootstrap_ci(ic_on), "ic_debate_off": block_bootstrap_ci(ic_off),
           "ic_diff_on_minus_off": block_bootstrap_ci(diff), "horizon": args.horizon, "after": args.after,
           "caveat": "CI 含 0 即不能认为辩论提升了准确性；样本日期须晚于模型训练截止日；财报/资金流/新闻已屏蔽。"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--after", required=True, help="样本起始日(须晚于所用 LLM 的训练截止日)")
    ap.add_argument("--before", default="2099-12-31")
    ap.add_argument("--dates-step", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--out", default="experiments/results_debate.json")
    a = ap.parse_args()
    res = asyncio.run(run(a))
    open(a.out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

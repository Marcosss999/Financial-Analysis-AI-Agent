import os
import tempfile

from app.core.llm_provider import llm_provider
from app.memory.chat_memory import ChatMemory, extractive_summary, format_run_brief
from app.memory.store import MemoryStore
from tests.helpers import override, run
from app.core.config import settings


def _run_row(i, ticker="600519", score=7.0, verdict="持有"):
    return {"run_id": f"r{i}", "session_id": "s", "ticker": ticker, "target_name": "贵州茅台",
            "created_at": 1000.0 + i, "strategy_style": "balanced", "verdict": verdict, "consensus_score": score,
            "position": "30%", "risk_status": "APPROVED", "is_vetoed": False,
            "data_quality": {"quote_source": "real"}, "summary": "摘要", "report_md": "# 研报", "trace": {"spans": []}}


def test_store_recall_orders_by_time_and_scopes_by_ticker():
    st = MemoryStore(":memory:")
    for i in range(4):
        st.save_run(_run_row(i))
    st.save_run(_run_row(9, ticker="000001"))
    recent = st.recent_runs("600519", limit=3)
    assert [r["run_id"] for r in recent] == ["r3", "r2", "r1"]
    assert "report_md" not in recent[0]                       # 召回只带摘要，不带研报全文
    assert st.recent_runs("600519", limit=3, exclude_run_id="r3")[0]["run_id"] == "r2"
    full = st.get_run("r0")
    assert full["report_md"] == "# 研报" and full["trace"] == {"spans": []} and full["data_quality"]["quote_source"] == "real"
    assert st.get_run("nope") is None


def test_memory_persists_across_store_instances():
    path = os.path.join(tempfile.mkdtemp(), "m.db")
    a = MemoryStore(path)
    a.save_run(_run_row(1))
    a.add_message("sess", "user", "你好")
    a.close()
    b = MemoryStore(path)                                      # 模拟服务重启
    assert b.get_run("r1") and b.get_messages("sess")[0]["content"] == "你好"


def test_chat_context_compresses_old_turns_and_keeps_recent_verbatim():
    st = MemoryStore(":memory:")
    cm = ChatMemory(st)
    for i in range(10):
        run(cm.remember("s1", "600519", f"问题{i}", f"回答{i}"))
    with override(settings.eng, CHAT_KEEP_RECENT_TURNS=3):
        msgs = run(cm.build_context("s1", "600519", "SYS", "新问题"))
    contents = [m["content"] for m in msgs]
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "新问题"}
    assert "问题9" in contents and "回答9" in contents and "问题0" not in contents    # 近期原文保留，久远原文被压缩
    assert "此前对话摘要" in msgs[0]["content"] and "问题0" in msgs[0]["content"]       # 久远内容以摘要形式保留
    row = st.get_summary("s1")
    assert row and row["upto_id"] == 14                       # 10 轮共 20 条，保留最近 6 条 → 压缩前 14 条


def test_rolling_summary_only_summarizes_new_overflow():
    st = MemoryStore(":memory:")
    cm = ChatMemory(st)
    with override(settings.eng, CHAT_KEEP_RECENT_TURNS=2):
        for i in range(5):
            run(cm.remember("s", "T", f"q{i}", f"a{i}"))
        run(cm.build_context("s", "T", "SYS", "x"))
        first = st.get_summary("s")["upto_id"]
        run(cm.remember("s", "T", "q5", "a5"))
        run(cm.build_context("s", "T", "SYS", "x"))
        assert st.get_summary("s")["upto_id"] == first + 2    # 只多压缩了新溢出的 2 条


def test_context_has_a_hard_character_budget():
    st = MemoryStore(":memory:")
    cm = ChatMemory(st)
    for i in range(6):
        run(cm.remember("s", "T", "长" * 800, "答" * 800))
    with override(settings.eng, CHAT_CONTEXT_CHAR_BUDGET=3000, CHAT_KEEP_RECENT_TURNS=6):
        msgs = run(cm.build_context("s", "T", "SYS", "问"))
    assert sum(len(m["content"]) for m in msgs[1:-1]) <= 3000


def test_latest_analysis_is_injected_as_long_term_memory():
    st = MemoryStore(":memory:")
    st.save_run(_run_row(1, score=8.1, verdict="逢低增配"))
    msgs = run(ChatMemory(st).build_context("s", "600519", "SYS", "怎么看"))
    assert "逢低增配" in msgs[0]["content"] and "8.1" in msgs[0]["content"]


def test_summary_falls_back_to_extractive_without_live_llm_and_never_uses_mock_text():
    assert llm_provider.status()["mode"] == "mock"
    out = extractive_summary([{"role": "user", "content": "茅台估值贵吗"}], "旧摘要")
    assert "旧摘要" in out and "茅台估值贵吗" in out
    assert "杜邦" not in out                                   # 不会混入模拟引擎的金融套话


def test_format_run_brief_contains_key_fields():
    b = format_run_brief(_run_row(1, score=7.5, verdict="持有"))
    assert "持有" in b and "7.5" in b and "600519" in b

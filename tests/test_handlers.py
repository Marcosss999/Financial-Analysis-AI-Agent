"""直接调用路由处理函数(不经过 HTTP 层)。在装有真实 FastAPI 的环境里同样可运行。"""
import json

from fastapi import HTTPException, Response

import app.api.financial_router as r
from app.core.resilience import IdempotencyCache
from app.memory.chat_memory import ChatMemory
from app.memory.store import MemoryStore
from app.models.schemas import AnalysisRequest, StockChatRequest
from app.services.analysis_service import AnalysisService
from tests.helpers import run


def _fresh():
    store = MemoryStore(":memory:")
    r.memory_store, r.analysis_service, r.chat_memory = store, AnalysisService(store), ChatMemory(store)
    return store


def test_sync_analyze_and_idempotent_replay_header():
    _fresh()
    resp = Response()
    first = run(r.run_sync_analysis(AnalysisRequest(ticker="600519"), resp, "k1"))
    assert first.run_id and first.data_quality["is_demo_data"] and "Idempotent-Replay" not in resp.headers
    resp2 = Response()
    second = run(r.run_sync_analysis(AnalysisRequest(ticker="600519"), resp2, "k1"))
    assert second.run_id == first.run_id and resp2.headers["Idempotent-Replay"] == "true"


def test_stream_handler_emits_sse_frames_ending_with_completed():
    _fresh()

    async def main():
        out = await r.run_stream_analysis(AnalysisRequest(ticker="300750"))
        return [c async for c in out.body_iterator]

    frames = run(main())
    events = [json.loads(f[len("data: "):]) for f in frames]
    assert events[0]["event"] == "started" and events[-1]["event"] == "completed"
    assert events[-1]["data"]["debate_history"]


def test_run_history_endpoints():
    _fresh()
    data = run(r.run_sync_analysis(AnalysisRequest(ticker="600519"), Response(), None))
    listing = run(r.list_runs("600519", 10))
    assert listing["runs"][0]["run_id"] == data.run_id
    full = run(r.get_run(data.run_id))
    assert full["trace"]["spans"]
    try:
        run(r.get_run("missing"))
    except HTTPException as e:
        assert e.status_code == 404
    else:
        raise AssertionError


def test_chat_keeps_server_side_session_memory():
    store = _fresh()
    run(r.run_sync_analysis(AnalysisRequest(ticker="600519"), Response(), None))
    a = run(r.chat_with_advisor_sync(StockChatRequest(ticker="600519", question="估值贵吗")))
    sid = a["session_id"]
    b = run(r.chat_with_advisor_sync(StockChatRequest(ticker="600519", question="那止损设多少", session_id=sid)))
    assert b["session_id"] == sid
    msgs = store.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_chat_imports_legacy_client_history_once():
    store = _fresh()
    hist = [{"role": "user", "content": "以前的问题"}, {"role": "assistant", "content": "以前的回答"}]
    out = run(r.chat_with_advisor_sync(StockChatRequest(ticker="600519", question="新问题", history=hist)))
    contents = [m["content"] for m in store.get_messages(out["session_id"])]
    assert contents[:2] == ["以前的问题", "以前的回答"] and "新问题" in contents


def test_llm_settings_never_return_plaintext_key():
    from app.core.config import settings
    old = settings.llm.API_KEY
    settings.llm.API_KEY = "sk-secret-1234567890"
    try:
        out = run(r.get_llm_settings())
    finally:
        settings.llm.API_KEY = old
    assert out.api_key_raw == "" and "secret" not in out.api_key_masked

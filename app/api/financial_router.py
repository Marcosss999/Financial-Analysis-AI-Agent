"""
金融分析 API 路由与 SSE 流式事件推送模块 (Financial API Router & SSE)
职责：
1. 提供同步分析端点 (/analyze)，接收标的代码并返回结构化研报与多空博弈产物。
2. 提供真流式 SSE 事件总线端点 (/analyze/stream)，逐节点实时推送 LangGraph 状态图执行进度与中间节点决策轨迹。
3. 提供标的即时行情与历史 K 线数据查询端点 (/quote/{ticker})，赋能前端动态图表绘制。
"""

import asyncio
import json
import uuid
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.core.container import analysis_service, chat_memory, memory_store
from app.core.llm_provider import llm_provider
from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    MarketQuote,
    LLMConfigReq,
    LLMConfigTestReq,
    LLMConfigResp,
    StockChatRequest,
)
from app.services.analysis_service import AnalysisTimeoutError
from app.tools.market_data import DataUnavailableError, market_data_provider

router = APIRouter(prefix="/finance", tags=["Financial Analysis"])


@router.get("/quote/{ticker}", response_model=MarketQuote)
async def get_stock_quote(ticker: str):
    """查询标的最新行情与历史 K 线"""
    try:
        quote = market_data_provider.get_market_quote(ticker.strip().upper(), days=60)
        return quote
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取行情失败: {str(e)}")


def _http_error(e: Exception) -> HTTPException:
    if isinstance(e, DataUnavailableError):
        return HTTPException(status_code=503, detail=str(e))
    if isinstance(e, AnalysisTimeoutError):
        return HTTPException(status_code=504, detail=str(e))
    return HTTPException(status_code=500, detail=f"分析失败: {e}")


@router.post("/analyze", response_model=AnalysisResponse)
async def run_sync_analysis(
    request: AnalysisRequest,
    response: Response,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """
    同步发起全流程多智能体投研分析。
    携带 Idempotency-Key 请求头时，相同 key + 相同参数的重复提交只执行一次(网络重试/用户连点不会重复烧 LLM)，
    命中缓存时响应头带 Idempotent-Replay: true。
    """
    try:
        if idempotency_key:
            data, replay = await analysis_service.run_idempotent(idempotency_key, request)
            if replay:
                response.headers["Idempotent-Replay"] = "true"
        else:
            data = await analysis_service.run(request)
    except Exception as e:  # noqa: BLE001
        raise _http_error(e)
    return AnalysisResponse(**{k: v for k, v in data.items() if k in AnalysisResponse.model_fields})


@router.post("/analyze/stream")
async def run_stream_analysis(request: AnalysisRequest):
    """
    SSE 流式端点：按节点粒度推送执行进度、辩论进展与最终结构化结果。
    事件：started(含 run_id / session_id) → node_update* → completed | error
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        async for ev in analysis_service.stream(request):
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# 三(补)、 记忆库与可观测性查询
# =============================================================================

@router.get("/runs")
async def list_runs(ticker: Optional[str] = None, limit: int = 20):
    """历史分析记录列表(长期记忆)"""
    return {"runs": await asyncio.to_thread(memory_store.list_runs, ticker.strip().upper() if ticker else None,
                                            max(1, min(limit, 100)))}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """某次分析的完整记录：结论、研报全文、数据质量，以及节点级 trace(耗时 / LLM 调用 / 降级)"""
    run = await asyncio.to_thread(memory_store.get_run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_id 不存在")
    return run


@router.get("/status")
async def runtime_status():
    """运行时状态：LLM 模式、熔断器状态、并行配置"""
    return {"llm": llm_provider.status(), "parallel_experts": settings.eng.PARALLEL_EXPERTS,
            "strict_real_data": settings.eng.STRICT_REAL_DATA}


# =============================================================================
# 四、 大模型 API Key 与网关热更新设置端点
# =============================================================================

@router.get("/settings/llm", response_model=LLMConfigResp)
async def get_llm_settings():
    """获取当前系统大模型配置与运行状态"""
    key = settings.llm.API_KEY or ""
    masked_key = ""
    if key and len(key) > 8:
        masked_key = key[:4] + "***" + key[-4:]
    elif key:
        masked_key = "***"

    return LLMConfigResp(
        is_configured=llm_provider.is_configured,
        api_key_masked=masked_key,
        api_key_raw="",  # 不再回传明文 Key (旧版本会把完整 Key 返回给任何能访问该接口的来源)
        base_url=settings.llm.BASE_URL,
        model=settings.llm.DEFAULT_MODEL,
        temperature=settings.llm.TEMPERATURE,
    )


@router.post("/settings/llm", response_model=LLMConfigResp)
async def save_llm_settings(req: LLMConfigReq):
    """保存并热更新大模型配置"""
    try:
        llm_provider.update_config(
            api_key=req.api_key,
            base_url=req.base_url,
            model=req.model,
            temperature=req.temperature,
        )
        return await get_llm_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存大模型配置失败: {str(e)}")


@router.post("/settings/llm/test")
async def test_llm_connection(req: LLMConfigTestReq):
    """测试指定或当前大模型配置的连通性"""
    try:
        result = await llm_provider.test_connection(
            api_key=req.api_key,
            base_url=req.base_url,
            model=req.model,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"测试连接失败: {str(e)}")


# =============================================================================
# 五、 AI 投顾深度多轮追问与答疑对话端点
# =============================================================================

@router.post("/chat/stream")
async def chat_with_advisor_stream(req: StockChatRequest):
    """与资深投顾针对当前标的与投研研报进行多轮追问对话 (SSE 流式推送)"""
    ticker = req.ticker.strip().upper()
    target_name = req.target_name or ticker

    # 尝试快速获取标的最新即时盘面数据，增强投顾对当下走势的敏锐度
    quote_info = ""
    try:
        q = await asyncio.to_thread(market_data_provider.get_market_quote, ticker, 5)
        if q:
            target_name = q.name or target_name
            quote_info = (
                f"【标的当前实时盘面数据参考】：\n"
                f"- 最新现价: ¥{q.current_price:.2f} 元 (当日涨跌幅: {q.change_pct:+.2f}%)\n"
                f"- 动态估值 PE(TTM): {q.pe_ttm:.1f}x | 市净率 PB: {q.pb_ratio:.2f}x\n"
                f"- 当日换手率: {q.turnover_rate:.2f}%\n\n"
            )
    except Exception as e:  # 行情只是增强上下文，失败不阻断对话，但要留日志(旧版本在这里把字段名错误静默吞掉了)
        import logging
        logging.getLogger(__name__).warning("chat quote enrich failed: %s", e)

    system_instruction = (
        f"你是一位拥有20年A股实操经验的华尔街兼公募基金资深投资顾问。用户正在针对标的【{target_name} ({ticker})】向你深度咨询。\n"
        f"{quote_info}"
        "【投顾核心回答规范与禁令】：\n"
        "1. 【严禁照本宣科】：绝对禁止机械复述多空辩论（Bull vs Bear）的原话或摘要套话！用户已经阅读过研报，不需要当复读机。你必须以专业资深投顾第一视角，用第一性原理正面、深度、犀利地解答用户提出的具体问题！\n"
        "2. 【直击痛点与资金本质】：\n"
        "   - 若用户询问【暴跌/大跌归因、大环境与板块影响】：必须多层次剖析（大盘系统性风险 Beta、板块情绪退潮 Alpha 踩踏、个股前期获利盘出逃、估值回调归因），切忌归为单一空泛原因。\n"
        "   - 若用户询问【龙虎榜与主力意图（如昨日净买今日暴跌）】：必须揭秘主力与游资博弈底层真相（例如游资做T高开砸盘、量化程序化对倒出货、席位诱多以及外围恐慌盘踩踏击穿接盘），用通俗大白话讲透资金意图。\n"
        "   - 若用户询问【买卖点/止损/仓位】：给出明确的防守支撑位、梯次建仓节奏（如金字塔三批次打底）与严格止损纪律。\n"
        "3. 【严禁输出任何 JSON 或代码结构】：绝对严禁直接输出字典、JSON、代码块或内部变量！必须全程输出优美、清晰、排版专业的人类自然语言 Markdown。\n"
        "4. 【通俗犀利、专业客观】：语言既生动接地气，又兼具机构级深度，不盲目打鸡血，也不无脑唱空吓唬人。\n"
    )
    session_id = req.session_id or uuid.uuid4().hex[:12]
    await chat_memory.import_history(session_id, ticker, req.history)  # 兼容旧前端自带的 history
    messages = await chat_memory.build_context(
        session_id, ticker, system_instruction, req.question.strip(), req.report_context
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        answer_parts = []
        try:
            yield f"data: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"
            async for chunk in llm_provider.chat_stream(messages):
                answer_parts.append(chunk)
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            await chat_memory.remember(session_id, ticker, req.question.strip(), "".join(answer_parts))
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/chat")
async def chat_with_advisor_sync(req: StockChatRequest):
    """同步单次追问接口 (用于非流式快捷调用)"""
    ticker = req.ticker.strip().upper()
    target_name = req.target_name or ticker

    quote_info = ""
    try:
        q = await asyncio.to_thread(market_data_provider.get_market_quote, ticker, 5)
        if q:
            target_name = q.name or target_name
            quote_info = (
                f"【标的实时盘面数据】：现价 ¥{q.current_price:.2f} 元 (涨跌幅: {q.change_pct:+.2f}%), "
                f"PE(TTM): {q.pe_ttm:.1f}x, 换手率: {q.turnover_rate:.2f}%\n"
            )
    except Exception as e:  # 行情只是增强上下文，失败不阻断对话，但要留日志(旧版本在这里把字段名错误静默吞掉了)
        import logging
        logging.getLogger(__name__).warning("chat quote enrich failed: %s", e)

    system_instruction = (
        f"你是一位拥有20年A股实操经验的华尔街兼公募基金资深投资顾问。用户正在针对标的【{target_name} ({ticker})】向你深度咨询。\n"
        f"{quote_info}"
        "【投顾核心原则】：禁止机械复述多空辩论原话套话；第一优先级正面解答用户具体问题（如暴跌归因、龙虎榜博弈、买卖点）；严禁输出内部 JSON；全程使用专业通俗的 Markdown 自然语言。\n"
    )
    session_id = req.session_id or uuid.uuid4().hex[:12]
    await chat_memory.import_history(session_id, ticker, req.history)  # 兼容旧前端自带的 history
    messages = await chat_memory.build_context(
        session_id, ticker, system_instruction, req.question.strip(), req.report_context
    )

    reply = await llm_provider.chat(messages)
    await chat_memory.remember(session_id, ticker, req.question.strip(), reply)
    return {
        "status": "success",
        "session_id": session_id,
        "ticker": ticker,
        "target_name": target_name,
        "question": req.question,
        "reply": reply,
    }


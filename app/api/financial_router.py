"""
金融分析 API 路由与 SSE 流式事件推送模块 (Financial API Router & SSE)
职责：
1. 提供同步分析端点 (/analyze)，接收标的代码并返回结构化研报与多空博弈产物。
2. 提供真流式 SSE 事件总线端点 (/analyze/stream)，逐节点实时推送 LangGraph 状态图执行进度与中间节点决策轨迹。
3. 提供标的即时行情与历史 K 线数据查询端点 (/quote/{ticker})，赋能前端动态图表绘制。
"""

import asyncio
import json
import time
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.core.llm_provider import llm_provider
from app.graph.financial_graph import financial_graph
from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    MarketQuote,
    LLMConfigReq,
    LLMConfigTestReq,
    LLMConfigResp,
    StockChatRequest,
)
from app.tools.market_data import market_data_provider

router = APIRouter(prefix="/finance", tags=["Financial Analysis"])


@router.get("/quote/{ticker}", response_model=MarketQuote)
async def get_stock_quote(ticker: str):
    """查询标的最新行情与历史 K 线"""
    try:
        quote = market_data_provider.get_market_quote(ticker.strip().upper(), days=60)
        return quote
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取行情失败: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
async def run_sync_analysis(request: AnalysisRequest):
    """同步发起全流程多智能体投研分析"""
    start_time = time.time()
    ticker = request.ticker.strip().upper()

    initial_state = {
        "ticker": ticker,
        "target_name": request.target_name,
        "custom_query": request.custom_query,
        "enable_debate": request.enable_debate,
        "strategy_style": request.strategy_style,
        "fundamental_weight": request.fundamental_weight,
        "technical_weight": request.technical_weight,
        "macro_weight": request.macro_weight,
        "capital_flow_weight": request.capital_flow_weight,
        "execution_logs": [],
        "debate_turns": [],
    }

    try:
        final_state = await financial_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多智能体工作流执行异常: {str(e)}")

    elapsed = round(time.time() - start_time, 2)
    arbiter = final_state.get("arbiter_verdict")
    audit = final_state.get("risk_audit")
    quote = final_state.get("quote")
    macro = final_state.get("macro_output")
    fund = final_state.get("fundamental_output")
    tech = final_state.get("technical_output")
    dcf = final_state.get("dcf_result")
    var_res = final_state.get("var_result")

    return AnalysisResponse(
        ticker=ticker,
        target_name=final_state.get("target_name", ticker),
        current_price=quote.current_price if quote else 0.0,
        consensus_score=arbiter.consensus_score if arbiter else 7.0,
        verdict=arbiter.final_verdict if arbiter else "中性",
        target_position=arbiter.target_position_pct if arbiter else "0%",
        risk_status=audit.compliance_status if audit else "APPROVED",
        is_vetoed=audit.is_risk_violated if audit else False,
        macro_score=macro.score if macro else 7.0,
        fundamental_score=fund.score if fund else 7.0,
        technical_score=tech.score if tech else 7.0,
        dcf_margin_of_safety=dcf.margin_of_safety_pct if dcf else 0.0,
        var_95_drawdown=var_res.var_percentage if var_res else 0.0,
        full_report_md=final_state.get("final_report_md", ""),
        debate_history=final_state.get("debate_turns", []),
        arbiter_verdict=arbiter,
        audit_details=audit,
        market_intel=final_state.get("market_intel"),
        strategy_style=final_state.get("strategy_style", "balanced"),
        weights_used={
            "fundamental": final_state.get("fundamental_weight", 0.40),
            "technical": final_state.get("technical_weight", 0.35),
            "macro": final_state.get("macro_weight", 0.15),
            "capital_flow": final_state.get("capital_flow_weight", 0.10),
        },
        execution_time_seconds=elapsed,
    )


@router.post("/analyze/stream")
async def run_stream_analysis(request: AnalysisRequest):
    """
    真流式 Server-Sent Events (SSE) 端点
    逐节点推送多智能体执行状态、辩论进展及最终研报
    """
    ticker = request.ticker.strip().upper()

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.time()
        initial_state = {
            "ticker": ticker,
            "target_name": request.target_name,
            "custom_query": request.custom_query,
            "enable_debate": request.enable_debate,
            "strategy_style": request.strategy_style,
            "fundamental_weight": request.fundamental_weight,
            "technical_weight": request.technical_weight,
            "macro_weight": request.macro_weight,
            "capital_flow_weight": request.capital_flow_weight,
            "execution_logs": [],
            "debate_turns": [],
        }

        # 1. 广播任务启动事件
        yield f"data: {json.dumps({'event': 'started', 'ticker': ticker, 'message': f'启动标的 {ticker} 投委会多Agent全流程分析...'}, ensure_ascii=False)}\n\n"

        accumulated_state = dict(initial_state)

        try:
            # 2. 监听 LangGraph 状态图节点异步流
            async for output_chunk in financial_graph.astream(initial_state):
                for node_name, node_update in output_chunk.items():
                    accumulated_state.update(node_update)

                    # 提取日志或节点简要信息
                    node_desc = {
                        "init_data_node": "底数据加载与K线提取",
                        "macro_node": "宏观经济与政策周期研判",
                        "fundamental_node": "基本面三表穿透与杜邦/DCF计算",
                        "technical_node": "量化技术形态与动量计算",
                        "bull_node": "多头首席辩手陈词",
                        "bear_node": "空头首席辩手反驳",
                        "arbiter_node": "投委会多空分歧仲裁",
                        "risk_node": "合规风控硬沙箱与一票否决审核",
                        "reporter_node": "机构级投研研报排版归档",
                    }.get(node_name, node_name)

                    payload = {
                        "event": "node_update",
                        "node": node_name,
                        "description": node_desc,
                        "step": node_update.get("current_step", ""),
                    }

                    # 如果是辩论节点，附带最新分数与论辩内容
                    if "bull_score" in node_update:
                        payload["bull_score"] = node_update["bull_score"]
                    if "bull_argument" in node_update:
                        payload["bull_argument"] = node_update["bull_argument"]
                    if "bear_score" in node_update:
                        payload["bear_score"] = node_update["bear_score"]
                    if "bear_argument" in node_update:
                        payload["bear_argument"] = node_update["bear_argument"]
                    if "current_debate_round" in node_update:
                        payload["debate_round"] = node_update["current_debate_round"]
                    if "arbiter_verdict" in node_update:
                        payload["verdict"] = node_update["arbiter_verdict"].final_verdict
                        payload["consensus_score"] = node_update["arbiter_verdict"].consensus_score

                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.04)

            # 3. 广播全部完成事件与结构化结果契约
            elapsed = round(time.time() - start_time, 2)
            arbiter = accumulated_state.get("arbiter_verdict")
            audit = accumulated_state.get("risk_audit")
            quote = accumulated_state.get("quote")
            macro = accumulated_state.get("macro_output")
            fund = accumulated_state.get("fundamental_output")
            tech = accumulated_state.get("technical_output")
            dcf = accumulated_state.get("dcf_result")
            var_res = accumulated_state.get("var_result")

            final_response = AnalysisResponse(
                ticker=ticker,
                target_name=accumulated_state.get("target_name", ticker),
                current_price=quote.current_price if quote else 0.0,
                consensus_score=arbiter.consensus_score if arbiter else 7.0,
                verdict=arbiter.final_verdict if arbiter else "中性",
                target_position=arbiter.target_position_pct if arbiter else "0%",
                risk_status=audit.compliance_status if audit else "APPROVED",
                is_vetoed=audit.is_risk_violated if audit else False,
                macro_score=macro.score if macro else 7.0,
                fundamental_score=fund.score if fund else 7.0,
                technical_score=tech.score if tech else 7.0,
                dcf_margin_of_safety=dcf.margin_of_safety_pct if dcf else 0.0,
                var_95_drawdown=var_res.var_percentage if var_res else 0.0,
                full_report_md=accumulated_state.get("final_report_md", ""),
                debate_history=accumulated_state.get("debate_turns", []),
                arbiter_verdict=arbiter,
                audit_details=audit,
                market_intel=accumulated_state.get("market_intel"),
                strategy_style=accumulated_state.get("strategy_style", "balanced"),
                weights_used={
                    "fundamental": accumulated_state.get("fundamental_weight", 0.40),
                    "technical": accumulated_state.get("technical_weight", 0.35),
                    "macro": accumulated_state.get("macro_weight", 0.15),
                    "capital_flow": accumulated_state.get("capital_flow_weight", 0.10),
                },
                execution_time_seconds=elapsed,
            )

            data_dict = final_response.model_dump()
            dupont = accumulated_state.get("dupont_result")
            indicators = accumulated_state.get("quant_indicators")
            if dupont:
                data_dict["dupont_roe"] = dupont.calculated_roe
                data_dict["net_margin"] = dupont.net_profit_margin
            if dcf:
                data_dict["dcf_value"] = dcf.intrinsic_value_per_share
            if indicators:
                data_dict["rsi_14"] = indicators.rsi_14
                data_dict["macd_bar"] = indicators.macd_bar
                data_dict["trend_signal"] = indicators.trend_signal
                data_dict["support_level"] = indicators.support_level
                data_dict["resistance_level"] = indicators.resistance_level
            if var_res:
                data_dict["var_absolute_loss"] = var_res.var_absolute_loss

            complete_payload = {
                "event": "completed",
                "elapsed": elapsed,
                "data": data_dict,
            }
            yield f"data: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"

        except Exception as e:
            err_payload = {"event": "error", "message": f"流式处理失败: {str(e)}"}
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
        api_key_raw=key,
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
        q = market_data_provider.get_market_quote(ticker, days=5)
        if q:
            target_name = q.name or target_name
            quote_info = (
                f"【标的当前实时盘面数据参考】：\n"
                f"- 最新现价: ¥{q.current_price:.2f} 元 (当日涨跌幅: {q.change_pct:+.2f}%)\n"
                f"- 动态估值 PE(TTM): {q.pe_ttm:.1f}x | 市净率 PB: {q.pb:.2f}x\n"
                f"- 当日换手率: {q.turnover_rate:.2f}% | 成交量: {q.volume/10000:.1f}万手\n\n"
            )
    except Exception:
        pass

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
    if req.report_context:
        trimmed_context = req.report_context[:2000]
        system_instruction += f"\n【背景参考研报（注意：仅作为分析依据，请勿照抄以下内容）】：\n{trimmed_context}\n"

    messages = [{"role": "system", "content": system_instruction}]
    for h in req.history[-6:]:
        messages.append({
            "role": h.get("role", "user"),
            "content": h.get("content", "")
        })
    messages.append({"role": "user", "content": req.question.strip()})

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in llm_provider.chat_stream(messages):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
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
        q = market_data_provider.get_market_quote(ticker, days=5)
        if q:
            target_name = q.name or target_name
            quote_info = (
                f"【标的实时盘面数据】：现价 ¥{q.current_price:.2f} 元 (涨跌幅: {q.change_pct:+.2f}%), "
                f"PE(TTM): {q.pe_ttm:.1f}x, 换手率: {q.turnover_rate:.2f}%\n"
            )
    except Exception:
        pass

    system_instruction = (
        f"你是一位拥有20年A股实操经验的华尔街兼公募基金资深投资顾问。用户正在针对标的【{target_name} ({ticker})】向你深度咨询。\n"
        f"{quote_info}"
        "【投顾核心原则】：禁止机械复述多空辩论原话套话；第一优先级正面解答用户具体问题（如暴跌归因、龙虎榜博弈、买卖点）；严禁输出内部 JSON；全程使用专业通俗的 Markdown 自然语言。\n"
    )
    if req.report_context:
        trimmed_context = req.report_context[:2000]
        system_instruction += f"\n【参考背景研报（切勿照抄）】：\n{trimmed_context}\n"

    messages = [{"role": "system", "content": system_instruction}]
    for h in req.history[-6:]:
        messages.append({
            "role": h.get("role", "user"),
            "content": h.get("content", "")
        })
    messages.append({"role": "user", "content": req.question.strip()})

    reply = await llm_provider.chat(messages)
    return {
        "status": "success",
        "ticker": ticker,
        "target_name": target_name,
        "question": req.question,
        "reply": reply,
    }


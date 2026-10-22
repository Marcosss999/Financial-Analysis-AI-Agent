"""
大模型统一服务驱动与金融离线模拟引擎 (LLM Provider Service)
职责：
1. 封装统一的 OpenAI 兼容协议异步客户端 (AsyncOpenAI)，支持 Qwen / DeepSeek / GPT 等模型。
2. 原生支持流式输出 (chat_stream) 与非流式结构化输出 (chat)。
3. 内置高保真领域金融 Mock 引擎：在离线或无 API Key 场景下，按金融专家角色（宏观、基本面、技术面、多空辩论、风控仲裁、研报生成）输出严谨的专业金融研报与结构化 JSON 数据，保证系统无依赖端到端跑通。
"""

import asyncio
import json
import logging
import re
import time
from contextvars import ContextVar
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI
from app.core.config import settings, save_env_llm_settings
from app.core.observability import get_trace
from app.core.resilience import CircuitBreaker, retry_async

logger = logging.getLogger(__name__)

# 最近一次 chat() 是否来自真实远程模型（按 asyncio 任务隔离，并行专家互不干扰）
_LAST_CALL_LIVE: ContextVar[bool] = ContextVar("llm_last_call_live", default=False)


def last_call_was_live() -> bool:
    return _LAST_CALL_LIVE.get()


class LLMUnavailableError(RuntimeError):
    """调用方声明不接受模拟响应(allow_mock=False)，而远程模型不可用时抛出"""


class LLMProvider:
    """企业级金融统一大模型服务提供者"""

    def __init__(self):
        self.is_configured = bool(
            settings.llm.API_KEY
            and "your_api_key" not in settings.llm.API_KEY.lower()
            and len(settings.llm.API_KEY.strip()) > 5
        )

        self.breaker = CircuitBreaker(
            failure_threshold=settings.llm.BREAKER_FAILURE_THRESHOLD,
            recovery_seconds=settings.llm.BREAKER_RECOVERY_SECONDS,
        )

        if self.is_configured:
            self.client: Optional[AsyncOpenAI] = AsyncOpenAI(
                api_key=settings.llm.API_KEY,
                base_url=settings.llm.BASE_URL,
                timeout=settings.llm.REQUEST_TIMEOUT,
                max_retries=0,  # 重试统一由应用层 retry_async 负责，避免 SDK 重试与应用层重试相乘

            )
        else:
            self.client = None

    def update_config(self, api_key: str, base_url: str, model: str, temperature: float = 0.2):
        """动态热更新大模型客户端配置，并持久化到 .env"""
        save_env_llm_settings(api_key=api_key, base_url=base_url, model=model, temperature=temperature)

        settings.llm.API_KEY = api_key.strip()
        settings.llm.BASE_URL = base_url.strip()
        settings.llm.DEFAULT_MODEL = model.strip()
        settings.llm.TEMPERATURE = float(temperature)

        self.is_configured = bool(
            settings.llm.API_KEY
            and "your_api_key" not in settings.llm.API_KEY.lower()
            and len(settings.llm.API_KEY.strip()) > 5
        )

        if self.is_configured:
            self.client = AsyncOpenAI(
                api_key=settings.llm.API_KEY,
                base_url=settings.llm.BASE_URL,
                timeout=settings.llm.REQUEST_TIMEOUT,
                max_retries=0,  # 重试统一由应用层 retry_async 负责，避免 SDK 重试与应用层重试相乘

            )
        else:
            self.client = None

        logger.info(f"[LLMProvider] 配置已热更新: Model={settings.llm.DEFAULT_MODEL}, BaseUrl={settings.llm.BASE_URL}, Configured={self.is_configured}")

    async def test_connection(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """连通性与可用性极速测试"""
        target_key = (api_key if api_key is not None else settings.llm.API_KEY).strip()
        target_url = (base_url if base_url is not None else settings.llm.BASE_URL).strip()
        target_model = (model if model is not None else settings.llm.DEFAULT_MODEL).strip()

        if not target_key or len(target_key) < 5 or "your_api_key" in target_key.lower():
            return {
                "success": False,
                "message": "API Key 格式无效或为空，请先输入有效的 API Key！",
                "latency_ms": 0,
            }

        start_t = time.time()
        try:
            test_client = AsyncOpenAI(api_key=target_key, base_url=target_url, timeout=12)
            resp = await test_client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": "请只回复两个汉字：正常"}],
                max_tokens=10,
            )
            elapsed = round((time.time() - start_t) * 1000)
            reply = resp.choices[0].message.content or ""
            return {
                "success": True,
                "message": f"连接成功！延迟 {elapsed}ms，模型响应: \"{reply.strip()}\"",
                "model": target_model,
                "latency_ms": elapsed,
            }
        except Exception as e:
            elapsed = round((time.time() - start_t) * 1000)
            return {
                "success": False,
                "message": f"连接测试失败 ({elapsed}ms): {str(e)}",
                "model": target_model,
                "latency_ms": elapsed,
            }

    def status(self) -> Dict[str, Any]:
        """当前 LLM 运行模式与熔断器状态(用于 /health 与 trace)"""
        return {
            "mode": "live" if self.is_configured and self.client is not None else "mock",
            "model": settings.llm.DEFAULT_MODEL,
            "breaker": self.breaker.snapshot(),
        }

    def _record(self, started: float, retries: int, fallback: bool, failed: bool,
                prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        trace = get_trace()
        if trace is not None:
            trace.record_llm(latency_ms=(time.perf_counter() - started) * 1000, retries=retries,
                             fallback=fallback, failed=failed,
                             prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        allow_mock: bool = True,
    ) -> str:
        """
        统一异步非流式对话调用，带：单次超时 + 指数退避重试 + 熔断 + 降级留痕。
        降级(无 Key / 熔断打开 / 重试耗尽)时返回本地模拟响应，同时在当前 trace 里记录 fallback，
        因此一份报告里有多少内容来自模拟引擎是可以被统计和展示的，而不是静默混入。
        """
        started = time.perf_counter()
        _LAST_CALL_LIVE.set(False)
        if not self.is_configured or self.client is None:
            self._record(started, 0, fallback=True, failed=False)
            if not allow_mock:
                raise LLMUnavailableError("未配置 LLM，且调用方不接受模拟响应")
            return self._mock_financial_response(messages)

        if not self.breaker.allow():
            logger.warning("[LLMProvider] 熔断器打开，跳过远程调用并直接降级。")
            self._record(started, 0, fallback=True, failed=True)
            if not allow_mock:
                raise LLMUnavailableError("熔断器打开，且调用方不接受模拟响应")
            return self._mock_financial_response(messages)

        extra_kwargs: Dict[str, Any] = {}
        if json_mode:
            extra_kwargs["response_format"] = {"type": "json_object"}

        usage = {"p": 0, "c": 0}

        async def _call() -> str:
            response = await self.client.chat.completions.create(
                model=model or settings.llm.DEFAULT_MODEL,
                messages=messages,
                temperature=temperature if temperature is not None else settings.llm.TEMPERATURE,
                **extra_kwargs,
            )
            text = response.choices[0].message.content or ""
            if not text.strip():
                raise ValueError("模型返回空内容")  # 空响应视为失败，触发重试
            u = getattr(response, "usage", None)
            if u is not None:
                usage["p"] = getattr(u, "prompt_tokens", 0) or 0
                usage["c"] = getattr(u, "completion_tokens", 0) or 0
            return text

        try:
            text, retries = await retry_async(
                _call,
                retries=settings.llm.MAX_RETRIES,
                base_delay=settings.llm.RETRY_BASE_DELAY,
                timeout=settings.llm.REQUEST_TIMEOUT,
            )
            self.breaker.record_success()
            _LAST_CALL_LIVE.set(True)
            self._record(started, retries, fallback=False, failed=False,
                         prompt_tokens=usage["p"], completion_tokens=usage["c"])
            return text
        except (Exception, asyncio.CancelledError) as e:  # 含 CancelledError，保证半开探测位一定被释放
            self.breaker.record_failure()
            if isinstance(e, asyncio.CancelledError):
                raise
            logger.warning(f"[LLMProvider] 远程模型调用在重试后仍失败 ({type(e).__name__}: {e})，降级至本地模拟引擎。")
            self._record(started, settings.llm.MAX_RETRIES, fallback=True, failed=True)
            if not allow_mock:
                raise LLMUnavailableError(f"远程模型不可用: {e}") from e
            return self._mock_financial_response(messages)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        异步流式调用生成器 (用于 SSE 逐字实时推送到前端)
        具备连接故障自动重试与中断平滑保护机制。
        """
        if not self.is_configured or self.client is None:
            mock_text = self._mock_financial_response(messages)
            chunk_size = 8
            for i in range(0, len(mock_text), chunk_size):
                yield mock_text[i: i + chunk_size]
            return

        has_yielded = False
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                stream = await self.client.chat.completions.create(
                    model=model or settings.llm.DEFAULT_MODEL,
                    messages=messages,
                    temperature=temperature if temperature is not None else settings.llm.TEMPERATURE,
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        has_yielded = True
                        yield chunk.choices[0].delta.content
                # 正常流式传输结束
                return
            except Exception as e:
                logger.warning(
                    f"[LLMProvider] 流式调用异常 (attempt={attempt+1}/{max_attempts}, has_yielded={has_yielded}): {e}"
                )
                if has_yielded:
                    # 已经给前端输出过部分真实内容，绝对不能拼接 mock，否则会导致文字前后断裂或输出内部 JSON 结构
                    yield f"\n\n> ⚠️ *[上游大模型连接因网络波动中断。建议稍后重试或查看上方研报结论]*"
                    return
                if attempt == 0:
                    # 首包前连接异常（如 DeepSeek 排队超时或握手断开），等待 0.8s 后自动重试一次
                    await asyncio.sleep(0.8)
                    continue
                # 重试仍失败且完全未吐字，降级为高质量模拟响应
                break

        mock_text = self._mock_financial_response(messages)
        chunk_size = 8
        for i in range(0, len(mock_text), chunk_size):
            yield mock_text[i: i + chunk_size]

    def _mock_advisor_qa_response(
        self,
        target_name: str,
        ticker: str,
        user_question: str,
        roe_val: float,
        pe_val: float,
        debt_val: float,
        rsi_val: float,
        margin_val: float,
        consensus_score: float,
    ) -> str:
        """
        资深投资顾问多轮对话高保真响应器：
        绝对输出自然语言 Markdown，严禁泄露内部 JSON 数据结构，
        针对暴跌归因、龙虎榜席位玄机、大盘板块分化及买卖止损策略进行深度剖析。
        """
        q = user_question.strip()
        q_lower = q.lower()

        # 场景 1: 暴跌 / 大跌 / 龙虎榜 / 价值超支 / 回调 / 踩踏 / 大环境与板块影响
        if any(k in q_lower for k in ["跌", "暴跌", "大跌", "回调", "超支", "龙虎榜", "大环境", "板块", "为何依然", "为什么依然"]):
            return f"""针对你关心的【{target_name} ({ticker}) 近期大跌归因与龙虎榜疑虑】，我结合 20 年 A 股公募与自营实盘经验，为你做一次穿透式的资金与逻辑拆解：

### 一、 这波大跌到底是谁的锅？（多维度深度归因）
1. **大环境与流动性分歧（Beta 风险）**：
   近期大盘指数处于震荡分化格局，市场整体风险偏好明显收缩。在增量资金不足的存量博弈行情中，资金天生具有防御避险倾向。
2. **板块主线退潮与高低切换（Alpha 踩踏）**：
   {target_name} 所属板块在前期积累了一定的涨幅，近期出现明显的“多杀多”资金抢跑现象。板块情绪一旦见顶，无论个股质地如何，短期都会遭受流动性无差别抛售。
3. **估值透支与预期差修正**：
   从基本面数据看，当前 PE(TTM) 约为 {pe_val:.1f} 倍。如果此前市场对其赋予了过高预期，短期稍微有一丝业绩增速放缓或情绪转弱，都会触发剧烈的“杀估值”均值回归。

---

### 二、 核心疑惑：为什么昨日龙虎榜还是净买入，今天依然暴跌？
这在 A 股实盘中是非常经典的**主力资金博弈陷阱**，通常有以下三大底层玄机：
1. **“假买真出”与席位对倒出货**：
   龙虎榜只公布买卖前五大营业部。知名游资完全可以通过上榜席位高调买入 3000 万元（制造“机构大佬抄底”的狂热舆论），但同时在未上榜的数十个普通席位或利用融券底仓暗中砸出 8000 万元，实现**表面净买入、实际大额净套现**。
2. **短线游资隔日反手砸盘（做 T 获利了结）**：
   游资的持仓周期极短（往往仅有 1~2 天）。如果昨日尾盘或盘中拉高建仓后，次日早盘发现跟风接盘的散户资金不足、或者大盘环境转冷，游资会毫不犹豫地选择“割肉砸盘”或“快速止盈跑路”，踩踏式抛盘直接造成今日深跌。
3. **恐慌情绪踩踏击穿主力承接底线**：
   即使昨日进场的某些席位是真心看好的“真机构”，但当全市场恐慌盘集中涌出时，机构单日的买入资金犹如杯水车薪，承接盘瞬间被击穿，从而引发雪崩式补跌。

---

### 三、 散户实操应对锦囊与纪律红线
* **切忌盲目“左侧接飞刀”**：在放量大跌、尚未出现明确“缩量十字星”或“止跌双底”之前，绝不要急于满仓抄底。
* **分批建仓与止损纪律**：
  - 如果对 {target_name} 的长期逻辑（ROE {roe_val:.1f}%、DCF 安全边际 {margin_val:.1f}%）仍有信心，建议采用**三步金字塔挂单法**（首笔试仓 20%~30%）。
  - **关键防守线**：若后续跌破下方核心均线支撑且 3 个交易日无法收复，必须坚决执行止损纪律，不可心存侥幸！
"""

        # 场景 2: 资金流向 / 洗盘 / 出货 / 诱多
        if any(k in q_lower for k in ["资金", "主力", "洗盘", "出货", "诱多", "筹码"]):
            return f"""关于【{target_name} ({ticker}) 的主力资金意图研判】：

1. **主力洗盘 vs 诱多出货的核心区别**：
   - **典型洗盘特征**：下跌时成交量显著萎缩（无量空跌），分时图急跌但随后慢速推升收复，核心均线支撑未破，这表明场内主力并未真正离场，意在震荡恐吓浮筹。
   - **典型出货特征**：高位反复脉冲放量却滞涨，分时图早盘拉高诱多、午后一路阴跌走弱，且换手率异常放大，大单资金持续呈净流出状态。
2. **结合当前形态判断**：
   {target_name} 当前 RSI(14) 约为 {rsi_val:.1f}，均线处于整固通道。近期盘面若伴随大单分歧，短期更偏向于**震荡换手清洗浮筹**。
3. **实操建议**：
   切勿在盘中急拉时追高，建议观察尾盘 14:30 后的量价承接力，若量能企稳可逢低关注。
"""

        # 场景 3: 支撑位 / 阻力位 / 止损 / 防守
        if any(k in q_lower for k in ["支撑", "阻力", "止损", "防守", "破位"]):
            return f"""关于【{target_name} ({ticker}) 的核心攻防点位与止损红线】：

1. **关键技术防守位**：
   - **第一支撑位**：以 20 日均线 (MA20) 及前期密集成交平台为核心防守线。
   - **强支撑/生命线**：以 60 日均线 (MA60) 作为波段趋势分水岭。
2. **动态止损纪律**：
   - **短线交易**：设置 **5% ~ 7%** 动态追踪止损线，一旦跌破必须坚决离场，保住本金。
   - **波段配置**：若收盘价连续 2 个交易日收在核心均线下方且反抽无力，触发防御性减仓 50%。
3. **投顾心得**：
   在 A 股做交易，“截断亏损，让利润奔跑”永远是第一法则，绝不可将短线交易被动做成“长期股东”！
"""

        # 场景 4: 建仓 / 买入 / 仓位 / 抄底
        if any(k in q_lower for k in ["建仓", "买入", "仓位", "抄底", "怎么买", "加仓"]):
            return f"""关于【{target_name} ({ticker}) 的分批建仓与仓位管理攻略】：

1. **总仓位上限控制**：
   根据投委会综合评分 ({consensus_score:.1f} 分) 与风控模型，建议当前该标的在整体组合中的持仓上限控制在 **40% ~ 55%** 之间，切忌单一标的满仓重仓。
2. **金字塔分批建仓法则（分 3 批次介入）**：
   - **第一批（底仓 20%~30%）**：在股价回调至关键支撑位且出现企稳迹象时首次轻仓试水，建立观察仓位；
   - **第二批（确认仓 30%）**：当股价带量突破短期阻力位、均线形成向上拐头时顺势加仓；
   - **第三批（机动防守 40%）**：留作现金储备或右侧主升浪确认后的追加资金。
3. **心理预期管理**：
   当前估值安全垫约为 {margin_val:.1f}%，耐心等待市场给出舒服的击球点，右侧交易往往比盲目抄底更具确定性。
"""

        # 场景 5: 默认通用资深投顾解答
        return f"""你好！关于【{target_name} ({ticker})】的投资逻辑，我从资深买方投资顾问视角为你梳理核心要点：

1. **商业基本盘与护城河**：
   公司当前净资产收益率 ROE 约为 {roe_val:.1f}%，资产负债率维持在 {debt_val:.1f}% 的健康区间，内生盈利能力具备较强的抗周期韧性。
2. **估值与量化博弈**：
   目前 PE(TTM) 为 {pe_val:.1f} 倍，DCF 模型测算具备约 {margin_val:.1f}% 的估值安全边际。量化 RSI 指标处于 {rsi_val:.1f}，短期多空博弈处于再平衡阶段。
3. **核心操作建议**：
   不盲目追高，也不必恐慌杀跌。建议关注 20 日均线企稳状态，以分批梯次吸筹的方式控制持仓成本，设置好严格的止损防线。
"""

    def _mock_financial_response(self, messages: List[Dict[str, str]]) -> str:
        """
        高保真领域金融模拟响应器：
        优先根据 system 角色明确匹配 Agent 任务，防止上下文中的引用词导致角色误判，
        并结合不同标的输入动态调整多空博弈得分。
        """
        import re

        system_content = next((m.get("content", "") for m in messages if m.get("role") == "system"), "").lower()
        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
        latest_question = user_messages[-1] if user_messages else ""
        user_content = " ".join(user_messages)
        user_content_lower = user_content.lower()
        # 旧版本这里漏定义 full_prompt：任何没命中前面分支的提示词(例如没有 system 消息的普通调用)都会在第 7 个分支抛 NameError
        full_prompt = f"{system_content} {user_content_lower}"

        # 动态提取标的名与财务指标
        target_name = "标的资产"
        ticker = ""
        # 优先从 system 的规范格式提取: 标的【名称 (代码)】
        t_sys_match = re.search(r"【([^】]+)\s*\(([A-Za-z0-9.]+)\)】", system_content)
        if t_sys_match:
            target_name = t_sys_match.group(1).strip()
            ticker = t_sys_match.group(2).strip()
        else:
            t_match = re.search(r"标的(?:代码)?:\s*([A-Za-z0-9.]+)?\s*(?:\(([^)]+)\))?", user_content)
            if t_match:
                target_name = t_match.group(2) or t_match.group(1) or "标的资产"
                ticker = t_match.group(1) or ""

        roe_match = re.search(r"roe[^\d%]{0,10}([0-9.]+)%", user_content, re.IGNORECASE)
        roe_val = float(roe_match.group(1)) if roe_match else 16.0

        debt_match = re.search(r"资产负债率[^\d%]{0,10}([0-9.]+)%", user_content, re.IGNORECASE)
        debt_val = float(debt_match.group(1)) if debt_match else 35.0

        pe_match = re.search(r"pe(?:\(ttm\))?[^\d]{0,10}([0-9.]+)x?", user_content, re.IGNORECASE)
        pe_val = float(pe_match.group(1)) if pe_match else 22.0

        rsi_match = re.search(r"rsi(?:\(14\))?[^\d]{0,10}([0-9.]+)", user_content, re.IGNORECASE)
        rsi_val = float(rsi_match.group(1)) if rsi_match else 52.0

        margin_match = re.search(r"(?:安全边际|安全垫|dcf)[^\d%]{0,15}([0-9.-]+)%", user_content, re.IGNORECASE) or re.search(r"([0-9.-]+)%\s*(?:安全边际|安全垫)", user_content, re.IGNORECASE)
        margin_val = float(margin_match.group(1)) if margin_match else 12.0

        # 动态根据标的财务质地与量化走势测算多空博弈置信度
        ref_bull_match = re.search(r"参考多头量化基准分:\s*([0-9.]+)", user_content)
        if ref_bull_match:
            calc_bull_score = float(ref_bull_match.group(1))
        else:
            calc_bull_score = round(min(9.4, max(5.2, 5.6 + (roe_val / 25.0) * 2.2 + (rsi_val / 100.0) * 1.4)), 1)

        ref_bear_match = re.search(r"参考空头量化风险分:\s*([0-9.]+)", user_content)
        if ref_bear_match:
            calc_bear_score = float(ref_bear_match.group(1))
        else:
            base_risk = 4.2
            if debt_val > 65.0:
                base_risk += 2.0
            elif debt_val > 50.0:
                base_risk += 1.0
            if pe_val > 45.0:
                base_risk += 1.6
            elif pe_val > 25.0:
                base_risk += 0.8
            calc_bear_score = round(min(9.0, max(4.2, base_risk)), 1)

        ref_arbiter_match = re.search(r"参考综合共识分:\s*([0-9.]+)", user_content)
        if ref_arbiter_match:
            consensus_score = float(ref_arbiter_match.group(1))
        else:
            consensus_score = round(calc_bull_score * 0.65 + (10.0 - calc_bear_score) * 0.35, 1)

        # 0. 优先且绝对匹配：资深投资顾问多轮追问 (AI Investment Advisor Q&A)
        # 无论输入何种文本或风控字样，对话端点必须返回自然语言 Markdown，绝不能泄露内部 JSON！
        if (
            "投资顾问" in system_content
            or "资深投顾" in system_content
            or "advisor" in system_content
            or "深度追问" in system_content
            or "深度咨询" in system_content
        ):
            return self._mock_advisor_qa_response(
                target_name=target_name,
                ticker=ticker or "标的",
                user_question=latest_question,
                roe_val=roe_val,
                pe_val=pe_val,
                debt_val=debt_val,
                rsi_val=rsi_val,
                margin_val=margin_val,
                consensus_score=consensus_score,
            )

        # 1. 优先匹配：多头首席辩手 (Bull Advocate)
        if "bull advocate" in system_content or "多头首席辩手" in system_content:
            return json.dumps({
                "speaker": "Bull_Advocate",
                "conviction_score": calc_bull_score,
                "core_thesis": f"{target_name} 具备坚实的商业护城河与内生现金流驱动力，当前估值安全边际明确，多头共振蓄势突破。",
                "catalysts": [
                    f"{target_name} 核心主营产品市占率高且具备产业链强势定价权，抗周期韧性强",
                    f"ROE 维持在 {roe_val:.1f}% 景气区间，在手订单充沛，未来业绩存在超预期重估概率",
                    f"DCF 绝对估值模型测算安全边际达 {margin_val:.1f}%，现价折价防御垫充足"
                ]
            }, ensure_ascii=False, indent=2)

        # 2. 优先匹配：空头首席辩手 (Bear Critic)
        if "bear critic" in system_content or "空头首席辩手" in system_content:
            return json.dumps({
                "speaker": "Bear_Critic",
                "conviction_score": calc_bear_score,
                "core_thesis": f"{target_name} 估值需审慎防守看待，市场预期过于饱满，需警惕原材料成本波动与资产流动性带来的下行风险。",
                "vulnerabilities": [
                    f"{target_name} 当前资产负债率为 {debt_val:.1f}%，需密切跟踪流动资产周转天数与偿债付息压力",
                    f"当前估值 PE(TTM) 为 {pe_val:.1f} 倍，若宏观周期走弱或行业内卷降价，估值中枢面临回撤压力",
                    "历史波动率在季报披露日前后显著放大，右侧追高盈亏比相对平庸"
                ]
            }, ensure_ascii=False, indent=2)

        # 3. 优先匹配：投委会仲裁官 (Arbiter)
        if "arbiter" in system_content or "投委会主席" in system_content or "仲裁" in system_content:
            verdict = "积极增配 (Strong Buy)" if consensus_score >= 8.2 else ("逢低增配 (Buy on Dips)" if consensus_score >= 7.0 else ("中性配置 (Hold)" if consensus_score >= 6.0 else "审慎观望 (Underweight)"))
            pos_pct = "60% ~ 75%" if consensus_score >= 8.2 else ("45% ~ 55%" if consensus_score >= 7.0 else ("25% ~ 35%" if consensus_score >= 6.0 else "10% ~ 20%"))
            return json.dumps({
                "final_verdict": verdict,
                "consensus_score": consensus_score,
                "target_position_pct": pos_pct,
                "arbitration_summary": f"投委会充分评估了 {target_name} 的多空博弈论据：采纳多头关于高商业壁垒与内生现金流护城河的主干逻辑；同时采纳空头审慎防守警示，在建仓节奏上保持克制，严格执行仓位约束。",
                "entry_tactics": "建议分 3 批在回调至技术均线支撑位时梯次吸筹，设置 7% 动态追踪止损线。"
            }, ensure_ascii=False, indent=2)

        # 4. 合规风控 Agent (Risk Guardrail)
        if "risk_guardrail" in system_content or "合规官" in system_content or "风控" in system_content:
            is_violated = debt_val > 70.0
            return json.dumps({
                "agent_name": "Risk_Guardrail_Officer",
                "compliance_status": "REJECTED" if is_violated else "APPROVED",
                "is_risk_violated": is_violated,
                "audit_checklist": {
                    "debt_to_asset_check": f"{'未通过' if is_violated else '通过'} (资产负债率 {debt_val:.1f}%)",
                    "interest_coverage_check": "通过 (利息保障倍数充沛)",
                    "var_95_check": "通过 (单周最大回撤 VaR 处于可控阈值内)",
                    "regulatory_penalty_check": "通过 (无重大立案调查或违规记录)"
                },
                "risk_rating": "中高风险" if is_violated else "中低风险 (R2 稳健型)",
                "hard_constraints_memo": f"{'触碰高负债风控红线，触发一票否决' if is_violated else '准予出具增持评级，强制附注周期波动与仓位约束提示'}。"
            }, ensure_ascii=False, indent=2)

        # 5. 宏观分析 Agent 判定
        if "macro" in system_content or "宏观经济" in system_content:
            return json.dumps({
                "stance": "中性偏多 (Mildly Bullish)",
                "score": 7.6,
                "summary": "宏观流动性维持适度充裕，降准降息预期窗口提供估值分母端支撑，产业政策定向扶持战略性核心产业。",
                "details": {
                    "economic_cycle": "温和复苏早期至扩张期",
                    "policy_impact": "货币与财政政策保持精准支持，利好核心资产分母端估值修复",
                    "interest_rate_outlook": "长端国债收益率低位盘整，贴现率利好成长与红利估值",
                    "industry_tailwinds": ["新质生产力定向贴息支持", "设备更新与高端转型专项扶持"]
                },
                "key_points": [
                    "宏观环境流动性合理充裕，无系统性信用紧缩风险",
                    "关注海外贸易外需波动与美元汇率扰动"
                ]
            }, ensure_ascii=False, indent=2)

        # 6. 基本面分析 Agent 判定
        if "fundamental" in system_content or "基本面" in system_content:
            calc_fund_score = round(min(9.5, max(4.8, 6.0 + (roe_val / 25.0) * 2.2 + (margin_val / 30.0) * 1.0)), 1)
            return json.dumps({
                "stance": "积极买入 (Strong Buy)" if calc_fund_score >= 8.0 else ("逢低增配 (Hold)" if calc_fund_score >= 6.5 else "谨慎观望"),
                "score": calc_fund_score,
                "summary": f"{target_name} 杜邦拆解表明公司 ROE 达 {roe_val:.1f}%，盈利成色良好，DCF 自由现金流估值提供 {margin_val:.1f}% 安全垫。",
                "details": {
                    "roe_assessment": f"ROE 达 {roe_val:.1f}%，由内生高质量经营性现金流驱动",
                    "dcf_margin_of_safety": f"DCF 模型测算安全边际为 {margin_val:.1f}%",
                    "balance_sheet_health": f"资产负债率为 {debt_val:.1f}%，债务结构总体稳健",
                    "growth_sustainability": f"{target_name} 具备良好行业竞争壁垒与产品定价权"
                },
                "key_points": [
                    f"充沛经营现金流保障净利润成色 (ROE: {roe_val:.1f}%)",
                    f"资产负债率为 {debt_val:.1f}%，偿债结构健康",
                    "需跟踪下游宏观复苏斜率对动销周期的影响"
                ]
            }, ensure_ascii=False, indent=2)

        # 7. 量化技术面 Agent 判定
        if "technical" in system_content or "技术面" in system_content:
            calc_tech_score = round(min(9.2, max(5.0, 5.2 + (rsi_val / 100.0) * 4.0)), 1)
            return json.dumps({
                "stance": "多头蓄势 (Bullish Accumulation)" if rsi_val >= 50 else "震荡整固 (Consolidation)",
                "score": calc_tech_score,
                "summary": f"{target_name} RSI 强弱度为 {rsi_val:.1f}，均线走势平稳，下行具备技术防御支撑。",
                "details": {
                    "trend_evaluation": f"{target_name} 关键均线企稳，未见明显破位迹象",
                    "momentum_analysis": f"RSI(14) 读数为 {rsi_val:.1f}，动能处于健康区间",
                    "volatility_channel": "布林带通道温和运行，波动率处于可控通道",
                    "support_resistance_tactic": "建议依托核心均线支撑位逢低吸纳，破位严格止损"
                },
                "key_points": [
                    f"技术面 RSI 指标处于 {rsi_val:.1f}，未触及极度超买区",
                    "均线密集区域提供良好支撑防守"
                ]
            }, ensure_ascii=False, indent=2)

        # 7. 合规风控 Agent (Risk Guardrail)
        if "风控" in full_prompt or "risk" in full_prompt or "guardrail" in full_prompt:
            return json.dumps({
                "agent_name": "Risk_Guardrail_Officer",
                "compliance_status": "APPROVED",
                "is_risk_violated": False,
                "audit_checklist": {
                    "debt_to_asset_check": "32.4% (低于 75% 警戒线，通过)",
                    "interest_coverage_check": "8.6x (远高于 1.5x 底线，通过)",
                    "var_95_check": "单周最大回撤 VaR 测算为 6.2% (低于 18% 阈值，通过)",
                    "regulatory_penalty_check": "无近3年立案调查或重大违规记录，合规通过"
                },
                "risk_rating": "中低风险 (R2 稳健型)",
                "hard_constraints_memo": "准予出具买入评级报告，强制附注周期波动与汇率敏感性提示。"
            }, ensure_ascii=False, indent=2)

        # 8. 默认返回综合投研研报 Markdown
        return """# 投资价值综合深度研究报告

## 一、 投委会核心结论
经过宏观经济、基本面财务三表、量化技术指标以及多空辩论（Bull vs Bear）的全面穿透审查，投委会出具 **【增持 (Overweight)】** 评级。

* **综合投研评分**：**7.9 / 10.0**
* **DCF 估值合理区间**：160.00 ~ 175.00 元（较当前市价具备约 20% 安全边际）
* **推荐仓位上限**：组合配置权重 45%~55%，分批布局
* **合规风控审核**：已通过资产负债率、利息覆盖率及蒙特卡洛 VaR 极限测算，符合公募/私募投资基准。

---

## 二、 维度专项剖析
1. **宏观与政策面（得分：7.6）**：当前处于流动性适度充裕周期，核心业务受益于国家设备更新与高端制造专项政策支持。
2. **基本面与财务质量（得分：8.4）**：杜邦拆解表明公司 ROE 达 22.7%，主要受 19.8% 净利率与高效运营驱动，财务杠杆率仅 32%，盈利成色极高。
3. **量化与形态面（得分：7.8）**：日线均线呈典型多头排列，MACD 零轴上方二次金叉，量能温和放大，短期支撑位明确。

---

## 三、 多空博弈委员会意见
* **多头辩手观点**：在手订单充沛，产品高端化溢价能力增强，具备较强的戴维斯双击动能。
* **空头辩手提醒**：需防范海外供应链汇兑波动与原材料短期脉冲成本，警惕二季度现金流回款节奏。
* **风控决议**：设定 8% 追踪止损红线，若连续 3 日跌破 60 日均线则触发防御性减仓。

> **免责声明**：本报告由多智能体协作投研系统自动生成，仅供专业投资参考，不构成直接买卖要约。市场有风险，投资需谨慎。
"""


# 全局单例
llm_provider = LLMProvider()

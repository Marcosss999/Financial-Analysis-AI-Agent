"""
对话记忆管理 (Chat Memory & Context Compression)
解决三个问题：
1. 谁来保存历史？  服务端(SQLite)按 session_id 保存，客户端不再自己维护、截断历史。
2. 历史太长怎么办？ 最近 N 轮保留原文，更早的消息压缩成摘要(滚动摘要：旧摘要 + 新溢出的消息 → 新摘要)，
                   再按字符预算兜底裁剪，保证 prompt 长度有上界。
3. 研报背景从哪来？ 直接取该标的最近一次分析的结构化结论(长期记忆)，而不是让前端把研报截断 2000 字再传回来。
压缩优先用 LLM 生成摘要；LLM 不可用(无 Key / 熔断)时退化为抽取式摘要，绝不把模拟引擎的无关文本写进记忆。
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.llm_provider import LLMUnavailableError, llm_provider
from app.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "你是对话记忆压缩器。把下面的对话压缩成不超过 200 字的要点，必须保留：标的代码/名称、"
    "用户关注的问题、已给出的关键数字与结论。只输出摘要正文，不要寒暄。"
)


def format_run_brief(run: Dict[str, Any]) -> str:
    """把一次历史分析压缩成一段可直接放进 prompt 的结构化摘要"""
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(run.get("created_at", 0)))
    dq = run.get("data_quality") or {}
    src = dq.get("quote_source", "?")
    return (
        f"[{ts}] {run.get('target_name') or run.get('ticker')}({run.get('ticker')}) "
        f"策略={run.get('strategy_style')} 结论={run.get('verdict')} 共识分={run.get('consensus_score')} "
        f"建议仓位={run.get('position')} 风控={run.get('risk_status')} 数据源={src}"
    )


def extractive_summary(messages: List[Dict[str, Any]], prev_summary: str = "", max_chars: int = 400) -> str:
    """不依赖 LLM 的兜底摘要：保留旧摘要 + 每条用户提问的开头，超长则截断(保留最近的部分)"""
    parts = [prev_summary] if prev_summary else []
    for m in messages:
        tag = "用户问" if m["role"] == "user" else "助手答"
        parts.append(f"{tag}：{m['content'].strip().replace(chr(10), ' ')[:60]}")
    text = "；".join(p for p in parts if p)
    return text[-max_chars:]


class ChatMemory:
    def __init__(self, store: MemoryStore):
        self.store = store

    async def _summarize(self, old_messages: List[Dict[str, Any]], prev_summary: str) -> str:
        if llm_provider.status()["mode"] != "live":
            return extractive_summary(old_messages, prev_summary)
        convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in old_messages)
        prompt = (f"已有摘要：{prev_summary or '无'}\n\n新增对话：\n{convo}\n\n请输出合并后的最新摘要。")
        try:
            text = await llm_provider.chat(
                [{"role": "system", "content": _SUMMARY_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.1, allow_mock=False,
            )
            return text.strip()[:600]
        except LLMUnavailableError:
            return extractive_summary(old_messages, prev_summary)

    async def build_context(self, session_id: str, ticker: str, system_prompt: str,
                            question: str, client_report_context: Optional[str] = None) -> List[Dict[str, str]]:
        """组装发给 LLM 的 messages：系统提示 + 历史分析结论 + 滚动摘要 + 最近原文 + 当前问题"""
        keep_msgs = settings.eng.CHAT_KEEP_RECENT_TURNS * 2
        all_msgs = await asyncio.to_thread(self.store.get_messages, session_id)
        summary_row = await asyncio.to_thread(self.store.get_summary, session_id)
        summary = summary_row["summary"] if summary_row else ""
        summarized_upto = summary_row["upto_id"] if summary_row else 0

        recent = all_msgs[-keep_msgs:] if len(all_msgs) > keep_msgs else all_msgs
        overflow = [m for m in all_msgs[:-keep_msgs] if m["id"] > summarized_upto] if len(all_msgs) > keep_msgs else []
        if overflow:
            summary = await self._summarize(overflow, summary)
            await asyncio.to_thread(self.store.set_summary, session_id, summary, overflow[-1]["id"])

        # 长期记忆：该标的最近一次分析结论 (取代前端回传被截断的研报全文)
        latest = await asyncio.to_thread(self.store.latest_run, ticker)
        sys_text = system_prompt
        if latest:
            sys_text += f"\n【该标的最近一次系统分析结论（来自记忆库）】\n{format_run_brief(latest)}\n"
            if latest.get("summary"):
                sys_text += f"{latest['summary'][:800]}\n"
        elif client_report_context:
            sys_text += f"\n【背景参考研报（客户端提供，已截断）】\n{client_report_context[:2000]}\n"
        if summary:
            sys_text += f"\n【此前对话摘要】\n{summary}\n"

        # 字符预算兜底：从最旧的原文开始丢，保证 prompt 有上界
        budget = settings.eng.CHAT_CONTEXT_CHAR_BUDGET - len(sys_text) - len(question)
        kept: List[Dict[str, str]] = []
        for m in reversed(recent):
            cost = len(m["content"])
            if budget - cost < 0:
                break
            budget -= cost
            kept.append({"role": m["role"], "content": m["content"]})
        kept.reverse()
        return [{"role": "system", "content": sys_text}, *kept, {"role": "user", "content": question}]

    async def remember(self, session_id: str, ticker: str, question: str, answer: str) -> None:
        await asyncio.to_thread(self.store.add_message, session_id, "user", question, ticker)
        await asyncio.to_thread(self.store.add_message, session_id, "assistant", answer, ticker)

    async def import_history(self, session_id: str, ticker: str, history: List[Dict[str, str]]) -> None:
        """兼容旧前端：会话在服务端还是空的、但客户端带了 history 时，把它一次性导入记忆库"""
        existing = await asyncio.to_thread(self.store.get_messages, session_id)
        if existing or not history:
            return
        for h in history[-settings.eng.CHAT_KEEP_RECENT_TURNS * 2:]:
            role = h.get("role", "user")
            if role in ("user", "assistant") and h.get("content"):
                await asyncio.to_thread(self.store.add_message, session_id, role, h["content"], ticker)

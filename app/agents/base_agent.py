"""
金融投研专业 Agent 抽象基类模块 (Base Specialist Agent)
职责：
1. 统一规范所有角色智能体 (宏观、基本面、技术面、辩论对决、风控仲裁、研报生成) 的生命周期与状态增量流转协议。
2. 封装高鲁棒性 JSON 提取解析器 (_parse_json_safely)，精准剥离 Markdown 围栏代码块并防御截断脏数据。
3. 统一接入系统底层 LLM 服务提供者 (LLMProvider)，支持统一重试与降级。
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.core.llm_provider import llm_provider, last_call_was_live
from app.models.schemas import SpecialistAgentOutput
from app.tools.scoring import stance_from_score
from app.models.agent_state import FinancialAgentState

logger = logging.getLogger(__name__)


class BaseSpecialistAgent(ABC):
    """金融投研领域专家 Agent 抽象基类"""

    def __init__(self, name: str, role_description: str, temperature: float = 0.2):
        self.name = name
        self.role_description = role_description
        self.temperature = temperature
        self.llm = llm_provider

    @abstractmethod
    async def run(self, state: FinancialAgentState) -> Dict[str, Any]:
        """
        执行专业投研推理并在状态图中产出状态增量字典
        :param state: 全局共享状态
        :return: 状态更新字典 (将合并至全局 FinancialAgentState)
        """
        pass

    async def _call_llm(self, messages: List[Dict[str, str]], json_mode: bool = True) -> str:
        """调用底层统一大模型服务门面"""
        return await self.llm.chat(
            messages=messages,
            temperature=self.temperature,
            json_mode=json_mode,
        )

    async def _call_llm_ex(self, messages: List[Dict[str, str]], json_mode: bool = True) -> "tuple[str, bool]":
        """调用 LLM 并返回 (文本, 是否来自真实远程模型)。离线/熔断/重试耗尽时 live=False，文本为本地模拟内容"""
        text = await self._call_llm(messages, json_mode=json_mode)
        return text, last_call_was_live()

    def build_output(
        self,
        *,
        quant_score: float,
        components: Dict[str, float],
        raw_reply: str,
        live: bool,
        fallback: Dict[str, Any],
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> SpecialistAgentOutput:
        """
        组装专家产出。关键规则：
        - score / stance 恒由确定性量化评分决定，与 LLM 是否在线无关；
        - LLM 在线时：采用 LLM 的 summary/details/key_points 作为解读文字，其主观评分只记为 llm_score(参考)；
        - LLM 不在线时：不使用任何模拟引擎的文字，改用基于数据事实生成的模板叙述。
        """
        parsed = self._parse_json_safely(raw_reply, {}) if live else {}
        use_llm_text = live and bool(parsed.get("summary"))
        src = parsed if use_llm_text else fallback
        details = dict(src.get("details") or {})
        if extra_details:
            details.update(extra_details)
        llm_score = None
        if use_llm_text and parsed.get("score") is not None:
            llm_score = to_score(parsed.get("score"), quant_score)
            details["llm_stance"] = parsed.get("stance", "")
        return SpecialistAgentOutput(
            agent_name=self.name,
            stance=stance_from_score(quant_score),
            score=round(quant_score, 2),
            summary=str(src.get("summary", "")),
            details=details,
            key_points=list(src.get("key_points") or []),
            quant_score=round(quant_score, 2),
            llm_score=llm_score,
            score_components=components,
            score_source="quant+llm_narrative" if use_llm_text else "quant+template_narrative",
        )

    def _parse_json_safely(self, raw_text: str, default_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        高鲁棒性结构化 JSON 解析器：
        1. 剥离 Markdown ```json 代码围栏
        2. 正则定位首尾最外层大括号
        3. 容错解析，避免 LLM 幻觉掺杂杂质文本导致崩溃
        """
        if not raw_text:
            return default_data or {}

        cleaned = raw_text.strip()

        # 剥离 ```json ... ``` 或 ``` ... ```
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        # 尝试直接反序列化
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 正则抓取最外层 { ... }
        brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
        if brace_match:
            try:
                return json.loads(brace_match.group(1).strip())
            except Exception as e:
                logger.warning(f"[{self.name}] 正则提取 JSON 反序列化失败: {e}")

        logger.warning(f"[{self.name}] 无法从模型回复中提取合法 JSON，启用默认保真结构。")
        return default_data or {}


def to_score(value, default: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """
    把 LLM 返回的分数安全转成 [lo, hi] 内的浮点数。
    LLM 可能返回 "7.5/10"、"约8分"、null、负数或 15 等脏值；这里统一容错并限幅，
    避免一次格式漂移让整条分析链路抛 ValueError，也避免越界分数污染共识分。
    """
    import re as _re
    try:
        if isinstance(value, (int, float)):
            num = float(value)
        else:
            m = _re.search(r"-?\d+(?:\.\d+)?", str(value))
            num = float(m.group()) if m else float(default)
    except (TypeError, ValueError):
        num = float(default)
    return max(lo, min(hi, num))

import numpy as np

from app.agents.base_agent import BaseSpecialistAgent, to_score
from app.core.config import settings
from app.core.llm_provider import LLMProvider, LLMUnavailableError
from app.core.observability import start_trace
from tests.helpers import override, run


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]
        self.usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 7})()


class FakeClient:
    """可编程的假 OpenAI 客户端：outcomes 里每项要么是返回文本，要么是要抛出的异常"""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0
        self.chat = type("Chat", (), {"completions": self})()

    async def create(self, **kwargs):
        self.calls += 1
        out = self.outcomes.pop(0) if self.outcomes else "default"
        if isinstance(out, Exception):
            raise out
        return _Resp(out)


def _provider(outcomes):
    p = LLMProvider()
    p.is_configured, p.client = True, FakeClient(outcomes)
    return p


MSGS = [{"role": "user", "content": "hi"}]


def test_no_key_falls_back_and_is_recorded_in_trace():
    p = LLMProvider()
    p.is_configured, p.client = False, None
    trace = start_trace("T")
    text = run(p.chat(MSGS))
    assert text and trace.llm["fallbacks"] == 1 and p.status()["mode"] == "mock"


def test_allow_mock_false_raises_instead_of_returning_mock():
    p = LLMProvider()
    p.is_configured, p.client = False, None
    try:
        run(p.chat(MSGS, allow_mock=False))
    except LLMUnavailableError:
        return
    raise AssertionError("应抛 LLMUnavailableError")


def test_success_records_tokens_and_no_fallback():
    p = _provider(["hello"])
    trace = start_trace("T")
    assert run(p.chat(MSGS)) == "hello"
    assert trace.llm["calls"] == 1 and trace.llm["fallbacks"] == 0
    assert (trace.llm["prompt_tokens"], trace.llm["completion_tokens"]) == (11, 7)


def test_retries_then_succeeds():
    p = _provider([ConnectionError("a"), ConnectionError("b"), "finally"])
    trace = start_trace("T")
    with override(settings.llm, RETRY_BASE_DELAY=0.001, MAX_RETRIES=2):
        assert run(p.chat(MSGS)) == "finally"
    assert p.client.calls == 3 and trace.llm["retries"] == 2 and trace.llm["fallbacks"] == 0


def test_empty_response_is_treated_as_failure():
    p = _provider(["", "  ", "real"])
    with override(settings.llm, RETRY_BASE_DELAY=0.001, MAX_RETRIES=2):
        assert run(p.chat(MSGS)) == "real"


def test_exhausted_retries_fall_back_and_trip_breaker():
    p = _provider([ConnectionError("x")] * 50)
    trace = start_trace("T")
    with override(settings.llm, RETRY_BASE_DELAY=0.001, MAX_RETRIES=1):
        for _ in range(3):                      # 连续 3 次失败 → 熔断器打开
            run(p.chat(MSGS))
        assert p.breaker.state == "open"
        calls_before = p.client.calls
        run(p.chat(MSGS))                       # 熔断打开后：不再调用远程，直接降级
        assert p.client.calls == calls_before
    assert trace.llm["fallbacks"] == 4
    try:
        run(p.chat(MSGS, allow_mock=False))
    except LLMUnavailableError:
        return
    raise AssertionError("熔断打开且 allow_mock=False 应抛异常")


def test_to_score_is_tolerant_and_clamped():
    assert to_score(7.5, 5) == 7.5
    assert to_score("8/10", 5) == 8.0
    assert to_score("约 6.5 分", 5) == 6.5
    assert to_score(None, 5.5) == 5.5
    assert to_score(15, 5) == 10.0 and to_score(-3, 5) == 0.0
    assert to_score("无法评分", 4.0) == 4.0


class _A(BaseSpecialistAgent):
    async def run(self, state):
        return {}


def test_parse_json_safely_variants():
    a = _A(name="t", role_description="t")
    dflt = {"d": 1}
    assert a._parse_json_safely('{"a": 1}', dflt) == {"a": 1}
    assert a._parse_json_safely('```json\n{"a": 2}\n```', dflt) == {"a": 2}
    assert a._parse_json_safely('前言 {"a": 3} 后语', dflt) == {"a": 3}
    assert a._parse_json_safely("完全不是 json", dflt) == dflt
    assert a._parse_json_safely("", dflt) == dflt

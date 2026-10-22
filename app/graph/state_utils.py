"""状态归并工具：按 FinancialAgentState 中 Annotated 声明的 reducer 合并节点增量更新。

流式接口需要在图外自己累积状态。直接 dict.update() 会把带 reducer 的字段
(debate_turns / execution_logs / degraded_nodes) 覆盖成“最后一个节点的值”，
导致 SSE 最终结果丢失辩论历史和执行日志。这里复用与 LangGraph 相同的归并规则。
"""

import typing
from typing import Any, Callable, Dict

from app.models.agent_state import FinancialAgentState

_REDUCERS: Dict[str, Callable[[Any, Any], Any]] = {
    k: v.__metadata__[0]
    for k, v in typing.get_type_hints(FinancialAgentState, include_extras=True).items()
    if typing.get_origin(v) is typing.Annotated
}


def merge_state(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """就地把节点 update 合并进 state：有 reducer 的字段追加，其余字段覆盖"""
    for key, value in (update or {}).items():
        reducer = _REDUCERS.get(key)
        state[key] = reducer(state.get(key) or [], value) if reducer else value
    return state

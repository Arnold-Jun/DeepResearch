"""
Common building blocks for graph-based agents.

Important: keep imports lazy to avoid forcing optional heavy deps (e.g. langgraph)
when only importing small helpers.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BaseGraphAgent": ("src.agent.common.base_graph_agent", "BaseGraphAgent"),
    "LLMReasoningNode": ("src.agent.common.llm_reasoning_node", "LLMReasoningNode"),
    "ToolExecutorNode": ("src.agent.common.tool_executor_node", "ToolExecutorNode"),
    "DecisionSchemaTool": ("src.agent.common.decision_schema", "DecisionSchemaTool"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        mod = import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(name)


__all__ = sorted(_EXPORTS.keys())



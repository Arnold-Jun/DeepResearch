from __future__ import annotations

from typing import Any, Sequence

from src.agent.common import DecisionSchemaTool


@TOOL.register_module(name="deep_analyzer_decision", force=True)
def make_deep_analyzer_decision_tool(*, tool_names: Sequence[str]) -> DecisionSchemaTool:
    """
    深度分析代理的专用决策模式。
    
    动作空间限制为：
    - route: call_tool | finish
    - tool_name: 配置的工具名称之一
    """
    tool_names_list = [t for t in tool_names if t]

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "description": "下一步路由到哪里。",
                "enum": ["call_tool", "finish"],
            },
            "tool_name": {
                "type": "string",
                "description": "要执行的工具名称（当 route=call_tool 时必需）。",
                "nullable": True,
                **({"enum": tool_names_list} if tool_names_list else {}),
            },
            "tool_args": {
                "type": "object",
                "description": "工具调用的参数（当 route=call_tool 时必需）。",
                "nullable": True,
            },
            "final_answer": {
                "type": "string",
                "description": "最终答案（当 route=finish 时必需）。",
                "nullable": True,
            },
        },
        "additionalProperties": False,
    }

    return DecisionSchemaTool(
        name="deep_analyzer_decide",
        description="为深度分析代理决定下一步操作。",
        parameters=parameters,
    )


__all__ = ["make_deep_analyzer_decision_tool"]



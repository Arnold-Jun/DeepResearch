from __future__ import annotations

from typing import Any, Sequence

from src.agent.common import DecisionSchemaTool


def make_browser_use_decision_tool(*, tool_names: Sequence[str]) -> DecisionSchemaTool:
    """浏览器使用代理的专用决策模式。"""
    tool_names_list = [t for t in tool_names if t]

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "description": "下一步路由到哪里。",
                "enum": ["call_tool", "call_tools_parallel", "call_llm_node", "finish"],
            },
            "tool_calls": {
                "type": "array",
                "description": "并行执行的工具调用列表（当 route=call_tools_parallel 时使用）。",
                "nullable": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "nullable": True,
                            **({"enum": tool_names_list} if tool_names_list else {}),
                        },
                        "tool_args": {
                            "type": "object",
                            "nullable": True,
                        },
                        "timeout_s": {
                            "type": "number",
                            "nullable": True,
                        },
                    },
                    "additionalProperties": False,
                    "required": ["tool_name"],
                },
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
            "llm_node": {
                "type": "string",
                "description": "要执行的内部节点名称（当 route=call_llm_node 时必需）。",
                "nullable": True,
                "enum": ["run"],
            },
            "llm_args": {
                "type": "object",
                "description": "内部节点的参数（可选）。",
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
        name="browser_use_decide",
        description="为浏览器使用代理决定下一步操作。",
        parameters=parameters,
    )


__all__ = ["make_browser_use_decision_tool"]



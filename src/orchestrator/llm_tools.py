from __future__ import annotations

from typing import Any

from src.tools import Tool


class TodoDeltaTool(Tool):
    """
    仅用于 PlanningLLM 输出 TodoDelta 的模式工具（仅父任务变更）。
    """

    name: str = "todo_delta"
    description: str = (
        "更新有序的父任务（PARENT）待办列表。"
        "你可以添加、更新或删除父任务待办项。"
        "不要重新排序项目。不要更新子任务（parent_id != null 的项目）。"
        "你不需要生成最终答案，最终答案将由专门的总结模块生成。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "description": "待办变更列表（添加/更新/删除）。",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "update", "delete"], "description": "操作类型"},
                        "id": {"type": "string", "nullable": True, "description": "目标待办 id（更新/删除时使用）"},
                        "item": {
                            "type": "object",
                            "nullable": True,
                            "description": "要添加的待办项（添加操作）。必须包含唯一的 id 和 task。",
                        },
                        "patch": {
                            "type": "object",
                            "nullable": True,
                            "description": "更新操作的补丁字段（更新操作时使用）。",
                        },
                    },
                    "required": ["op"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["actions"],
        "additionalProperties": False,
    }
    output_type: str = "any"

    def forward(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        return {"actions": actions}


class RoundPlanTool(Tool):
    """
    仅用于 SchedulerLLM 输出 RoundPlan 的模式工具。
    """

    name: str = "round_plan"
    description: str = (
        "决定如何执行选中父任务的下一轮。"
        "你必须选择恰好一个 agent_type 并为该父任务生成 N 个并行任务。"
        "如果可以完成，设置 route=finish，系统会自动调用总结模块生成最终答案。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["run_round", "finish"], "description": "下一步操作"},
            "agent_type": {
                "type": "string",
                "enum": ["deep_researcher_agent", "browser_use_agent", "deep_analyzer_agent"],
                "description": "此轮使用的单一代理类型。",
            },
            "parallelism": {"type": "integer", "description": "并行任务数量（<= 配置的上限）。"},
            "tasks": {
                "type": "array",
                "description": "此轮的子任务。所有子任务必须共享相同的 parent_id。注意：task_id 将由系统自动生成，你不需要提供。",
                "items": {
                    "type": "object",
                    "properties": {
                        "parent_id": {"type": "string", "description": "父任务 ID，必须等于所选父任务的 id"},
                        "task": {"type": "string", "description": "子任务描述"},
                    },
                    "required": ["parent_id", "task"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["route", "agent_type", "parallelism", "tasks"],
        "additionalProperties": False,
    }
    output_type: str = "any"

    def forward(
        self,
        route: str,
        agent_type: str,
        parallelism: int,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "route": route,
            "agent_type": agent_type,
            "parallelism": parallelism,
            "tasks": tasks,
        }


__all__ = ["TodoDeltaTool", "RoundPlanTool"]



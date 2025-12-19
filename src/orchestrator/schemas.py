from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


class TodoAddAction(TypedDict):
    op: Literal["add"]
    item: dict[str, Any]


class TodoUpdateAction(TypedDict):
    op: Literal["update"]
    id: str
    patch: dict[str, Any]


class TodoDeleteAction(TypedDict):
    op: Literal["delete"]
    id: str


TodoAction = TodoAddAction | TodoUpdateAction | TodoDeleteAction


class TodoDelta(TypedDict):
    actions: list[TodoAction]
    final_answer: Optional[str]


class RoundTask(TypedDict):
    task_id: str
    parent_id: str
    task: str


class RoundPlan(TypedDict):
    route: Literal["run_round", "finish"]
    agent_type: Literal["deep_researcher_agent", "browser_use_agent", "deep_analyzer_agent"]
    parallelism: int
    tasks: list[RoundTask]
    final_answer: Optional[str]



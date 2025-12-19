from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import time


TodoStatus = Literal["todo", "in_progress", "done", "blocked", "failed"]


@dataclass
class TodoItem:
    id: str
    task: str
    status: TodoStatus = "todo"
    parent_id: Optional[str] = None
    failure_count: int = 0
    agent_type: Optional[str] = None
    last_result_summary: Optional[str] = None
    last_error: Optional[str] = None
    attempt: int = 0
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    @property
    def is_parent(self) -> bool:
        return self.parent_id is None


@dataclass
class OrchestratorConfig:
    max_rounds: int = 12
    deadline_seconds: int = 600
    max_parallelism: int = 5
    max_failures_per_parent: int = 6
    subtask_timeout_seconds: int = 240
    subtask_output_max_chars: int = 1800
    subtask_failure_threshold: float = 0.5  # 子任务失败率阈值（0.0-1.0），超过此阈值则算作父任务失败
    planning_model_id: str = "qwen3-32b"
    scheduler_model_id: str = "qwen3-32b"
    summarizer_model_id: str = "qwen3-32b"  # Summarizer 使用的模型，默认与 Planner 相同


@dataclass
class OrchestratorState:
    task: str
    todo_list: list[TodoItem] = field(default_factory=list)
    round_index: int = 0
    deadline_ts: float = 0.0
    last_round_summary: Optional[str] = None
    last_round_agent_type: Optional[str] = None
    last_round_parent_id: Optional[str] = None

    def now(self) -> float:
        return time.time()

    def is_timed_out(self) -> bool:
        return self.deadline_ts > 0 and self.now() >= self.deadline_ts



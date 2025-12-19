from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from src.orchestrator.state import TodoItem
from src.logger import logger


@dataclass
class SubtaskRunResult:
    subtask_id: str
    ok: bool
    output: str | None
    error: str | None


class RoundRunner:
    def __init__(
        self,
        *,
        agent_factory: Callable[[str], Any],
        max_parallelism: int,
        timeout_s: int,
        output_max_chars: int,
    ):
        self.agent_factory = agent_factory
        self.max_parallelism = max_parallelism
        self.timeout_s = timeout_s
        self.output_max_chars = output_max_chars

    def _truncate(self, s: str | None) -> str | None:
        if s is None:
            return None
        if len(s) <= self.output_max_chars:
            return s
        return s[: self.output_max_chars] + "\n...[truncated]..."

    async def run_round(self, agent_type: str, subtasks: list[TodoItem]) -> list[SubtaskRunResult]:
        sem = asyncio.Semaphore(self.max_parallelism)

        async def run_one(st: TodoItem) -> SubtaskRunResult:
            async with sem:
                st.attempt += 1
                logger.info(f"[RoundRunner] 开始执行子任务 {st.id} (agent_type={agent_type}, task={st.task[:100]}...)")
                try:
                    agent = self.agent_factory(agent_type)
                    logger.info(f"[RoundRunner] Agent {agent_type} 已创建，开始执行任务...")
                    # agent.run is async
                    raw = await asyncio.wait_for(agent.run(st.task), timeout=self.timeout_s)
                    out = self._truncate(str(raw))
                    logger.info(f"[RoundRunner] 子任务 {st.id} 执行成功，输出长度: {len(out) if out else 0}")
                    return SubtaskRunResult(subtask_id=st.id, ok=True, output=out, error=None)
                except asyncio.TimeoutError:
                    error_msg = f"子任务 {st.id} 执行超时 (timeout={self.timeout_s}s)"
                    logger.error(f"[RoundRunner] {error_msg}")
                    return SubtaskRunResult(subtask_id=st.id, ok=False, output=None, error=error_msg)
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    logger.error(f"[RoundRunner] 子任务 {st.id} 执行失败: {error_msg}", exc_info=True)
                    return SubtaskRunResult(subtask_id=st.id, ok=False, output=None, error=error_msg)

        results = await asyncio.gather(*[run_one(st) for st in subtasks], return_exceptions=False)
        return results



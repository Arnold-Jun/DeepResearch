from __future__ import annotations

import time
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.logger import logger


class BaseGraphAgent:
    """
    Minimal agent wrapper around a LangGraph StateGraph.

    Requirements for integration with existing system:
    - has .name / .description
    - has async .run(task: str) -> Any
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        max_steps: int = 20,
        time_limit_seconds: int | None = None,
    ):
        self.name = name
        self.description = description
        self.max_steps = max_steps
        self.time_limit_seconds = time_limit_seconds

        self._graph = None

    def _build_graph(self) -> Any:
        raise NotImplementedError

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _initial_state(self, task: str) -> dict[str, Any]:
        deadline = None
        if self.time_limit_seconds:
            deadline = time.time() + float(self.time_limit_seconds)
        
        config = getattr(self, 'config', {}) or {}
        trace_max_len = config.get("trace_max_len", 8)  # 默认 8 条，与 prompt 中显示的 trace[-8:] 保持一致
        trace_observation_max_chars = config.get("trace_observation_max_chars", 1000)  # 默认 1000 字符
        
        return {
            "task": task,
            "step": 0,
            "max_steps": self.max_steps,
            "deadline": deadline,
            "trace": [],
            "trace_max_len": trace_max_len,
            "trace_observation_max_chars": trace_observation_max_chars,
            "decision": None,
            "tool_name": None,
            "tool_args": None,
            "tool_result": None,
            "observation": None,
            "last_tool_name": None,
            "is_final": False,
            "final_answer": None,
            "error": None,
        }

    async def run(self, task: str, **kwargs) -> Any:
        logger.info(f"[{self.name}] 开始执行任务: {task[:200]}...")
        state = self._initial_state(task)

        config = {
            "configurable": {
                "thread_id": f"{self.name}_{int(time.time())}",
            }
        }

        step_count = 0
        final_state = None
        try:
            async for update in self.graph.astream(state, config=config):
                if update:
                    step_count += 1
                    final_state = list(update.values())[-1]
                    # 记录关键状态变化
                    decision = final_state.get("decision")
                    if decision:
                        # 确保 decision 是字典类型
                        if isinstance(decision, dict):
                            route = decision.get("route", "unknown")
                            logger.debug(f"[{self.name}] Step {step_count}: route={route}")
                        else:
                            logger.debug(f"[{self.name}] Step {step_count}: decision={type(decision).__name__}")
                    if final_state.get("error"):
                        logger.warning(f"[{self.name}] Step {step_count}: 检测到错误: {final_state.get('error')}")
        except GeneratorExit:
            # 正常关闭，忽略此异常
            pass
        except Exception as e:
            logger.error(f"[{self.name}] 执行过程中出错: {e}", exc_info=True)
            if final_state:
                final_state["error"] = str(e)
            else:
                final_state = {"error": str(e), "final_answer": None}

        logger.info(f"[{self.name}] 任务执行完成，共 {step_count} 步")

        if not final_state:
            logger.warning(f"[{self.name}] 未获得最终状态")
            return "No result"
        if final_state.get("final_answer"):
            answer = final_state["final_answer"]
            logger.info(f"[{self.name}] 返回最终答案，长度: {len(str(answer))}")
            return answer
        if final_state.get("error"):
            error_msg = f"Failed: {final_state['error']}"
            logger.error(f"[{self.name}] 任务失败: {error_msg}")
            return error_msg
        logger.warning(f"[{self.name}] 未获得最终答案")
        return "No final answer"


__all__ = ["BaseGraphAgent", "StateGraph", "END", "MemorySaver"]



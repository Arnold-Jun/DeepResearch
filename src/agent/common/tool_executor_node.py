from __future__ import annotations

import asyncio
import json
import inspect
from typing import Any, Awaitable, Callable

from datetime import datetime, timezone

from src.logger import logger


ReducerFn = Callable[[dict[str, Any], Any], None]


class ToolExecutorNode:
    """
    Generic LangGraph node that executes tools based on state fields.

    Convention:
    - state["tool_name"] (str)
    - state["tool_args"] (dict)
    - OR state["tool_calls"] (list[{"tool_name": str, "tool_args": dict}])
    Writes back:
    - state["tool_result"] (Any)
    - state["tool_results"] (list[Any]) for parallel tool calls
    - state["observation"] (str)
    - state["error"] (str|None) on failure
    """

    def __init__(
        self,
        *,
        tools: list[Any] | dict[str, Any],
        reducers: dict[str, ReducerFn] | None = None,
        default_timeout_s: float | None = None,
        max_concurrency: int = 8,
    ):
        if isinstance(tools, dict):
            self.tools: dict[str, Any] = tools
        else:
            self.tools = {getattr(t, "name", type(t).__name__): t for t in tools}
        self.reducers = reducers or {}
        self.default_timeout_s = default_timeout_s
        self.max_concurrency = max(1, int(max_concurrency))

    def _stringify(self, value: Any) -> str:
        """将工具结果转换为字符串，优先处理 ToolResult 对象"""
        if hasattr(value, "output") and hasattr(value, "error"):
            if value.error:
                return f"错误: {value.error}"
            else:
                return str(value.output) if value.output is not None else ""
        
        try:
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        except Exception:
            return str(value)

    async def _call_tool(self, tool: Any, tool_args: dict[str, Any]) -> Any:
        if not callable(tool):
            raise TypeError(f"Unsupported tool type: {type(tool)}")

        call_target = tool
        try:
            if not (inspect.isfunction(tool) or inspect.ismethod(tool)) and hasattr(tool, "__call__"):
                call_target = tool.__call__  # type: ignore[assignment]
        except Exception:
            call_target = tool

        if inspect.iscoroutinefunction(call_target):
            try:
                return await tool(**tool_args, sanitize_inputs_outputs=True)
            except TypeError:
                return await tool(**tool_args)

        def _sync_call():
            try:
                return tool(**tool_args, sanitize_inputs_outputs=True)
            except TypeError:
                return tool(**tool_args)

        result = await asyncio.to_thread(_sync_call)
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            return await result
        return result

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        def _append_trace(*, tool_name: str, tool_args: dict[str, Any], ok: bool, observation: str, error: str | None) -> None:
            trace = state.get("trace")
            if not isinstance(trace, list):
                trace = []
            max_len = state.get("trace_max_len", 6)
            try:
                max_len = int(max_len)
            except Exception:
                max_len = 6

            max_obs = state.get("trace_observation_max_chars", 600)
            try:
                max_obs = int(max_obs)
            except Exception:
                max_obs = 600

            obs_preview = observation if isinstance(observation, str) else str(observation)
            if len(obs_preview) > max_obs:
                obs_preview = obs_preview[:max_obs] + "\n...[truncated]..."

            trace.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "ok": ok,
                    "observation_preview": obs_preview,
                    "error": error,
                }
            )
            if max_len > 0 and len(trace) > max_len:
                trace = trace[-max_len:]
            state["trace"] = trace

        tool_calls = None
        decision = state.get("decision")
        if isinstance(decision, dict):
            tool_calls_from_decision = decision.get("tool_calls")
            if isinstance(tool_calls_from_decision, list) and tool_calls_from_decision:
                tool_calls = tool_calls_from_decision
                logger.info(f"[ToolExecutorNode] Using tool_calls from decision: {len(tool_calls)} calls")
        
        if not tool_calls:
            tool_calls = state.get("tool_calls")
        
        if isinstance(tool_calls, list) and tool_calls:
            calls: list[dict[str, Any]] = [c for c in tool_calls if isinstance(c, dict)]
            if not calls:
                state["error"] = "tool_calls is empty/invalid"
                return state

            unknown = [c.get("tool_name") for c in calls if c.get("tool_name") not in self.tools]
            unknown = [u for u in unknown if u]
            if unknown:
                state["error"] = f"Unknown tool(s) {unknown}. Available: {list(self.tools.keys())}"
                return state

            timeout_default = state.get("tool_timeout_s", self.default_timeout_s)
            concurrency = state.get("tool_max_concurrency", self.max_concurrency)
            try:
                concurrency = max(1, int(concurrency))
            except Exception:
                concurrency = self.max_concurrency

            sem = asyncio.Semaphore(concurrency)

            async def _run_one(idx: int, call: dict[str, Any]) -> tuple[int, str, dict[str, Any], Any, str | None]:
                tool_name_i = call.get("tool_name")
                tool_args_i = call.get("tool_args") or {}
                tool_i = self.tools[tool_name_i]
                timeout_s = call.get("timeout_s", timeout_default)
                try:
                    async with sem:
                        if timeout_s:
                            result_i = await asyncio.wait_for(self._call_tool(tool_i, tool_args_i), timeout=float(timeout_s))
                        else:
                            result_i = await self._call_tool(tool_i, tool_args_i)
                    return (idx, tool_name_i, tool_args_i, result_i, None)
                except Exception as e:
                    return (idx, tool_name_i, tool_args_i, None, f"{type(e).__name__}: {e}")

            tasks = [asyncio.create_task(_run_one(i, c)) for i, c in enumerate(calls)]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0])

            tool_results: list[Any] = []
            errors: list[str] = []
            obs_lines: list[str] = []

            for _, tool_name_i, tool_args_i, result_i, err_i in results:
                if err_i is None:
                    tool_results.append(result_i)
                    obs_i = self._stringify(result_i).strip()
                    obs_lines.append(f"- {tool_name_i}: {obs_i[:800]}")
                    _append_trace(tool_name=tool_name_i, tool_args=tool_args_i, ok=True, observation=obs_i, error=None)
                else:
                    tool_results.append(None)
                    errors.append(f"{tool_name_i}: {err_i}")
                    obs_i = f"ERROR: {err_i}"
                    obs_lines.append(f"- {tool_name_i}: {obs_i}")
                    _append_trace(tool_name=tool_name_i, tool_args=tool_args_i, ok=False, observation=obs_i, error=err_i)

            state["tool_results"] = tool_results
            state["tool_result"] = tool_results[-1] if tool_results else None
            state["last_tool_name"] = results[-1][1] if results else None
            state["observation"] = "\n".join(obs_lines).strip()
            state.pop("tool_calls", None)
            for _, tool_name_i, _, result_i, err_i in results:
                if err_i is not None:
                    continue
                reducer = self.reducers.get(tool_name_i)
                if reducer:
                    try:
                        reducer(state, result_i)
                    except Exception as e:
                        logger.warning(f"[ToolExecutorNode] reducer failed for {tool_name_i}: {type(e).__name__}: {e}")

            state["error"] = "; ".join(errors) if errors else None
            return state

        decision = state.get("decision")
        tool_name = None
        tool_args = {}
        
        if isinstance(decision, dict):
            tool_name = decision.get("tool_name")
            tool_args = decision.get("tool_args") or {}
            if tool_name:
                logger.info(f"[ToolExecutorNode] Using tool_name from decision: {tool_name!r}")
        
        if not tool_name:
            tool_name = state.get("tool_name")
            tool_args = state.get("tool_args") or {}
            if tool_name:
                logger.info(f"[ToolExecutorNode] Using tool_name from state: {tool_name!r}")
        
        logger.debug(f"[ToolExecutorNode] Single tool mode - tool_name: {tool_name!r}, tool_args: {tool_args}")
        logger.debug(f"[ToolExecutorNode] State keys: {list(state.keys())}")
        logger.debug(f"[ToolExecutorNode] tool_name value: {repr(tool_name)}, type: {type(tool_name)}")

        if not tool_name:
            logger.error(f"[ToolExecutorNode] tool_name is missing or empty! State contains: {list(state.keys())}")
            logger.error(f"[ToolExecutorNode] tool_name value: {repr(tool_name)}, type: {type(tool_name)}")
            logger.error(f"[ToolExecutorNode] decision in state: {state.get('decision')}")
            state["error"] = "tool_name is missing"
            return state

        if tool_name not in self.tools:
            state["error"] = f"Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
            return state

        tool = self.tools[tool_name]

        try:
            timeout_s = state.get("tool_timeout_s", self.default_timeout_s)
            if timeout_s:
                result = await asyncio.wait_for(self._call_tool(tool, tool_args), timeout=timeout_s)
            else:
                result = await self._call_tool(tool, tool_args)

            state["tool_result"] = result
            state["observation"] = self._stringify(result).strip()
            state["last_tool_name"] = tool_name
            state.pop("tool_calls", None)

            _append_trace(tool_name=tool_name, tool_args=tool_args, ok=True, observation=state["observation"], error=None)

            reducer = self.reducers.get(tool_name)
            if reducer:
                try:
                    reducer(state, result)
                except Exception as e:
                    logger.warning(f"[ToolExecutorNode] reducer failed for {tool_name}: {type(e).__name__}: {e}")

            state["error"] = None
            return state
        except Exception as e:
            logger.error(f"[ToolExecutorNode] tool '{tool_name}' failed: {type(e).__name__}: {e}")
            state["tool_result"] = None
            state["observation"] = f"ERROR: {type(e).__name__}: {e}"
            state["error"] = str(e)
            state.pop("tool_calls", None)
            _append_trace(tool_name=tool_name, tool_args=tool_args, ok=False, observation=state["observation"], error=str(e))
            return state



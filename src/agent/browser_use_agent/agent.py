from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.models.base import Model
from src.registry import AGENT

from src.agent.common import (
    BaseGraphAgent,
    LLMReasoningNode,
    ToolExecutorNode,
)

from .decision import make_browser_use_decision_tool


def _tool_catalog_text(tools: dict[str, Any]) -> str:
    lines: list[str] = []
    for name, tool in tools.items():
        desc = getattr(tool, "description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "(no tools)"


@AGENT.register_module(name="browser_use_agent", force=True)
class BrowserUseGraphAgent(BaseGraphAgent):
    """
    基于 LangGraph 的浏览器使用子代理。

    方案 A（图优先）：
    - 确定性图控制工作流（无 LLM 路由决策）
    - 执行固定浏览器自动化工具（auto_browser_use_tool）并完成
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Model,
        tools: list[Any],
        max_steps: int = 20,
        name: str | None = None,
        description: str | None = None,
        **kwargs,
    ):
        self.config = config or {}
        self.model = model

        tools_dict: dict[str, Any] = {getattr(t, "name", type(t).__name__): t for t in (tools or [])}

        def reduce_any(state: dict[str, Any], result: Any) -> None:
            obs = state.get("observation", "")
            state["state_summary"] = {
                "last_tool": state.get("last_tool_name"),
                "observation_preview": obs[:1200] if isinstance(obs, str) else str(obs)[:1200],
            }

        reducers = {name: reduce_any for name in tools_dict.keys()}
        self._tool_executor = ToolExecutorNode(tools=tools_dict, reducers=reducers, default_timeout_s=300)

        # LLM 推理节点（代理"大脑"）+ 代理特定决策模式
        tool_names = list(self._tool_executor.tools.keys())
        decision_tool = make_browser_use_decision_tool(tool_names=tool_names)
        system_prompt = (
            "你是浏览器自动化控制器。"
            "决定下一步操作。"
            "优先调用浏览器工具完成任务，然后以提取的答案完成。"
            "如果需要多个工具调用，使用 route=call_tools_parallel 和 tool_calls。"
            "如果调用工具，使用允许列表中的 tool_name。"
            "如果调用 llm 节点，只选择 'run'。"
        )
        self._reason_node = LLMReasoningNode(
            model=self.model,
            system_prompt=system_prompt,
            decision_tool=decision_tool,
            tool_catalog_text=_tool_catalog_text(tools_dict) + "\n\nAvailable nodes:\n- run",
        )

        super().__init__(
            name=name or self.config.get("name") or "browser_use_agent",
            description=description or self.config.get("description") or "Browser use (LangGraph)",
            max_steps=max_steps or int(self.config.get("max_steps", 20)),
            time_limit_seconds=int(self.config.get("time_limit_seconds", 300)),
        )

    async def _run_browser_tool(self, state: dict[str, Any]) -> dict[str, Any]:
        tools = list(self._tool_executor.tools.keys())
        tool_name = "auto_browser_use_tool" if "auto_browser_use_tool" in self._tool_executor.tools else (tools[0] if tools else None)
        if not tool_name:
            state["error"] = "No tools available for browser_use_agent."
            state["final_answer"] = "Failed: no tools configured."
            state["is_final"] = True
            return state

        state["tool_name"] = tool_name
        # 大多数浏览器工具接受 task/instruction 作为 `task`
        state["tool_args"] = {"task": state.get("task", "")}
        state = await self._tool_executor(state)
        return state

    def _route_from_decision(self, state: dict[str, Any]) -> str:
        decision = state.get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}

        route = decision.get("route")
        if route == "finish":
            state["is_final"] = True
            state["final_answer"] = decision.get("final_answer") or state.get("observation") or "Done."
            return "end"

        if route in ("call_tool", "call_tools_parallel"):
            if route == "call_tools_parallel":
                tool_calls = decision.get("tool_calls")
                if not (isinstance(tool_calls, list) and tool_calls):
                    route = "call_tool"
                else:
                    normalized = []
                    tools = list(self._tool_executor.tools.keys())
                    default_tool = "auto_browser_use_tool" if "auto_browser_use_tool" in self._tool_executor.tools else (tools[0] if tools else None)
                    for c in tool_calls:
                        if not isinstance(c, dict):
                            continue
                        tn = c.get("tool_name") or default_tool
                        if not tn:
                            continue
                        ta = c.get("tool_args")
                        if not isinstance(ta, dict):
                            ta = {"task": state.get("task", "")}
                        call = {"tool_name": tn, "tool_args": ta}
                        if c.get("timeout_s") is not None:
                            call["timeout_s"] = c.get("timeout_s")
                        normalized.append(call)
                    if normalized:
                        state["tool_calls"] = normalized
                        return "tool_executor"
                    route = "call_tool"

            tool_name = decision.get("tool_name")
            if not tool_name:
                tools = list(self._tool_executor.tools.keys())
                tool_name = "auto_browser_use_tool" if "auto_browser_use_tool" in self._tool_executor.tools else (tools[0] if tools else None)
            if not tool_name:
                state["is_final"] = True
                state["final_answer"] = "Failed: no tools configured."
                return "end"
            return "tool_executor"

        if route == "call_llm_node":
            llm_node = decision.get("llm_node")
            if llm_node == "run":
                return "run"
        return "run"

    async def _finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error"):
            state["final_answer"] = f"Failed: {state['error']}"
        else:
            state["final_answer"] = state.get("observation") or state.get("final_answer") or "Done."
        state["is_final"] = True
        return state

    def _build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("tool_executor", self._tool_executor)
        workflow.add_node("run", self._run_browser_tool)

        workflow.set_entry_point("reason")
        workflow.add_conditional_edges(
            "reason",
            self._route_from_decision,
            {
                "tool_executor": "tool_executor",
                "run": "run",
                "end": END,
            },
        )
        workflow.add_edge("tool_executor", "reason")
        workflow.add_edge("run", "reason")
        return workflow.compile()



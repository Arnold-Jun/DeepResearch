from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.models.base import Model
from src.registry import AGENT
from src.tools.research.smart_search_tool import SmartSearchTool
from src.tools.research.insight_reflector_tool import InsightReflectorTool
from src.tools.research.research_tools import GenerateSummaryTool

from src.agent.common import (
    BaseGraphAgent,
    LLMReasoningNode,
    ToolExecutorNode,
)

from .decision import make_deep_researcher_decision_tool


def _tool_catalog_text(tools: dict[str, Any]) -> str:
    lines: list[str] = []
    for name, tool in tools.items():
        desc = getattr(tool, "description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "(no tools)"


def _ensure_tool(tools: dict[str, Any], tool_name: str, factory):
    if tool_name not in tools:
        tools[tool_name] = factory()


@AGENT.register_module(name="deep_researcher_agent", force=True)
class DeepResearcherGraphAgent(BaseGraphAgent):
    """
    基于 LangGraph 的深度研究子代理（重构版）。

    架构变更：
    - 旧架构：OptimizeNode -> SearchNode -> ExtractNode -> FollowUpNode
    - 新架构：Reason -> ToolExecutor (SmartSearchTool | InsightReflectorTool) -> Reason
    
    优势：
    - 减少图节点和状态开销
    - 将确定性工作流封装在工具内部
    - 状态机逻辑更清晰
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
        self.model_id = self.config.get("model_id") or getattr(model, "model_id", None) or "qwen3-14b"
        self.summary_model_id = self.config.get("summary_model_id") or self.model_id

        tools_dict: dict[str, Any] = {getattr(t, "name", type(t).__name__): t for t in (tools or [])}

        # 1. 注册新工具：SmartSearchTool
        _ensure_tool(tools_dict, "smart_search_tool", lambda: SmartSearchTool(model_id=self.model_id))
        
        # 2. 注册新工具：InsightReflectorTool
        _ensure_tool(tools_dict, "insight_reflector_tool", lambda: InsightReflectorTool(model_id=self.model_id))
        
        # 3. 确保总结工具（用于 summarize 节点）
        self._summary_tool = GenerateSummaryTool(summary_model_id=self.summary_model_id)

        # 归约器：将工具执行结果映射到全局状态
        def reduce_smart_search(state: dict[str, Any], result: Any) -> None:
            output = result if isinstance(result, dict) else (getattr(result, "output", {}) or {})
            
            # 更新当前查询状态
            if output.get("optimized_query"):
                state["current_query"] = output.get("optimized_query")
            
            # 存储搜索结果（原始列表，供 InsightReflectorTool 使用）
            raw_results = output.get("search_results", [])
            state["search_results"] = raw_results
            
            # 将搜索结果转化为文本摘要，放入 observation 供 LLM 阅读
            count = len(raw_results)
            state["observation"] = f"SmartSearch completed. Found {count} results for query: {state.get('current_query')}. Ready for analysis."
            
            # 状态标记
            state["need_extract"] = True
            state["need_search"] = False
            state["state_summary"] = self._build_state_summary(state)

        def reduce_insight_reflector(state: dict[str, Any], result: Any) -> None:
            output = result if isinstance(result, dict) else (getattr(result, "output", {}) or {})
            
            new_insights = output.get("insights", [])
            new_followups = output.get("new_followups", [])
            
            # 合并 Insights
            current_insights = state.get("insights", [])
            state["insights"] = current_insights + new_insights
            
            # 替换 Pending Followups (模型已基于最新情况重新规划)
            # 或者我们选择追加？这里选择追加并去重，保留最新的在前
            current_pending = state.get("pending_followups", [])
            merged_followups = []
            seen = set()
            for q in new_followups + current_pending:
                if q not in seen and isinstance(q, str):
                    merged_followups.append(q)
                    seen.add(q)
            
            state["pending_followups"] = merged_followups[: int(self.config.get("max_follow_ups", 3))]
            
            state["observation"] = (
                f"InsightReflector completed. Extracted {len(new_insights)} new insights. "
                f"Generated {len(new_followups)} new follow-up questions. "
                f"Top follow-up: {merged_followups[0] if merged_followups else 'None'}"
            )
            
            state["need_extract"] = False
            state["need_search"] = True # 通常分析完后，如果有新问题，就需要搜
            state["state_summary"] = self._build_state_summary(state)

        def reduce_any(state: dict[str, Any], result: Any) -> None:
            # Fallback for other tools
            obs = str(result)[:800]
            state["observation"] = obs
            state["state_summary"] = self._build_state_summary(state)

        reducers = {
            "smart_search_tool": reduce_smart_search,
            "insight_reflector_tool": reduce_insight_reflector,
        }
        # 为其他工具添加默认 reducer
        for name in tools_dict:
            if name not in reducers:
                reducers[name] = reduce_any

        self._tool_executor = ToolExecutorNode(tools=tools_dict, reducers=reducers, default_timeout_s=120)

        # LLM 推理节点
        node_names = ["summarize"]
        tool_names = list(self._tool_executor.tools.keys())
        decision_tool = make_deep_researcher_decision_tool(tool_names=tool_names)
        
        system_prompt = (
            "你是深度研究控制器（重构版）。决定下一步操作。\n"
            "工作流指导：\n"
            "1. 如果需要搜索信息，调用 'smart_search_tool'（它会自动优化关键词）。\n"
            "2. 搜索完成后，必须调用 'insight_reflector_tool' 来提取知识并规划后续问题。\n"
            "3. 即使 search_results 看起来很多，也必须经过 insight_reflector_tool 处理才能存入长期记忆。\n"
            "4. 如果 pending_followups 中有高质量问题，选择它作为下一次搜索的 query。\n"
            "5. 当收集到足够信息或没有更多有价值的问题时，路由到 'summarize' 节点结束。\n"
        )
        
        self._reason_node = LLMReasoningNode(
            model=self.model,
            system_prompt=system_prompt,
            decision_tool=decision_tool,
            tool_catalog_text=_tool_catalog_text(tools_dict) + "\n\nAvailable nodes:\n- summarize",
        )

        super().__init__(
            name=name or self.config.get("name") or "deep_researcher_agent",
            description=description or self.config.get("description") or "Deep researcher (Refactored)",
            max_steps=max_steps or int(self.config.get("max_steps", 20)),
            time_limit_seconds=int(self.config.get("time_limit_seconds", 120)),
        )

    def _build_state_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "current_query": state.get("current_query"),
            "insights_count": len(state.get("insights", [])),
            "pending_followups_count": len(state.get("pending_followups", [])),
            "top_followup": (state.get("pending_followups", []) or ["None"])[0],
        }

    def _initial_state(self, task: str) -> dict[str, Any]:
        state = super()._initial_state(task)
        state.update(
            {
                "original_query": task,
                "current_query": task,
                "search_results": [],
                "insights": [],
                "pending_followups": [],
                "need_search": True,
                "need_extract": False,
            }
        )
        return state

    def _route_from_decision(self, state: dict[str, Any]) -> str:
        """路由器：将 LLM 决策映射到图节点。"""
        if state.get("is_final"):
            return "end"

        # 致命错误处理
        if state.get("error") and state.get("stop_on_error", False):
            state["is_final"] = True
            state["final_answer"] = f"Failed: {state['error']}"
            return "end"

        # 自动总结逻辑：如果步数用尽且有成果，强制跳转总结
        step = int(state.get("step", 0))
        max_steps = int(state.get("max_steps", 20))
        insights = state.get("insights", [])
        has_final_answer = bool(state.get("final_answer"))
        
        if step >= max_steps - 1 and insights and not has_final_answer:
            return "summarize"

        decision = state.get("decision") or {}
        route = decision.get("route")

        if route == "finish":
            state["is_final"] = True
            state["final_answer"] = decision.get("final_answer") or "Done."
            return "end"

        if route in ("call_tool", "call_tools_parallel"):
            # 标准工具调用逻辑
            if route == "call_tools_parallel":
                tool_calls = decision.get("tool_calls")
                if not (isinstance(tool_calls, list) and tool_calls):
                    route = "call_tool" # Fallback
                else:
                    normalized = []
                    for c in tool_calls:
                       if isinstance(c, dict) and c.get("tool_name"):
                           normalized.append(c)
                    if normalized:
                        state["tool_calls"] = normalized
                        return "tool_executor"
                    route = "call_tool"

            tool_name = decision.get("tool_name")
            if not tool_name:
                # 智能默认：根据状态推荐工具
                if state.get("need_search"):
                    tool_name = "smart_search_tool"
                elif state.get("need_extract"):
                    tool_name = "insight_reflector_tool"
            
            if tool_name:
                # 如果是自动推荐的，需要补全 args
                if not decision.get("tool_args"):
                    if tool_name == "smart_search_tool":
                        # 尝试从 pending 中取一个，或者用 current
                        pending = state.get("pending_followups", [])
                        query = pending[0] if pending else state.get("current_query")
                        state["decision"]["tool_args"] = {"query": query}
                    elif tool_name == "insight_reflector_tool":
                        state["decision"]["tool_args"] = {
                            "query": state.get("current_query"),
                            "search_results": state.get("search_results", []),
                            "original_task": state.get("original_query")
                        }
                return "tool_executor"

            state["is_final"] = True
            state["final_answer"] = "Failed: no tools configured."
            return "end"

        if route == "call_llm_node":
            node = decision.get("llm_node")
            if node == "summarize":
                return "summarize"
        
        return "summarize"

    async def _summarize_node(self, state: dict[str, Any]) -> dict[str, Any]:
        res = await self._summary_tool.forward(
            query=state.get("original_query", ""),
            insights=state.get("insights", []),
            visited_urls=[], # 不再显式追踪 URL，简化状态
            depth_reached=int(state.get("step", 0)), # 用步数代替深度
        )
        if res.error:
            state["error"] = res.error
        state["final_answer"] = res.output or state.get("final_answer") or ""
        state["is_final"] = True
        return state

    def _build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("tool_executor", self._tool_executor)
        workflow.add_node("summarize", self._summarize_node)

        workflow.set_entry_point("reason")
        
        workflow.add_conditional_edges(
            "reason",
            self._route_from_decision,
            {
                "tool_executor": "tool_executor",
                "summarize": "summarize",
                "end": END,
            },
        )

        workflow.add_edge("tool_executor", "reason")
        workflow.add_edge("summarize", END)

        return workflow.compile()



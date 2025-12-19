from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.models.base import Model
from src.registry import AGENT
from src.tools.research.web_searcher import WebSearcherTool, SearchResult
from src.tools.research.research_tools import (
    BatchExtractInsightsTool,
    GenerateFollowUpsTool,
    GenerateSummaryTool,
    OptimizeQueryTool,
)

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
    基于 LangGraph 的深度研究子代理。

    方案 B（LLM + 每代理路由器）：
    - LLMReasoningNode 产生结构化决策（模式工具调用）
    - 此代理的路由器确定性地将该决策映射到现有图节点
    - 确定性工具通过 ToolExecutorNode 执行
    - 单步 LLM 转换在固定节点中执行（优化/提取/后续问题/总结）
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

        _ensure_tool(tools_dict, "web_searcher_tool", lambda: WebSearcherTool())
        tools_dict["web_searcher_tool"].fetch_content = True

        self._optimize_query_tool = OptimizeQueryTool(model_id=self.model_id)
        self._batch_extract_tool = BatchExtractInsightsTool(model_id=self.model_id)
        self._followups_tool = GenerateFollowUpsTool(model_id=self.model_id)
        self._summary_tool = GenerateSummaryTool(summary_model_id=self.summary_model_id)

        # 归约器：将确定性工具执行结果映射到图状态
        def reduce_web_searcher(state: dict[str, Any], result: Any) -> None:
            # WebSearcherTool 返回带有 .results 的 SearchResponse（ToolResult 子类）
            results = getattr(result, "results", None) or []
            normalized: list[dict[str, Any]] = []
            for item in results:
                if isinstance(item, SearchResult):
                    normalized.append(
                        {
                            "url": item.url,
                            "title": item.title,
                            "content": item.raw_content or item.description or "",
                        }
                    )
                elif isinstance(item, dict):
                    normalized.append(
                        {
                            "url": item.get("url", ""),
                            "title": item.get("title") or "",
                            "content": item.get("raw_content") or item.get("content") or item.get("description") or "",
                        }
                    )
            state["search_results"] = normalized
            # 保持摘要简洁
            state["state_summary"] = {
                "original_query": state.get("original_query"),
                "current_query": state.get("current_query"),
                "depth": state.get("current_depth", 0),
                "insights": len(state.get("insights", [])),
                "pending_followups": len(state.get("pending_followups", [])),
                "search_results": len(normalized),
            }

        reducers = {
            "web_searcher_tool": reduce_web_searcher,
        }

        self._tool_executor = ToolExecutorNode(tools=tools_dict, reducers=reducers, default_timeout_s=60)

        # LLM 推理节点 + 代理特定决策模式（受限动作空间）
        node_names = [
            "optimize_query",
            "search_web",
            "extract_insights",
            "generate_followups",
            "select_followup",
            "summarize",
        ]
        tool_names = list(self._tool_executor.tools.keys())
        decision_tool = make_deep_researcher_decision_tool(tool_names=tool_names)
        system_prompt = (
            "你是深度研究控制器。决定下一步操作。"
            "你可以调用确定性工具（例如 web_searcher_tool），"
            "或并行调用多个工具（使用 route=call_tools_parallel 和 tool_calls），"
            "或路由到内部节点之一进行固定 LLM 转换（优化/提取/后续问题/总结），"
            "或在有足够证据时完成。"
        )
        self._reason_node = LLMReasoningNode(
            model=self.model,
            system_prompt=system_prompt,
            decision_tool=decision_tool,
            tool_catalog_text=_tool_catalog_text(tools_dict) + "\n\nAvailable nodes:\n- " + "\n- ".join(node_names),
        )

        super().__init__(
            name=name or self.config.get("name") or "deep_researcher_agent",
            description=description or self.config.get("description") or "Deep researcher (LangGraph)",
            max_steps=max_steps or int(self.config.get("max_steps", 20)),
            time_limit_seconds=int(self.config.get("time_limit_seconds", 120)),
        )

    def _initial_state(self, task: str) -> dict[str, Any]:
        state = super()._initial_state(task)
        state.update(
            {
                "original_query": task,
                "current_query": task,
                "filter_year": None,
                "search_results": [],
                "insights": [],
                "visited_urls": [],
                "pending_followups": [],
                "current_depth": 0,
                "max_depth": int(self.config.get("max_depth", 2)),
                "max_insights": int(self.config.get("max_insights", 20)),
                "optimized_once": False,
                "need_search": True,
                "need_extract": False,
            }
        )
        return state

    def _route_from_decision(self, state: dict[str, Any]) -> str:
        """每代理路由器：确定性地将 LLM 决策映射到图节点。"""
        if state.get("is_final"):
            return "end"

        # 致命错误处理
        if state.get("error") and state.get("stop_on_error", False):
            state["is_final"] = True
            state["final_answer"] = f"Failed: {state['error']}"
            return "end"

        # 检查是否需要总结：达到 max_steps 但有 insights 的情况
        needs_summary = state.get("_needs_summary_before_finish", False)
        insights = state.get("insights", [])
        has_final_answer = bool(state.get("final_answer"))
        
        # 如果需要总结且有 insights 但还没有 final_answer，自动路由到 summarize
        if needs_summary and insights and not has_final_answer:
            return "summarize"
        
        # 检查是否接近 max_steps，如果有 insights 但还没有 final_answer，自动路由到 summarize
        step = int(state.get("step", 0))
        max_steps = int(state.get("max_steps", 20))
        
        # 如果接近 max_steps（剩余1步）且有 insights 但还没有 final_answer，自动总结
        if step >= max_steps - 1 and insights and not has_final_answer:
            return "summarize"

        decision = state.get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}

        route = decision.get("route")
        if route == "finish":
            state["is_final"] = True
            state["final_answer"] = decision.get("final_answer") or state.get("final_answer") or "Done."
            return "end"

        if route in ("call_tool", "call_tools_parallel"):
            if route == "call_tools_parallel":
                tool_calls = decision.get("tool_calls")
                if not (isinstance(tool_calls, list) and tool_calls):
                    # 无效的并行请求；回退到单工具行为
                    route = "call_tool"
                else:
                    normalized = []
                    query_default = state.get("current_query") or state.get("task", "")
                    default_tool = "web_searcher_tool" if "web_searcher_tool" in self._tool_executor.tools else None
                    for c in tool_calls:
                        if not isinstance(c, dict):
                            continue
                        tn = c.get("tool_name") or default_tool
                        if not tn:
                            continue
                        ta = c.get("tool_args")
                        if not isinstance(ta, dict):
                            ta = {"query": query_default} if tn == "web_searcher_tool" else {}
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
                # 此代理的安全默认值
                tool_name = "web_searcher_tool" if "web_searcher_tool" in self._tool_executor.tools else None
            if not tool_name:
                state["is_final"] = True
                state["final_answer"] = "Failed: no tools configured."
                return "end"
            return "tool_executor"

        if route == "call_llm_node":
            node = decision.get("llm_node")
            allowed = {
                "optimize_query",
                "search_web",
                "extract_insights",
                "generate_followups",
                "select_followup",
                "summarize",
            }
            if node in allowed:
                return node

        if not state.get("optimized_once"):
            return "optimize_query"
        if state.get("need_search"):
            return "search_web"
        if state.get("need_extract"):
            return "extract_insights"
        return "summarize"

    async def _optimize_query_node(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("current_query") or state.get("original_query") or state.get("task", "")
        res = await self._optimize_query_tool.forward(query=query)
        out = res.output or {}
        state["current_query"] = out.get("optimized_query") or query
        state["filter_year"] = out.get("filter_year")
        state["observation"] = f"optimized_query={state['current_query']}, filter_year={state['filter_year']}"
        state["error"] = res.error
        state["optimized_once"] = True
        return state

    async def _search_web_node(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("current_query") or state.get("original_query") or state.get("task", "")
        state["tool_name"] = "web_searcher_tool"
        state["tool_args"] = {"query": query}
        state = await self._tool_executor(state)
        state["need_search"] = False
        state["need_extract"] = True
        return state

    async def _extract_insights_node(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("current_query") or state.get("original_query") or state.get("task", "")
        search_results = state.get("search_results", []) or []
        res = await self._batch_extract_tool.forward(query=query, search_results=search_results)
        output = res.output or []
        if not isinstance(output, list):
            output = []

        insights = state.get("insights", []) or []
        visited = set(state.get("visited_urls", []) or [])
        for it in output:
            if isinstance(it, dict):
                insights.append(it)
                url = it.get("source_url") or it.get("url")
                if url:
                    visited.add(url)

        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for it in insights:
            if not isinstance(it, dict):
                continue
            key = (it.get("content", "") or "", it.get("source_url", "") or "")
            if key not in dedup:
                dedup[key] = it

        state["insights"] = list(dedup.values())[: int(state.get("max_insights", 20))]
        state["visited_urls"] = list(visited)
        state["observation"] = f"extracted_insights={len(output)}, total_insights={len(state['insights'])}"
        state["error"] = res.error
        state["need_extract"] = False
        state["need_search"] = False

        state["state_summary"] = {
            "original_query": state.get("original_query"),
            "current_query": state.get("current_query"),
            "depth": state.get("current_depth", 0),
            "insights": len(state.get("insights", [])),
            "pending_followups": len(state.get("pending_followups", [])),
            "search_results": len(search_results),
        }
        return state

    async def _generate_followups_node(self, state: dict[str, Any]) -> dict[str, Any]:
        insights = state.get("insights", [])
        insight_texts = []
        for it in insights[:10]:
            if isinstance(it, dict):
                insight_texts.append(it.get("content", ""))
            else:
                insight_texts.append(str(it))
        res = await self._followups_tool.forward(
            original_query=state.get("original_query", ""),
            current_query=state.get("current_query", ""),
            insights=[t for t in insight_texts if t],
        )
        followups = res.output or []
        if isinstance(followups, list):
            existing = state.get("pending_followups", [])
            merged = list(dict.fromkeys(existing + [q for q in followups if isinstance(q, str) and q.strip()]))
            state["pending_followups"] = merged[: int(self.config.get("max_follow_ups", 3))]
        state["observation"] = f"generated_followups={len(state.get('pending_followups', []))}"
        state["error"] = res.error
        return state

    async def _select_followup_node(self, state: dict[str, Any]) -> dict[str, Any]:
        pending = state.get("pending_followups", []) or []
        if not pending:
            return state
        next_q = pending.pop(0)
        state["pending_followups"] = pending
        state["current_query"] = next_q
        state["current_depth"] = int(state.get("current_depth", 0)) + 1
        state["need_search"] = True
        state["need_extract"] = False
        state["observation"] = f"selected_followup depth={state['current_depth']} query={next_q}"
        return state

    async def _summarize_node(self, state: dict[str, Any]) -> dict[str, Any]:
        res = await self._summary_tool.forward(
            query=state.get("original_query", ""),
            insights=state.get("insights", []),
            visited_urls=state.get("visited_urls", []),
            depth_reached=int(state.get("current_depth", 0)),
        )
        if res.error:
            state["error"] = res.error
        state["final_answer"] = res.output or state.get("final_answer") or ""
        state["is_final"] = True
        state.pop("_needs_summary_before_finish", None)
        return state

    def _build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("tool_executor", self._tool_executor)
        workflow.add_node("optimize_query", self._optimize_query_node)
        workflow.add_node("search_web", self._search_web_node)
        workflow.add_node("extract_insights", self._extract_insights_node)
        workflow.add_node("generate_followups", self._generate_followups_node)
        workflow.add_node("select_followup", self._select_followup_node)
        workflow.add_node("summarize", self._summarize_node)

        workflow.set_entry_point("reason")
        workflow.add_conditional_edges(
            "reason",
            self._route_from_decision,
            {
                "tool_executor": "tool_executor",
                "optimize_query": "optimize_query",
                "search_web": "search_web",
                "extract_insights": "extract_insights",
                "generate_followups": "generate_followups",
                "select_followup": "select_followup",
                "summarize": "summarize",
                "end": END,
            },
        )

        workflow.add_edge("tool_executor", "reason")
        workflow.add_edge("optimize_query", "reason")
        workflow.add_edge("search_web", "reason")
        workflow.add_edge("extract_insights", "reason")
        workflow.add_edge("generate_followups", "reason")
        workflow.add_edge("select_followup", "reason")
        workflow.add_edge("summarize", END)

        return workflow.compile()



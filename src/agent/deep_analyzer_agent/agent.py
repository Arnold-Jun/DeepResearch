from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.models.base import Model
from src.registry import AGENT
from src.tools.analysis.smart_analyze_tool import SmartAnalyzeTool
from src.tools.analysis.analysis_tools import SummarizeAnalysisTool

from src.agent.common import (
    BaseGraphAgent,
    LLMReasoningNode,
    ToolExecutorNode,
)

from .decision import make_deep_analyzer_decision_tool


def _tool_catalog_text(tools: dict[str, Any]) -> str:
    lines: list[str] = []
    for name, tool in tools.items():
        desc = getattr(tool, "description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "(no tools)"


def _ensure_tool(tools: dict[str, Any], tool_name: str, factory):
    if tool_name not in tools:
        tools[tool_name] = factory()


@AGENT.register_module(name="deep_analyzer_agent", force=True)
class DeepAnalyzerGraphAgent(BaseGraphAgent):
    """
    基于 LangGraph 的深度分析子代理（重构版）。
    
    架构变更：
    - 旧架构：Reason -> AnalyzeNode / ParallelAnalyzeNode / SummarizeNode
    - 最终架构：Reason <-> ToolExecutor (SmartAnalyzeTool) -> SummarizeNode
    
    优势：
    - 总结节点增加了流程确定性（大门在身后锁死）
    - 节省了最后一次 LLM 推理（无需再次决策退出）
    - 维持了与 DeepResearcherAgent 的架构对称性
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
        self.summarizer_model_id = self.config.get("summarizer_model_id") or self.model_id
        
        analyzer_model_ids = self.config.get("analyzer_model_ids", [self.model_id])
        if not isinstance(analyzer_model_ids, list):
            analyzer_model_ids = [analyzer_model_ids]

        tools_dict: dict[str, Any] = {getattr(t, "name", type(t).__name__): t for t in (tools or [])}

        # 1. 注册 unified analyze tool
        _ensure_tool(tools_dict, "smart_analyze_tool", lambda: SmartAnalyzeTool(
            model_id=self.model_id, 
            analyzer_model_ids=analyzer_model_ids
        ))
        
        # 2. 这里的总结工具不再作为普通工具放入 tools_dict，
        # 而是作为类属性，供专门的 summarize 节点调用。
        self._summary_tool = SummarizeAnalysisTool(
            summarizer_model_id=self.summarizer_model_id
        )

        # Reducers
        def reduce_smart_analyze(state: dict[str, Any], result: Any) -> None:
            output = result if isinstance(result, dict) else (getattr(result, "output", {}) or {})
            
            if isinstance(output, dict):
                 state["analysis_results"] = output
                 # Build observation
                 obs_parts = []
                 for name, analysis in output.items():
                     analysis_snippet = str(analysis)[:500] + "..." if len(str(analysis)) > 500 else str(analysis)
                     obs_parts.append(f"Analysis from {name}:\n{analysis_snippet}\n")
                 state["observation"] = "\n".join(obs_parts)
            else:
                 state["analysis_results"] = {"default": str(output)}
                 state["observation"] = str(output)[:1000]
            
            state["analyzed_once"] = True
            state["state_summary"] = {
                "analyzed": True,
                "models_used": list(state["analysis_results"].keys())
            }

        def reduce_any(state: dict[str, Any], result: Any) -> None:
            obs = str(getattr(result, "output", result))[:800]
            state["observation"] = obs

        reducers = {
            "smart_analyze_tool": reduce_smart_analyze,
        }
        for name in tools_dict:
            if name not in reducers:
                reducers[name] = reduce_any

        self._tool_executor = ToolExecutorNode(tools=tools_dict, reducers=reducers, default_timeout_s=180)

        # LLM 推理节点
        tool_names = list(self._tool_executor.tools.keys())
        decision_tool = make_deep_analyzer_decision_tool(tool_names=tool_names)
        
        system_prompt = (
            "你是深度分析控制器（重构版）。决定下一步操作。\n"
            "工作流指导：\n"
            "1. 首先调用 'smart_analyze_tool' 对输入内容进行深入分析。\n"
            "2. 分析完成后，路由到 'summarize' 节点来汇总结果并生成最终报告。\n"
            "3. 只有在完成总结后，任务才算正式结束。\n"
        )
        
        self._reason_node = LLMReasoningNode(
            model=self.model,
            system_prompt=system_prompt,
            decision_tool=decision_tool,
            tool_catalog_text=_tool_catalog_text(tools_dict) + "\n\nAvailable nodes:\n- summarize",
        )

        super().__init__(
            name=name or self.config.get("name") or "deep_analyzer_agent",
            description=description or self.config.get("description") or "Deep analyzer (Refactored)",
            max_steps=max_steps or int(self.config.get("max_steps", 20)),
            time_limit_seconds=int(self.config.get("time_limit_seconds", 300)),
        )

    def _initial_state(self, task: str) -> dict[str, Any]:
        state = super()._initial_state(task)
        state.update({
            "task": task,
            "source": None,
            "analysis_results": {},
            "summary": None,
            "analyzed_once": False,
        })
        return state

    def _route_from_decision(self, state: dict[str, Any]) -> str:
        if state.get("is_final"):
            return "end"

        if state.get("error") and state.get("stop_on_error", False):
            state["is_final"] = True
            state["final_answer"] = f"Failed: {state['error']}"
            return "end"

        decision = state.get("decision") or {}
        route = decision.get("route")

        if route == "finish":
            state["is_final"] = True
            state["final_answer"] = decision.get("final_answer") or state.get("summary") or "Done."
            return "end"

        if route in ("call_tool", "call_tools_parallel"):
            if route == "call_tools_parallel":
                # 由于分析代理通常单步，暂不复杂化 parallel 逻辑，映射到 tool_executor 即可
                return "tool_executor"
                
            tool_name = decision.get("tool_name")
            if not tool_name:
                if not state.get("analyzed_once"):
                    tool_name = "smart_analyze_tool"
            
            if tool_name:
                if not decision.get("tool_args"):
                    if tool_name == "smart_analyze_tool":
                         state["decision"]["tool_args"] = {
                             "task": state.get("task"),
                             "source": state.get("source") or state.get("task", "")
                         }
                return "tool_executor"

        if route == "call_llm_node":
            node = decision.get("llm_node")
            if node == "summarize":
                return "summarize"

        # 默认：如果已分析但由于某种原因没路由，自动汇总
        if state.get("analyzed_once") and not state.get("summary"):
            return "summarize"
            
        return "end"

    async def _summarize_node(self, state: dict[str, Any]) -> dict[str, Any]:
        # 调用专门的总结工具
        res = await self._summary_tool.forward(
            model_analyses=state.get("analysis_results", {})
        )
        if res.error:
            state["error"] = res.error
            state["final_answer"] = f"Summarization failed: {res.error}"
        else:
            summary = res.output or str(res)
            state["summary"] = summary
            state["final_answer"] = summary
            state["observation"] = "Summarization complete."
            
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



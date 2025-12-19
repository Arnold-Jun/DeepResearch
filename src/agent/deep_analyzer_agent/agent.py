from __future__ import annotations

from typing import Any

import json5

from langgraph.graph import END, StateGraph

from src.models.base import Model
from src.registry import AGENT
from src.tools.analysis.analysis_tools import (
    AnalyzeTool,
    ParallelAnalyzeTool,
    SummarizeAnalysisTool,
)

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


@AGENT.register_module(name="deep_analyzer_agent", force=True)
class DeepAnalyzerGraphAgent(BaseGraphAgent):
    """基于 LangGraph 的深度分析子代理。"""

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

        self._analyze_tool = AnalyzeTool(model_id=self.model_id)
        self._parallel_analyze_tool = ParallelAnalyzeTool(analyzer_model_ids=analyzer_model_ids)
        self._summarize_tool = SummarizeAnalysisTool(summarizer_model_id=self.summarizer_model_id)

        def reduce_any(state: dict[str, Any], result: Any) -> None:
            obs = state.get("observation", "")
            state["state_summary"] = {
                "last_tool": state.get("last_tool_name"),
                "observation_preview": obs[:800] if isinstance(obs, str) else str(obs)[:800],
            }

        reducers = {name: reduce_any for name in tools_dict.keys()}
        self._tool_executor = ToolExecutorNode(tools=tools_dict, reducers=reducers, default_timeout_s=120)

        tool_names = list(self._tool_executor.tools.keys())
        decision_tool = make_deep_analyzer_decision_tool(tool_names=tool_names)
        node_names = [
            "analyze",
            "parallel_analyze",
            "summarize",
        ]
        system_prompt = (
            "你是深度分析控制器。决定下一步操作。"
            "你可以调用确定性工具（例如 python_interpreter_tool），"
            "或路由到内部节点之一进行固定 LLM 转换（分析/并行分析/总结），"
            "或在有足够结果时完成。"
        )
        self._reason_node = LLMReasoningNode(
            model=self.model,
            system_prompt=system_prompt,
            decision_tool=decision_tool,
            tool_catalog_text=_tool_catalog_text(tools_dict) + "\n\nAvailable nodes:\n- " + "\n- ".join(node_names),
        )

        super().__init__(
            name=name or self.config.get("name") or "deep_analyzer_agent",
            description=description or self.config.get("description") or "Deep analyzer (LangGraph)",
            max_steps=max_steps or int(self.config.get("max_steps", 20)),
            time_limit_seconds=int(self.config.get("time_limit_seconds", 180)),
        )

    def _initial_state(self, task: str) -> dict[str, Any]:
        state = super()._initial_state(task)
        state.update({
            "task": task,
            "source": None,
            "analysis_results": {},  # 存储各模型的分析结果
            "summary": None,
            "analyzed_once": False,
            "need_analyze": True,
            "need_summarize": False,
        })
        return state

    async def _analyze_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行单个分析节点"""
        task = state.get("task")
        source = state.get("source")
        res = await self._analyze_tool.forward(task=task, source=source)
        analysis = res.output or ""
        state["observation"] = analysis
        state["error"] = res.error
        state["analyzed_once"] = True
        state["need_analyze"] = False
        state["need_summarize"] = True
        return state

    async def _parallel_analyze_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """并行执行多个模型的分析节点"""
        task = state.get("task")
        source = state.get("source")
        res = await self._parallel_analyze_tool.forward(task=task, source=source)
        analysis_results = res.output or {}
        if isinstance(analysis_results, dict):
            state["analysis_results"] = analysis_results
            # 构建观察结果
            obs_parts = []
            for model_name, analysis in analysis_results.items():
                obs_parts.append(f"{model_name}:\n{analysis}\n")
            state["observation"] = "\n".join(obs_parts)
        else:
            state["observation"] = str(analysis_results)
        state["error"] = res.error
        state["analyzed_once"] = True
        state["need_analyze"] = False
        state["need_summarize"] = True
        return state

    async def _summarize_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """总结分析结果节点"""
        analysis_results = state.get("analysis_results", {})
        if not analysis_results:
            obs = state.get("observation", "")
            if obs:
                analysis_results = {"default": obs}
        
        res = await self._summarize_tool.forward(model_analyses=analysis_results)
        summary = res.output or ""
        state["summary"] = summary
        state["final_answer"] = summary
        state["observation"] = summary
        state["error"] = res.error
        state["is_final"] = True
        return state

    async def _finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error"):
            state["final_answer"] = f"Failed: {state['error']}"
        else:
            state["final_answer"] = state.get("observation") or state.get("final_answer") or "Done."
        state["is_final"] = True
        return state

    def _parse_decision_json(self, json_str: str) -> dict[str, Any]:
        """解析决策 JSON，处理截断和不完整的 JSON"""
        import re
        from src.logger import logger
        
        # 首先尝试直接解析
        try:
            return json5.loads(json_str)
        except Exception:
            pass
        
        # 尝试提取 JSON 对象
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_candidate = json_match.group()
            try:
                return json5.loads(json_candidate)
            except Exception:
                # 尝试修复常见的 JSON 错误
                json_candidate = self._fix_incomplete_json(json_candidate)
                try:
                    return json5.loads(json_candidate)
                except Exception:
                    pass
        
        # 如果仍然失败，尝试提取关键字段
        logger.warning(f"[DeepAnalyzerAgent] Failed to parse decision JSON, attempting to extract key fields")
        logger.debug(f"[DeepAnalyzerAgent] Original JSON string (first 500 chars): {json_str[:500]}")
        
        # 尝试提取关键字段
        extracted = {}
        route_match = re.search(r'"route"\s*:\s*"([^"]+)"', json_str)
        if route_match:
            extracted["route"] = route_match.group(1)
        
        tool_name_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', json_str)
        if tool_name_match:
            extracted["tool_name"] = tool_name_match.group(1)
        
        # 尝试提取 tool_args
        tool_args_match = re.search(r'"tool_args"\s*:\s*(\{.*?\})', json_str, re.DOTALL)
        if tool_args_match:
            try:
                tool_args_str = tool_args_match.group(1)
                tool_args_str = self._fix_incomplete_json(tool_args_str)
                extracted["tool_args"] = json5.loads(tool_args_str)
            except Exception:
                pass
        
        if extracted:
            logger.info(f"[DeepAnalyzerAgent] Extracted partial decision: {extracted}")
            return extracted
        
        # 如果完全无法解析，返回空字典
        logger.error(f"[DeepAnalyzerAgent] Completely failed to parse decision JSON")
        return {}
    
    def _fix_incomplete_json(self, json_str: str) -> str:
        """尝试修复不完整的 JSON"""
        json_str = json_str.rstrip().rstrip(',')
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        json_str += '}' * (open_braces - close_braces)
        json_str += ']' * (open_brackets - close_brackets)
        
        if json_str.count('"') % 2 != 0:
            last_quote_idx = json_str.rfind('"')
            if last_quote_idx > 0 and json_str[last_quote_idx - 1] != '\\':
                before_quote = json_str[:last_quote_idx].rstrip()
                if before_quote.endswith(':') or before_quote.endswith(','):
                    json_str = json_str[:last_quote_idx] + json_str[last_quote_idx + 1:]
        
        return json_str

    def _route_from_decision(self, state: dict[str, Any]) -> str:
        """每代理路由器：确定性地将 LLM 决策映射到图节点。"""
        if state.get("is_final"):
            return "end"

        # 致命错误处理
        if state.get("error") and state.get("stop_on_error", False):
            state["is_final"] = True
            state["final_answer"] = f"Failed: {state['error']}"
            return "end"

        decision = state.get("decision") or {}
        if not isinstance(decision, dict):
            if isinstance(decision, str):
                from src.logger import logger
                logger.debug(f"[DeepAnalyzerAgent] Decision is string, parsing: {decision[:200]}...")
                decision = self._parse_decision_json(decision)
            else:
                from src.logger import logger
                logger.warning(f"[DeepAnalyzerAgent] Decision has unexpected type: {type(decision)}, value: {decision}")
                decision = {}
        
        from src.logger import logger
        logger.info(f"[DeepAnalyzerAgent] Parsed decision: {decision}")

        route = decision.get("route")
        logger.info(f"[DeepAnalyzerAgent] Route from decision: {route!r} (type: {type(route).__name__})")
        logger.info(f"[DeepAnalyzerAgent] Full decision content: {decision}")
        
        if route == "finish":
            state["is_final"] = True
            state["final_answer"] = decision.get("final_answer") or state.get("observation") or "Done."
            return "end"

        route_str = str(route).strip() if route else None
        logger.info(f"[DeepAnalyzerAgent] Normalized route: {route_str!r}")
        
        if route_str in ("call_tool", "call_tools_parallel"):
            if route_str == "call_tools_parallel":
                tool_calls = decision.get("tool_calls")
                if not (isinstance(tool_calls, list) and tool_calls):
                    route = "call_tool"
                else:
                    normalized = []
                    tools = list(self._tool_executor.tools.keys())
                    default_tool = tools[0] if tools else None
                    for c in tool_calls:
                        if not isinstance(c, dict):
                            continue
                        tn = c.get("tool_name") or default_tool
                        if not tn:
                            continue
                        ta = c.get("tool_args")
                        if not isinstance(ta, dict):
                            ta = {}
                        call = {"tool_name": tn, "tool_args": ta}
                        if c.get("timeout_s") is not None:
                            call["timeout_s"] = c.get("timeout_s")
                        normalized.append(call)
                    if normalized:
                        state["tool_calls"] = normalized
                        return "tool_executor"
                    route_str = "call_tool"

            tool_name = decision.get("tool_name")
            from src.logger import logger
            logger.info(f"[DeepAnalyzerAgent] Route is call_tool, tool_name from decision: {tool_name!r}")
            if not tool_name:
                tools = list(self._tool_executor.tools.keys())
                tool_name = tools[0] if tools else None
                logger.info(f"[DeepAnalyzerAgent] tool_name not in decision, using default: {tool_name!r}")
            if not tool_name:
                state["is_final"] = True
                state["final_answer"] = "Failed: no tools configured."
                return "end"
            logger.info(f"[DeepAnalyzerAgent] Routing to tool_executor, tool_name={tool_name!r}")
            return "tool_executor"

        if route_str == "call_llm_node":
            node = decision.get("llm_node")
            allowed = {
                "analyze",
                "parallel_analyze",
                "summarize",
            }
            if node in allowed:
                return node

        from src.logger import logger
        logger.warning(f"[DeepAnalyzerAgent] Route '{route_str}' (original: {route!r}) not recognized, using fallback logic. Decision was: {decision}")
        if not state.get("analyzed_once"):
            return "parallel_analyze"
        if state.get("need_summarize"):
            return "summarize"
        return "summarize"

    def _build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("tool_executor", self._tool_executor)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("parallel_analyze", self._parallel_analyze_node)
        workflow.add_node("summarize", self._summarize_node)

        workflow.set_entry_point("reason")
        workflow.add_conditional_edges(
            "reason",
            self._route_from_decision,
            {
                "tool_executor": "tool_executor",
                "analyze": "analyze",
                "parallel_analyze": "parallel_analyze",
                "summarize": "summarize",
                "end": END,
            },
        )

        workflow.add_edge("tool_executor", "reason")
        workflow.add_edge("analyze", "reason")
        workflow.add_edge("parallel_analyze", "reason")
        workflow.add_edge("summarize", END)

        return workflow.compile()



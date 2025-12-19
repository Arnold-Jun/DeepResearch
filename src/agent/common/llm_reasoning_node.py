from __future__ import annotations

import time
from typing import Any

import json5

from src.logger import logger
from src.models import ChatMessage
from src.models.base import MessageRole, Model


class LLMReasoningNode:
    """
    Central LLM reasoning node.

    It produces a structured routing decision via tool-calling against a schema-only Tool.
    """

    def __init__(
        self,
        *,
        model: Model,
        system_prompt: str,
        decision_tool: Any,
        tool_catalog_text: str,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.decision_tool = decision_tool
        self.tool_catalog_text = tool_catalog_text

    def _parse_decision_json(self, json_str: str) -> dict[str, Any]:
        """解析决策 JSON，处理截断和不完整的 JSON"""
        import re
        
        try:
            return json5.loads(json_str)
        except Exception:
            pass
        
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_candidate = json_match.group()
            try:
                return json5.loads(json_candidate)
            except Exception:
                json_candidate = self._fix_incomplete_json(json_candidate)
                try:
                    return json5.loads(json_candidate)
                except Exception:
                    pass
        
        logger.warning(f"[LLMReasoningNode] Failed to parse decision JSON, attempting to extract key fields")
        logger.debug(f"[LLMReasoningNode] Original JSON string (first 500 chars): {json_str[:500]}")
        
        extracted = {}
        route_match = re.search(r'"route"\s*:\s*"([^"]+)"', json_str)
        if route_match:
            extracted["route"] = route_match.group(1)
        
        tool_name_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', json_str)
        if tool_name_match:
            extracted["tool_name"] = tool_name_match.group(1)
        
        tool_args_match = re.search(r'"tool_args"\s*:\s*(\{.*?\})', json_str, re.DOTALL)
        if tool_args_match:
            try:
                tool_args_str = tool_args_match.group(1)
                tool_args_str = self._fix_incomplete_json(tool_args_str)
                extracted["tool_args"] = json5.loads(tool_args_str)
            except Exception:
                pass
        
        if extracted:
            logger.info(f"[LLMReasoningNode] Extracted partial decision: {extracted}")
            return extracted
        
        logger.error(f"[LLMReasoningNode] Completely failed to parse decision JSON")
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

    def _build_user_prompt(self, state: dict[str, Any]) -> str:
        task = state.get("task", "")
        step = state.get("step", 0)
        max_steps = state.get("max_steps", 20)

        context_lines = [
            f"Task: {task}",
            f"Step: {step}/{max_steps}",
        ]

        observation = state.get("observation")
        if observation and observation not in ("", "No result"):
            obs_preview = str(observation)
            if len(obs_preview) > 1200:
                obs_preview = obs_preview[:1200] + "\n...[结果已截断]..."
            context_lines.append(f"\nLast observation (most recent result):\n{obs_preview}")

        tool_results = state.get("tool_results", [])
        if isinstance(tool_results, list) and tool_results:
            context_lines.append("\nTool execution history:")
            for tr in tool_results[-3:]:
                tool_name = tr.get("tool_name", "unknown")
                result = tr.get("result", "")
                if result:
                    result_preview = str(result)
                    if len(result_preview) > 800:
                        result_preview = result_preview[:800] + "\n...[结果已截断]..."
                    context_lines.append(f"- {tool_name}: {result_preview}")
        trace = state.get("trace")
        if isinstance(trace, list) and trace:
            context_lines.append("\nRecent tool calls (most recent last):")
            for entry in trace[-8:]:
                if not isinstance(entry, dict):
                    continue
                tool_name = entry.get("tool_name")
                ok = entry.get("ok")
                err = entry.get("error")
                obs = entry.get("observation_preview")
                args = entry.get("tool_args")
                line = f"- {tool_name} ok={ok}"
                if err:
                    line += f" error={err}"
                context_lines.append(line)
                if args is not None:
                    args_str = str(args)
                    if len(args_str) > 800:
                        args_str = args_str[:800] + "\n...[args truncated]..."
                    context_lines.append(f"  args: {args_str}")
                if obs:
                    obs_str = str(obs)
                    if len(obs_str) > 1200:
                        obs_str = obs_str[:1200] + "\n...[obs truncated]..."
                    context_lines.append(f"  obs: {obs_str}")
        summary = state.get("state_summary")
        if summary:
            context_lines.append("\nState summary:\n" + str(summary)[:2000])

        context_lines.append("\nAvailable tools:\n" + self.tool_catalog_text)
        context_lines.append(
            "\nDecide the next step using the decision tool schema. "
            "If you have enough information, finish with a final_answer."
        )
        return "\n".join(context_lines)

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("is_final"):
            return state
        if state.get("error") and state.get("stop_on_error", False):
            state["is_final"] = True
            state["final_answer"] = f"Failed: {state['error']}"
            return state

        step = int(state.get("step", 0)) + 1
        state["step"] = step

        if step > int(state.get("max_steps", 20)):
            if state.get("final_answer"):
                state["is_final"] = True
                return state
            insights = state.get("insights", [])
            observation = state.get("observation")
            if insights or (observation and observation not in ("", "No result")):
                state["_needs_summary_before_finish"] = True
                return state
            state["is_final"] = True
            state["final_answer"] = "Reached max steps."
            return state

        deadline = state.get("deadline")
        if deadline and time.time() >= float(deadline):
            state["is_final"] = True
            state["final_answer"] = state.get("final_answer") or "Reached time limit."
            return state

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            ChatMessage(role=MessageRole.USER, content=self._build_user_prompt(state)),
        ]

        try:
            response = await self.model(messages, tools_to_call_from=[self.decision_tool])
            if not getattr(response, "tool_calls", None):
                try:
                    response = self.model.parse_tool_calls(response)
                except Exception:
                    pass

            if not response.tool_calls:
                state["error"] = "LLM did not return a tool call decision."
                return state

            decision = response.tool_calls[0].function.arguments
            if isinstance(decision, str):
                logger.debug(f"[LLMReasoningNode] Decision is string, parsing: {decision[:200]}...")
                decision = self._parse_decision_json(decision)
            elif isinstance(decision, dict):
                logger.debug(f"[LLMReasoningNode] Decision is already dict: {decision}")
            else:
                logger.warning(f"[LLMReasoningNode] Decision has unexpected type: {type(decision)}, value: {decision}")
                if hasattr(decision, '__dict__'):
                    decision = decision.__dict__
                else:
                    decision = {}
            
            if not isinstance(decision, dict):
                logger.error(f"[LLMReasoningNode] Failed to parse decision as dict, got: {type(decision)}")
                decision = {}
            
            logger.debug(f"[LLMReasoningNode] Final decision: {decision}")
            state["decision"] = decision
            state["error"] = None
            return state
        except Exception as e:
            logger.error(f"[LLMReasoningNode] decision failed: {type(e).__name__}: {e}")
            state["error"] = str(e)
            return state



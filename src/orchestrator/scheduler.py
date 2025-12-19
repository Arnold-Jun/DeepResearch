from __future__ import annotations

from typing import Any

import json5

from src.models import ChatMessage
from src.models.base import MessageRole, Model
from src.orchestrator.llm_tools import RoundPlanTool
from src.orchestrator.state import OrchestratorState, TodoItem


def _format_parent_subtasks(todo_list: list[TodoItem], parent_id: str) -> str:
    """
    格式化父任务的所有子任务，包括子任务的返回结果。
    不显示子任务的状态（因为只对父任务定义状态）。
    """
    subtasks = [item for item in todo_list if item.parent_id == parent_id]
    if not subtasks:
        return "（无子任务）"
    
    lines: list[str] = []
    for idx, st in enumerate(subtasks, 1):
        lines.append(f"{idx}. id={st.id} task={st.task[:150]!r}")
        
        # 显示返回结果（优先显示成功结果，如果有错误则显示错误）
        if st.last_result_summary:
            # 截断过长的结果
            result_preview = st.last_result_summary
            if len(result_preview) > 800:
                result_preview = result_preview[:800] + "\n...[结果已截断]..."
            lines.append(f"   返回结果：{result_preview}")
        elif st.last_error:
            error_preview = st.last_error
            if len(error_preview) > 400:
                error_preview = error_preview[:400] + "...[错误信息已截断]"
            lines.append(f"   错误：{error_preview}")
        else:
            # 既没有结果也没有错误，可能是还未执行或执行中
            lines.append(f"   （暂无返回结果）")
    
    return "\n".join(lines)


class SchedulerLLM:
    def __init__(self, *, model: Model, max_parallelism: int):
        self.model = model
        self.tool = RoundPlanTool()
        self.max_parallelism = max_parallelism

    async def schedule(self, state: OrchestratorState, parent: TodoItem) -> dict[str, Any]:
        system = (
            "你是调度器模块（SCHEDULER）。\n"
            "给定一个选中的父任务待办项，决定如何执行下一轮。\n"
            "- 你必须为整个轮次选择恰好一个 agent_type。\n"
            "- 你必须为该父任务生成 N 个子任务（并行任务）。\n"
            "- N 必须 <= max_parallelism。\n"
            "- 所有子任务的 parent_id 必须等于所选父任务的 id。\n"
            "- 注意：task_id 将由系统自动生成，你只需要提供 parent_id 和 task 字段。\n"
            "- 如果你可以完成整体用户任务，可以设置 route=finish，系统会自动调用总结模块生成最终答案。\n"
            "\n"
            "失败重试机制：\n"
            "- 如果父任务的 failures 计数 > 0，说明之前的子任务执行失败率过高。\n"
            "- 当 failures < max_failures_per_parent 时，你需要重新拆解任务，生成新的子任务。\n"
            "- 可以尝试不同的拆解方式、不同的 agent_type，或者更细粒度的子任务。\n"
            "- 如果 failures >= max_failures_per_parent，系统将不再选择该父任务，你也不会被调用。\n"
            "\n"
            "Agent 类型选择指导：\n"
            "- browser_use_agent：优先用于搜索相关网页、浏览网站、与网页交互的任务。"
            "  适合：网页搜索、浏览特定网站、提取网页内容、与网页元素交互等。\n"
            "- deep_researcher_agent：用于深度研究、广泛的网络搜索和学术研究。"
            "  适合：需要多轮深度搜索、学术论文查找、广泛信息收集等。\n"
            "- deep_analyzer_agent：用于系统性分析、逐步推理和结构化分析。"
            "  适合：数据分析、逻辑推理、结构化分析、逐步解决问题等。\n"
            "\n"
            "重要：对于搜索任务，优先选择 browser_use_agent。"
            "只有在需要深度研究或多轮广泛搜索时，才考虑使用 deep_researcher_agent。\n"
            "\n"
            "重要：你必须使用 round_plan 工具来输出你的响应。"
            "不要输出纯文本。始终使用 round_plan 工具调用你的计划。\n"
            "\n"
            "重要：你返回的工具调用参数必须是有效的 JSON 格式。确保所有括号、引号都正确匹配。"
        )
        # 格式化该父任务的所有历史子任务
        existing_subtasks_text = _format_parent_subtasks(state.todo_list, parent.id)
        
        user = (
            f"用户任务：\n{state.task}\n\n"
            f"轮次：{state.round_index}\n"
            f"最大并行数={self.max_parallelism}\n\n"
            f"选中的父任务待办：\n"
            f"- id={parent.id}\n"
            f"- status={parent.status}\n"
            f"- failures={parent.failure_count}\n"
            f"- task={parent.task}\n"
            f"- last_result_summary={parent.last_result_summary or '（无）'}\n\n"
            f"已存在的子任务（历史拆解的子任务及其返回结果）：\n{existing_subtasks_text}\n\n"
            f"上一轮汇总：\n{state.last_round_summary or '（无）'}\n\n"
            "现在生成下一轮计划。"
            "你可以基于已有的子任务结果决定是否需要生成新的子任务，或者完成该父任务。"
            "注意：在 tasks 数组中，每个任务只需要提供 parent_id 和 task 字段，task_id 会由系统自动生成。"
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system),
            ChatMessage(role=MessageRole.USER, content=user),
        ]

        from src.logger import logger
        max_retries = 2  # 最多重试2次
        resp = None
        
        for attempt in range(max_retries + 1):
            try:
                # 显式要求模型必须使用工具调用
                resp = await self.model(messages, tools_to_call_from=[self.tool], tool_choice="required")
                
                # 记录原始响应以便调试
                logger.debug(f"[SchedulerLLM] Attempt {attempt + 1}: role={resp.role}, has_tool_calls={bool(resp.tool_calls)}, content_preview={str(resp.content)[:200] if resp.content else None}")
                
                if not getattr(resp, "tool_calls", None):
                    try:
                        resp = self.model.parse_tool_calls(resp)
                        logger.debug(f"[SchedulerLLM] After parse_tool_calls: has_tool_calls={bool(resp.tool_calls)}")
                    except Exception as e:
                        logger.warning(f"[SchedulerLLM] parse_tool_calls failed: {type(e).__name__}: {e}")
                        logger.warning(f"[SchedulerLLM] Response content: {resp.content}")
                        pass
                
                if not resp.tool_calls:
                    if resp.content:
                        try:
                            import re
                            json_match = re.search(r'\{[^{}]*"route"[^{}]*\}', resp.content, re.DOTALL)
                            if json_match:
                                extracted = json5.loads(json_match.group())
                                logger.info("[SchedulerLLM] Extracted tool call from content text")
                                return extracted
                        except Exception as e:
                            logger.warning(f"[SchedulerLLM] Failed to extract JSON from content: {e}")
                    
                    if attempt < max_retries:
                        error_msg = (
                            f"你的上一次响应中没有找到工具调用。\n"
                            f"响应内容：{resp.content[:500] if resp.content else '（无内容）'}\n\n"
                            f"请重新调用 round_plan 工具，确保使用工具调用的方式返回你的计划。"
                        )
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                        messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                        logger.info(f"[SchedulerLLM] Retrying due to no tool call (attempt {attempt + 1}/{max_retries + 1})")
                        continue
                    else:
                        logger.error(f"[SchedulerLLM] No tool call found after {max_retries + 1} attempts. Response content: {resp.content}")
                        raise RuntimeError(f"SchedulerLLM did not return a tool call. Response: {resp.content}")
                
                args = resp.tool_calls[0].function.arguments
                if isinstance(args, str):
                    try:
                        args = json5.loads(args)
                    except Exception as e:
                        logger.warning(f"[SchedulerLLM] Failed to parse tool arguments as JSON: {e}")
                        import re
                        json_match = re.search(r'\{.*\}', args, re.DOTALL)
                        if json_match:
                            try:
                                args = json5.loads(json_match.group())
                            except Exception as parse_error:
                                if attempt < max_retries:
                                    error_msg = (
                                        f"你的上一次响应中的工具调用参数格式不正确，无法解析为 JSON。\n"
                                        f"错误信息：{str(parse_error)}\n"
                                        f"你返回的参数内容：{args[:500]}\n\n"
                                        f"请重新调用 round_plan 工具，确保返回的参数是有效的 JSON 格式。"
                                        f"特别注意：检查所有括号、引号、逗号是否正确匹配。"
                                    )
                                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                                    messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                                    logger.info(f"[SchedulerLLM] Retrying due to JSON parse error (attempt {attempt + 1}/{max_retries + 1})")
                                    continue
                                else:
                                    raise ValueError(f"Cannot parse tool arguments after {max_retries + 1} attempts: {args}")
                        else:
                            # 无法提取 JSON，准备重试
                            if attempt < max_retries:
                                error_msg = (
                                    f"你的上一次响应中的工具调用参数格式不正确，无法找到有效的 JSON 内容。\n"
                                    f"你返回的参数内容：{args[:500]}\n\n"
                                    f"请重新调用 round_plan 工具，确保返回的参数是有效的 JSON 格式。"
                                )
                                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                                messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                                logger.info(f"[SchedulerLLM] Retrying due to invalid JSON format (attempt {attempt + 1}/{max_retries + 1})")
                                continue
                            else:
                                raise ValueError(f"Cannot parse tool arguments: {args}")
                
                if not isinstance(args, dict):
                    if attempt < max_retries:
                        error_msg = (
                            f"你的上一次响应中的工具调用参数类型不正确。期望是字典（dict），但得到：{type(args).__name__}\n"
                            f"参数内容：{str(args)[:500]}\n\n"
                            f"请重新调用 round_plan 工具，确保返回的参数是有效的 JSON 对象（字典）。"
                        )
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                        messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                        logger.info(f"[SchedulerLLM] Retrying due to wrong argument type (attempt {attempt + 1}/{max_retries + 1})")
                        continue
                    else:
                        raise TypeError(f"SchedulerLLM tool arguments are not a dict. Got: {type(args)}, value: {args}")
                
                return args
                
            except (ValueError, TypeError, RuntimeError) as e:
                if attempt >= max_retries:
                    logger.error(f"[SchedulerLLM] Failed after {max_retries + 1} attempts: {e}")
                    raise
                error_msg = (
                    f"你的上一次响应出现了错误：{str(e)}\n"
                    f"请重新调用 round_plan 工具，确保返回的参数格式正确。"
                )
                if resp:
                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                logger.info(f"[SchedulerLLM] Retrying due to error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                continue
        
        logger.error(f"[SchedulerLLM] Unexpected end of retry loop")
        raise RuntimeError("SchedulerLLM failed after all retry attempts")



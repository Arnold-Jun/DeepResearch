from __future__ import annotations

from typing import Any

import json5

from src.models import ChatMessage
from src.models.base import MessageRole, Model
from src.orchestrator.llm_tools import TodoDeltaTool
from src.orchestrator.state import OrchestratorState, TodoItem


def _format_todo_list(todo_list: list[TodoItem], *, include_subtasks: bool = True) -> str:
    """格式化待办列表，只显示父任务，不显示子任务状态"""
    parent_items = [item for item in todo_list if item.is_parent]
    if not parent_items:
        return "(empty)"
    
    lines: list[str] = []
    for idx, item in enumerate(parent_items, 1):
        lines.append(
            f"{idx}. id={item.id} status={item.status} "
            f"failures={item.failure_count} task={item.task[:200]!r}"
        )
    return "\n".join(lines)


def _format_parent_subtask_summary(todo_list: list[TodoItem], parent_id: str) -> str:
    """格式化父任务的子任务执行结果摘要（不显示状态，只显示结果）"""
    subtasks = [item for item in todo_list if item.parent_id == parent_id]
    if not subtasks:
        return "（无子任务）"
    
    total = len(subtasks)
    succeeded = len([st for st in subtasks if st.status == "done" and st.last_result_summary])
    failed = len([st for st in subtasks if st.status == "failed" or (st.status == "done" and not st.last_result_summary)])
    
    lines: list[str] = []
    lines.append(f"子任务统计：总数={total}, 成功={succeeded}, 失败={failed}")
    
    if succeeded > 0:
        lines.append("成功子任务的结果摘要：")
        for st in subtasks:
            if st.status == "done" and st.last_result_summary:
                result_preview = st.last_result_summary
                if len(result_preview) > 600:
                    result_preview = result_preview[:600] + "\n...[结果已截断]..."
                lines.append(f"  - {st.task[:100]}: {result_preview}")
    
    if failed > 0:
        lines.append("失败子任务：")
        for st in subtasks:
            if st.status == "failed" or (st.status == "done" and not st.last_result_summary):
                error_info = st.last_error or "无错误信息"
                if len(error_info) > 300:
                    error_info = error_info[:300] + "...[错误信息已截断]"
                lines.append(f"  - {st.task[:100]}: {error_info}")
    
    return "\n".join(lines)


class PlanningLLM:
    def __init__(self, *, model: Model):
        self.model = model
        self.tool = TodoDeltaTool()

    async def plan(self, state: OrchestratorState) -> dict[str, Any]:
        system = (
            "你是规划模块（PLANNING）。你的工作是维护一个有序的父任务（PARENT）待办列表。\n"
            "- 你可以添加/更新/删除父任务。\n"
            "- 你不得重新排序任务。\n"
            "- 你不得更新子任务（parent_id != null 的项目）。\n"
            "- 你不需要生成最终答案，最终答案将由专门的总结模块生成。\n"
            "- 当所有父任务都已完成时，你可以不再添加新任务，系统会自动调用总结模块。\n"
            "\n"
            "重要规则 - 何时将父任务标记为 done：\n"
            "- 只有当父任务的子任务成功率满足要求时，才能将父任务标记为 done。\n"
            "- 子任务成功率 = 成功子任务数 / 总子任务数\n"
            "- 只有当成功率 >= (1 - 失败率阈值) 时，才能将父任务标记为 done。\n"
            "- 如果父任务还没有子任务，或者子任务还在执行中，则不能将父任务标记为 done。\n"
            "- 不要在没有执行记录的情况下将任务标记为 done。\n"
            "\n"
            "重要：你必须使用 todo_delta 工具来输出你的响应。"
            "不要输出纯文本。始终使用 todo_delta 工具调用你的操作。\n"
            "如果待办列表为空，你必须使用 todo_delta 工具添加至少一个父任务。\n"
            "\n"
            "重要：你返回的工具调用参数必须是有效的 JSON 格式。确保所有括号、引号都正确匹配。"
        )
        todo_list_text = _format_todo_list(state.todo_list, include_subtasks=False)
        
        last_parent_summary = ""
        if state.last_round_parent_id:
            last_parent_summary = f"\n上一轮执行的父任务（id={state.last_round_parent_id}）的子任务执行结果：\n{_format_parent_subtask_summary(state.todo_list, state.last_round_parent_id)}\n"
        
        user = (
            f"用户任务：\n{state.task}\n\n"
            f"轮次：{state.round_index}\n"
            f"上一轮汇总：\n{state.last_round_summary or '（无）'}\n"
            f"{last_parent_summary}"
            f"当前父任务待办列表：\n{todo_list_text}\n\n"
            "你现在必须调用 todo_delta 工具来更新父任务待办列表。"
            "如果列表为空，添加初始父任务。"
            "\n"
            "重要：在标记父任务为 done 之前，请检查：\n"
            "- 该父任务是否已经有子任务？\n"
            "- 子任务成功率是否满足要求（成功率 >= 1 - 失败率阈值）？\n"
            "- 如果父任务还没有子任务，或者子任务成功率不满足要求，则不能将其标记为 done。\n"
            "\n"
            "如果所有父任务都已完成（子任务成功率满足要求），你可以不再添加新任务，系统会自动调用总结模块生成最终答案。"
            "不要用纯文本回复 - 使用工具调用。"
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
                resp = await self.model(messages, tools_to_call_from=[self.tool], tool_choice="required")
                
                # 记录原始响应以便调试
                logger.debug(f"[PlanningLLM] Attempt {attempt + 1}: role={resp.role}, has_tool_calls={bool(resp.tool_calls)}, content_preview={str(resp.content)[:200] if resp.content else None}")
                
                if not getattr(resp, "tool_calls", None):
                    try:
                        resp = self.model.parse_tool_calls(resp)
                        logger.debug(f"[PlanningLLM] After parse_tool_calls: has_tool_calls={bool(resp.tool_calls)}")
                    except Exception as e:
                        logger.warning(f"[PlanningLLM] parse_tool_calls failed: {type(e).__name__}: {e}")
                        logger.warning(f"[PlanningLLM] Response content: {resp.content}")
                        pass
                
                if not resp.tool_calls:
                    if resp.content:
                        try:
                            import re
                            json_match = re.search(r'\{[^{}]*"actions"[^{}]*\}', resp.content, re.DOTALL)
                            if json_match:
                                extracted = json5.loads(json_match.group())
                                logger.info("[PlanningLLM] Extracted tool call from content text")
                                return extracted
                        except Exception as e:
                            logger.warning(f"[PlanningLLM] Failed to extract JSON from content: {e}")
                    
                    logger.error(f"[PlanningLLM] No tool call found. Response content: {resp.content}")
                    return {
                        "actions": []
                    }
                
                args = resp.tool_calls[0].function.arguments
                if isinstance(args, str):
                    try:
                        args = json5.loads(args)
                    except Exception as e:
                        logger.warning(f"[PlanningLLM] Failed to parse tool arguments as JSON: {e}")
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
                                        f"请重新调用 todo_delta 工具，确保返回的参数是有效的 JSON 格式。"
                                        f"特别注意：检查所有括号、引号、逗号是否正确匹配。"
                                    )
                                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                                    messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                                    logger.info(f"[PlanningLLM] Retrying due to JSON parse error (attempt {attempt + 1}/{max_retries + 1})")
                                    continue
                                else:
                                    raise ValueError(f"Cannot parse tool arguments after {max_retries + 1} attempts: {args}")
                        else:
                            # 无法提取 JSON，准备重试
                            if attempt < max_retries:
                                error_msg = (
                                    f"你的上一次响应中的工具调用参数格式不正确，无法找到有效的 JSON 内容。\n"
                                    f"你返回的参数内容：{args[:500]}\n\n"
                                    f"请重新调用 todo_delta 工具，确保返回的参数是有效的 JSON 格式。"
                                )
                                messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                                messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                                logger.info(f"[PlanningLLM] Retrying due to invalid JSON format (attempt {attempt + 1}/{max_retries + 1})")
                                continue
                            else:
                                raise ValueError(f"Cannot parse tool arguments: {args}")
                
                if not isinstance(args, dict):
                    if attempt < max_retries:
                        error_msg = (
                            f"你的上一次响应中的工具调用参数类型不正确。期望是字典（dict），但得到：{type(args).__name__}\n"
                            f"参数内容：{str(args)[:500]}\n\n"
                            f"请重新调用 todo_delta 工具，确保返回的参数是有效的 JSON 对象（字典）。"
                        )
                        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                        messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                        logger.info(f"[PlanningLLM] Retrying due to wrong argument type (attempt {attempt + 1}/{max_retries + 1})")
                        continue
                    else:
                        raise TypeError(f"PlanningLLM tool arguments are not a dict. Got: {type(args)}, value: {args}")
                
                return args
                
            except (ValueError, TypeError) as e:
                if attempt >= max_retries:
                    logger.error(f"[PlanningLLM] Failed after {max_retries + 1} attempts: {e}")
                    raise
                error_msg = (
                    f"你的上一次响应出现了错误：{str(e)}\n"
                    f"请重新调用 todo_delta 工具，确保返回的参数格式正确。"
                )
                if resp:
                    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=resp.content or ""))
                messages.append(ChatMessage(role=MessageRole.USER, content=error_msg))
                logger.info(f"[PlanningLLM] Retrying due to error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                continue
        
        logger.error(f"[PlanningLLM] Unexpected end of retry loop")
        return {
            "actions": []
        }



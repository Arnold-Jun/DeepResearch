from __future__ import annotations

from typing import Any

from src.models import ChatMessage
from src.models.base import MessageRole, Model
from src.orchestrator.state import OrchestratorState, TodoItem


def _format_all_subtask_results(todo_list: list[TodoItem]) -> str:
    """
    格式化所有已完成子任务的结果，用于最终总结。
    """
    completed_subtasks = [
        item for item in todo_list 
        if not item.is_parent and item.status == "done" and item.last_result_summary
    ]
    
    if not completed_subtasks:
        return "（无已完成子任务）"
    
    lines: list[str] = []
    for idx, st in enumerate(completed_subtasks, 1):
        lines.append(f"\n【子任务 {idx}】")
        lines.append(f"任务：{st.task}")
        lines.append(f"Agent类型：{st.agent_type or '未知'}")
        
        # 显示结果（截断过长的内容）
        result = st.last_result_summary
        if len(result) > 2000:
            result = result[:2000] + "\n...[结果已截断，完整内容请查看日志]..."
        lines.append(f"执行结果：\n{result}")
        lines.append("")
    
    return "\n".join(lines)


class SummarizerLLM:
    """
    专门的总结模块，负责整合所有子任务的结果并生成最终答案。
    
    与 Planner 的区别：
    - Planner: 专注于任务规划和管理，使用结构化工具调用
    - Summarizer: 专注于内容整合和总结，生成自然语言答案
    """
    
    def __init__(self, *, model: Model):
        self.model = model
    
    async def summarize(self, state: OrchestratorState) -> str:
        """
        整合所有已完成子任务的结果，生成最终答案。
        
        Args:
            state: OrchestratorState，包含所有任务和结果
            
        Returns:
            str: 整合后的最终答案
        """
        system = (
            "你是总结模块（SUMMARIZER）。你的工作是整合所有已完成子任务的研究结果，生成一个详细、全面的最终答案。\n"
            "\n"
            "你的任务：\n"
            "- 仔细阅读所有已完成子任务的执行结果\n"
            "- 识别关键发现、重要见解、数据、案例等\n"
            "- 去除重复信息，整合相关内容\n"
            "- 按照逻辑结构组织答案（如：概述、主要发现、详细分析、结论等）\n"
            "- 确保答案全面、准确、有条理\n"
            "\n"
            "输出要求：\n"
            "- 必须是一个详细、全面的答案，不能只是简单的确认消息\n"
            "- 必须包含关键发现、重要见解、数据、案例等具体内容\n"
            "- 应该基于所有已完成子任务的研究结果来生成\n"
            "- 如果子任务结果中包含数据、引用、来源等信息，应该保留并标注\n"
            "- 答案应该结构清晰，易于阅读\n"
            "\n"
            "直接输出最终答案，不需要使用任何工具调用。"
        )
        
        # 收集所有已完成子任务的结果
        all_subtask_results = _format_all_subtask_results(state.todo_list)
        
        # 统计信息
        all_tasks = [item for item in state.todo_list if not item.is_parent]
        completed_count = len([item for item in all_tasks if item.status == "done"])
        failed_count = len([item for item in all_tasks if item.status == "failed"])
        total_count = len(all_tasks)
        
        completion_rate = (completed_count / total_count * 100) if total_count > 0 else 0.0
        
        user = (
            f"用户原始任务：\n{state.task}\n\n"
            f"任务执行统计：\n"
            f"- 总子任务数：{total_count}\n"
            f"- 已完成：{completed_count}\n"
            f"- 失败：{failed_count}\n"
            f"- 完成率：{completion_rate:.1f}%\n\n"
            f"所有已完成子任务的执行结果：\n{all_subtask_results}\n\n"
            f"上一轮汇总（最新）：\n{state.last_round_summary or '（无）'}\n\n"
            "请基于以上所有信息，生成一个详细、全面的最终答案。"
            "答案应该整合所有已完成子任务的研究结果，去除重复，按照逻辑结构组织。"
        )
        
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system),
            ChatMessage(role=MessageRole.USER, content=user),
        ]
        
        from src.logger import logger
        logger.info("[SummarizerLLM] 开始生成最终答案...")
        
        try:
            response = await self.model(messages, tools_to_call_from=[])
            
            final_answer = response.content or ""
            
            if not final_answer or len(final_answer.strip()) < 50:
                logger.warning("[SummarizerLLM] 生成的答案过短，可能不完整")
                # 如果答案太短，尝试从子任务结果中提取
                if all_subtask_results and all_subtask_results != "（无已完成子任务）":
                    final_answer = (
                        "基于已完成子任务的研究结果，以下是主要发现：\n\n"
                        + all_subtask_results
                    )
                else:
                    final_answer = "未能生成完整的最终答案。请查看子任务执行结果。"
            
            logger.info(f"[SummarizerLLM] 最终答案生成完成，长度：{len(final_answer)} 字符")
            return final_answer
            
        except Exception as e:
            logger.error(f"[SummarizerLLM] 生成最终答案时出错：{e}", exc_info=True)
            # 降级处理：返回所有子任务结果的简单拼接
            if all_subtask_results and all_subtask_results != "（无已完成子任务）":
                return (
                    "总结模块生成答案时出错，以下是所有已完成子任务的结果：\n\n"
                    + all_subtask_results
                )
            return "总结模块生成答案时出错，且没有可用的子任务结果。"



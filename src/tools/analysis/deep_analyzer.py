"""
深度分析工具 - 共享库

提供数据模型、结构化工具定义和提示词常量，供单步 LLM 工具使用。
"""

import os
from typing import Optional, Dict, Any, List
from PIL import Image
from pydantic import BaseModel, Field

from src.tools import AsyncTool
from src.tools.markdown.mdconvert import MarkitdownConverter

# LLM 交互的提示词常量
DEEP_ANALYZER_INSTRUCTION = """您应该逐步分析任务和/或附加内容。
* 当任务涉及玩游戏或执行计算时。请考虑游戏或计算规则施加的条件。您可以考虑极端条件。
* 当任务涉及拼写单词时，您必须确保遵循拼写规则，并且生成的单词是有意义的。
* 当任务涉及计算特定多边形的面积时。您应该将多边形分成子多边形，并确保每个子多边形的面积是可计算的（例如，矩形、圆形、三角形等）。逐步计算每个子多边形的面积并将它们相加得到最终面积。
* 当任务涉及计算和统计时，必须考虑所有约束。未能考虑这些约束很容易导致统计错误。

这是任务：
{task}"""

DEEP_ANALYZER_SUMMARY_DESCRIPTION = """请对不同模型的输出进行逐步分析。比较它们的结果，识别差异，提取准确的部分，消除不正确的部分，并综合一个连贯的摘要。

分析：
{analysis}"""

# 结构化输出工具定义（用于 LLM 工具调用）
class AnalyzeTool(AsyncTool):
    """用于执行单个分析的工具（结构化输出定义）"""
    
    name: str = "analyze"
    description: str = """对给定任务和/或源执行系统性、逐步分析。
必须提供 task 或 source 中的至少一个。当两者都可用时，工具将在提供的源的上下文中分析和解决任务。"""
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "要分析并应该解决的任务。如果未提供，工具将仅专注于为附加文件或链接的 URL 添加标题。",
                "nullable": True,
            },
            "source": {
                "type": "string",
                "description": "要分析的附加文件或 URI。工具将处理和解释文件或网页的内容。",
                "nullable": True
            },
            "analysis": {
                "type": "string",
                "description": "工具生成的分析结果。"
            }
        },
        "required": ["analysis"],
        "additionalProperties": False,
    }
    output_type = "any"
    
    async def forward(self, analysis: str, task: Optional[str] = None, source: Optional[str] = None) -> str:
        """执行分析并返回结果"""
        return analysis


class SummarizeAnalysisTool(AsyncTool):
    """用于总结多个分析结果的工具（结构化输出定义）"""
    
    name: str = "summarize_analysis"
    description: str = """对不同模型的输出进行逐步分析。比较它们的结果，识别差异，提取准确的部分，消除不正确的部分，并综合一个连贯的摘要。"""
    
    parameters: dict = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "综合摘要，整合所有模型的分析结果。"
            },
            "model_analyses": {
                "type": "object",
                "description": "各模型的分析结果字典（模型名 -> 分析结果）",
                "additionalProperties": {"type": "string"},
                "nullable": True
            }
        },
        "required": ["summary"],
        "additionalProperties": False,
    }
    output_type = "any"
    
    async def forward(self, summary: str, model_analyses: Optional[Dict[str, str]] = None) -> str:
        """生成综合摘要"""
        return summary


# 工具函数：准备分析内容
def prepare_analysis_content(task: Optional[str], source: Optional[str], converter: MarkitdownConverter) -> tuple[list, bool]:
    """
    准备分析内容（文本和/或图像）
    
    Returns:
        tuple[list, bool]: (内容列表，是否添加标题说明)
    """
    add_note = False
    if not task:
        add_note = True
        task = "请为附加文件或 URI 编写详细标题。"
    
    task_text = DEEP_ANALYZER_INSTRUCTION.format(task=task)
    content = [
        {"type": "text", "text": task_text},
    ]
    
    if source:
        ext = os.path.splitext(source)[-1].lower()
        
        if ext in ['.png', '.jpg', '.jpeg']:
            content.append({
                "type": "image",
                "image": Image.open(source),
            })
        else:
            try:
                extracted_content = converter.convert(source).text_content
            except Exception as e:
                extracted_content = f"未能从 {source} 提取内容。错误: {e}"
            
            content.append({
                "type": "text",
                "text": " - 附加文件内容: \n\n" + extracted_content,
            })
    
    return content, add_note


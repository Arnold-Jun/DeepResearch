"""
深度分析工具 - LLM 单步变换工具

这些工具都是单步分析变换，强约束，可单测：
- AnalyzeTool: 执行单个分析（可并行调用多个模型）
- SummarizeAnalysisTool: 总结多个分析结果
"""

import asyncio
import json5
from typing import Optional, Dict, Any, List

from src.models import model_manager, ChatMessage, MessageRole
from src.tools import AsyncTool, ToolResult
from src.tools.analysis.deep_analyzer import (
    AnalyzeTool as StructAnalyzeTool,
    SummarizeAnalysisTool as StructSummarizeAnalysisTool,
    DEEP_ANALYZER_INSTRUCTION,
    DEEP_ANALYZER_SUMMARY_DESCRIPTION,
    prepare_analysis_content,
)
from src.tools.markdown.mdconvert import MarkitdownConverter
from src.logger import logger
from src.registry import TOOL


@TOOL.register_module(name="analyze_tool", force=True)
class AnalyzeTool(AsyncTool):
    """
    执行单个分析的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入任务/源 → 输出分析结果
    - 强约束：结构化输出
    - 可单测：给定输入，验证输出格式和合理性
    - 支持多模型并行分析
    """
    
    name = "analyze_tool"
    description = """对给定任务执行系统性、逐步分析或计算，可选择利用来自外部资源（如附加文件或 URI）的信息来提供全面的推理和答案。
必须提供 task 或 source 中的至少一个。当两者都可用时，工具将在提供的源的上下文中分析和解决任务。
source 可以是本地文件路径或 URI。支持的文件扩展名和 URI 包括：文本、图像、音频、视频、归档、URI等。"""
    
    parameters = {
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
            }
        },
        "required": []
    }
    output_type = "string"
    
    def __init__(self, model_id: str = "qwen3-14b", **kwargs):
        super().__init__(**kwargs)
        self.model = model_manager.get_model(model_id)
        self.converter = MarkitdownConverter()
        # 使用结构化输出工具（从 deep_analyzer 导入）
        self.struct_tool = StructAnalyzeTool()
    
    async def forward(self, task: Optional[str] = None, source: Optional[str] = None) -> ToolResult:
        """执行分析（单步变换）"""
        if not task and not source:
            return ToolResult(
                output="",
                error="必须提供 task 或 source 中的至少一个。"
            )
        
        try:
            # 准备分析内容
            content, add_note = prepare_analysis_content(task, source, self.converter)
            
            messages = [
                ChatMessage(role=MessageRole.USER, content=content)
            ]
            
            response = await self.model(
                messages,
                tools_to_call_from=[self.struct_tool]
            )
            
            # 解析结构化输出
            if response.tool_calls and len(response.tool_calls) > 0:
                arguments = json5.loads(response.tool_calls[0].function.arguments)
                analysis = arguments.get("analysis", "")
            else:
                # 回退：使用原始响应
                analysis = response.content.strip() if response.content else ""
            
            if add_note:
                analysis = f"您没有提供特定问题，所以这里是图像的详细标题: {analysis}"
            
            logger.info(f"[AnalyzeTool] Analysis completed, length: {len(analysis)}")
            return ToolResult(output=analysis)
            
        except Exception as e:
            logger.error(f"[AnalyzeTool] Error: {e}")
            return ToolResult(
                output="",
                error=str(e)
            )


@TOOL.register_module(name="parallel_analyze_tool", force=True)
class ParallelAnalyzeTool(AsyncTool):
    """
    并行执行多个模型分析的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入任务/源 → 输出多个模型的分析结果
    - 强约束：结构化输出
    - 可单测：给定输入，验证输出格式和合理性
    - 并行执行：同时调用多个分析器模型
    """
    
    name = "parallel_analyze_tool"
    description = """并行使用多个分析器模型对给定任务执行系统性、逐步分析。
必须提供 task 或 source 中的至少一个。此工具会并行调用多个模型进行分析，返回所有模型的分析结果。"""
    
    parameters = {
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
            }
        },
        "required": []
    }
    output_type = "object"
    
    def __init__(self, analyzer_model_ids: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.analyzer_model_ids = analyzer_model_ids or ["qwen3-14b"]
        self.analyzer_models = {
            model_id: model_manager.get_model(model_id)
            for model_id in self.analyzer_model_ids
        }
        self.converter = MarkitdownConverter()
        self.struct_tool = StructAnalyzeTool()
    
    async def forward(self, task: Optional[str] = None, source: Optional[str] = None) -> ToolResult:
        """并行执行多个模型的分析"""
        if not task and not source:
            return ToolResult(
                output={},
                error="必须提供 task 或 source 中的至少一个。"
            )
        
        try:
            # 准备分析内容（所有模型共享）
            content, add_note = prepare_analysis_content(task, source, self.converter)
            
            messages = [
                ChatMessage(role=MessageRole.USER, content=content)
            ]
            
            # 并行调用所有分析器模型
            async def analyze_with_model(model_name: str, model):
                try:
                    response = await model(
                        messages,
                        tools_to_call_from=[self.struct_tool]
                    )
                    
                    # 解析结构化输出
                    if response.tool_calls and len(response.tool_calls) > 0:
                        arguments = json5.loads(response.tool_calls[0].function.arguments)
                        analysis = arguments.get("analysis", "")
                    else:
                        analysis = response.content.strip() if response.content else ""
                    
                    if add_note:
                        analysis = f"您没有提供特定问题，所以这里是图像的详细标题: {analysis}"
                    
                    logger.info(f"[ParallelAnalyzeTool] {model_name} analysis completed, length: {len(analysis)}")
                    return model_name, analysis
                except Exception as e:
                    logger.error(f"[ParallelAnalyzeTool] Error with {model_name}: {e}")
                    return model_name, f"分析失败: {e}"
            
            # 并行执行所有分析
            tasks = [analyze_with_model(model_name, model) for model_name, model in self.analyzer_models.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 构建结果字典
            analysis_dict = {}
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"[ParallelAnalyzeTool] Task failed: {result}")
                    continue
                if isinstance(result, tuple) and len(result) == 2:
                    model_name, analysis = result
                    analysis_dict[model_name] = analysis
            
            logger.info(f"[ParallelAnalyzeTool] Completed parallel analysis with {len(analysis_dict)} models")
            return ToolResult(output=analysis_dict)
            
        except Exception as e:
            logger.error(f"[ParallelAnalyzeTool] Error: {e}")
            return ToolResult(
                output={},
                error=str(e)
            )


@TOOL.register_module(name="summarize_analysis_tool", force=True)
class SummarizeAnalysisTool(AsyncTool):
    """
    总结多个分析结果的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入多个分析结果 → 输出综合摘要
    - 强约束：结构化输出
    - 可单测：给定多个分析结果，验证摘要质量
    """
    
    name = "summarize_analysis_tool"
    description = """对不同模型的输出进行逐步分析。比较它们的结果，识别差异，提取准确的部分，消除不正确的部分，并综合一个连贯的摘要。
输入多个模型的分析结果，输出综合摘要。"""
    
    parameters = {
        "type": "object",
        "properties": {
            "model_analyses": {
                "type": "object",
                "description": "各模型的分析结果字典（模型名 -> 分析结果）",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["model_analyses"]
    }
    output_type = "string"
    
    def __init__(self, summarizer_model_id: str = "qwen3-14b", **kwargs):
        super().__init__(**kwargs)
        self.summarizer_model = model_manager.get_model(summarizer_model_id)
        # 使用结构化输出工具（从 deep_analyzer 导入）
        self.struct_tool = StructSummarizeAnalysisTool()
    
    async def forward(self, model_analyses: Dict[str, str]) -> ToolResult:
        """生成综合摘要（单步变换）"""
        try:
            # 构建提示词
            analysis_text = ""
            for model_name, analysis in model_analyses.items():
                analysis_text += f"{model_name}:\n{analysis}\n\n"
            
            prompt = DEEP_ANALYZER_SUMMARY_DESCRIPTION.format(analysis=analysis_text)
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            
            response = await self.summarizer_model(
                messages,
                tools_to_call_from=[self.struct_tool]
            )
            
            # 解析结构化输出
            if response.tool_calls and len(response.tool_calls) > 0:
                arguments = json5.loads(response.tool_calls[0].function.arguments)
                summary = arguments.get("summary", "")
            else:
                # 回退：使用原始响应
                summary = response.content.strip() if response.content else ""
            
            logger.info(f"[SummarizeAnalysisTool] Summary generated, length: {len(summary)}")
            return ToolResult(output=summary)
            
        except Exception as e:
            logger.error(f"[SummarizeAnalysisTool] Error: {e}")
            return ToolResult(
                output="",
                error=str(e)
            )


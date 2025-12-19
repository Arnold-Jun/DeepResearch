"""
深度研究工具 - LLM 单步变换工具

这些工具都是单步预研变换，强约束，可单测：
- OptimizeQueryTool: 优化搜索查询
- ExtractInsightsTool: 从内容中提取见解
- GenerateFollowUpsTool: 生成后续查询
- GenerateSummaryTool: 生成研究摘要
"""

import asyncio
import json5
import time
from typing import List, Optional, Dict, Any, Set

from src.models import model_manager, ChatMessage, MessageRole
from src.tools import AsyncTool, ToolResult
from src.tools.research.web_searcher import WebSearcherTool, SearchResult
from src.tools.research.deep_researcher import (
    ResearchInsight,
    ResearchSummary,
    OptimizedQueryTool as StructOptimizedQueryTool,
    ExtractInsightsTool as StructExtractInsightsTool,
    GenerateFollowUpsTool as StructGenerateFollowUpsTool,
    OPTIMIZE_QUERY_INSTRUCTION,
    EXTRACT_INSIGHTS_PROMPT,
    GENERATE_FOLLOW_UPS_PROMPT,
    FALLBACK_RELEVANCE_SCORE,
)
from src.logger import logger
from src.registry import TOOL


@TOOL.register_module(name="optimize_query_tool", force=True)
class OptimizeQueryTool(AsyncTool):
    """
    优化搜索查询的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入查询 → 输出优化查询
    - 强约束：结构化输出
    - 可单测：给定输入，验证输出格式和合理性
    """
    
    name = "optimize_query_tool"
    description = """优化搜索查询，使其对网络搜索更有效。
输入原始查询，输出优化后的查询和可选的年份过滤。
此工具会重新表述查询，使其更具体、使用相关关键词，并确保清晰简洁。"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "原始搜索查询"
            }
        },
        "required": ["query"]
    }
    output_type = "object"
    
    def __init__(self, model_id: str = "qwen3-14b", **kwargs):
        super().__init__(**kwargs)
        self.model = model_manager.get_model(model_id)
        # 使用结构化输出工具（从 deep_researcher 导入）
        self.struct_tool = StructOptimizedQueryTool()
    
    async def forward(self, query: str) -> ToolResult:
        """优化查询（单步变换）"""
        try:
            prompt = OPTIMIZE_QUERY_INSTRUCTION.format(query=query)
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            
            response = await self.model(
                messages,
                tools_to_call_from=[self.struct_tool]
            )
            
            # 解析结构化输出
            if response.tool_calls and len(response.tool_calls) > 0:
                arguments = json5.loads(response.tool_calls[0].function.arguments)
                result = {
                    "optimized_query": arguments.get("optimized_query", query),
                    "filter_year": arguments.get("filter_year", None)
                }
            else:
                # 回退：使用原始响应
                result = {
                    "optimized_query": response.content.strip() if response.content else query,
                    "filter_year": None
                }
            
            logger.info(f"[OptimizeQueryTool] Optimized: {result['optimized_query']}, filter_year: {result['filter_year']}")
            return ToolResult(output=result)
            
        except Exception as e:
            logger.error(f"[OptimizeQueryTool] Error: {e}")
            return ToolResult(
                output={"optimized_query": query, "filter_year": None},
                error=str(e)
            )


@TOOL.register_module(name="extract_insights_tool", force=True)
class ExtractInsightsTool(AsyncTool):
    """
    从搜索结果中提取见解的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入内容 → 输出见解列表
    - 强约束：结构化输出，相关性评分
    - 可单测：给定内容，验证见解提取质量
    """
    
    name = "extract_insights_tool"
    description = """从网页内容中提取与研究查询相关的关键见解，并评估相关性。
输入研究查询、内容和来源信息，输出见解列表（每个见解包含内容和相关性评分）。
相关性评分范围：0.0-1.0，其中 1.0 表示完全相关。"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "研究查询"
            },
            "content": {
                "type": "string",
                "description": "要分析的内容（网页内容）"
            },
            "url": {
                "type": "string",
                "description": "内容来源 URL"
            },
            "title": {
                "type": "string",
                "description": "内容标题（可选）",
                "nullable": True
            }
        },
        "required": ["query", "content", "url"]
    }
    output_type = "array"
    
    def __init__(self, model_id: str = "qwen3-14b", **kwargs):
        super().__init__(**kwargs)
        self.model = model_manager.get_model(model_id)
        # 使用结构化输出工具（从 deep_researcher 导入）
        self.struct_tool = StructExtractInsightsTool()
    
    async def forward(
        self,
        query: str,
        content: str,
        url: str,
        title: Optional[str] = None
    ) -> ToolResult:
        """提取见解（单步变换）"""
        try:
            # 限制内容长度
            content_limited = content[:5000] if len(content) > 5000 else content
            
            prompt = EXTRACT_INSIGHTS_PROMPT.format(
                query=query,
                content=content_limited
            )
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            
            response = await self.model(
                messages,
                tools_to_call_from=[self.struct_tool]
            )
            
            # 解析结构化输出
            insights = []
            if response.tool_calls and len(response.tool_calls) > 0:
                arguments = json5.loads(response.tool_calls[0].function.arguments)
                extracted = arguments.get("insights", [])
                
                for insight_data in extracted:
                    insights.append({
                        "content": insight_data.get("content", ""),
                        "source_url": url,
                        "source_title": title,
                        "relevance_score": insight_data.get(
                            "relevance_score", FALLBACK_RELEVANCE_SCORE
                        )
                    })
            else:
                # 回退：使用内容的前500字符作为见解
                insights.append({
                    "content": content[:500],
                    "source_url": url,
                    "source_title": title,
                    "relevance_score": FALLBACK_RELEVANCE_SCORE
                })
            
            logger.info(f"[ExtractInsightsTool] Extracted {len(insights)} insights from {title or url}")
            return ToolResult(output=insights)
            
        except Exception as e:
            logger.warning(f"[ExtractInsightsTool] Error extracting from {url}: {e}")
            # 回退见解
            return ToolResult(output=[{
                "content": content[:500] if content else "",
                "source_url": url,
                "source_title": title,
                "relevance_score": FALLBACK_RELEVANCE_SCORE
            }])


@TOOL.register_module(name="batch_extract_insights_tool", force=True)
class BatchExtractInsightsTool(AsyncTool):
    """
    批量提取见解的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入多个搜索结果 → 输出见解列表
    - 强约束：结构化输出，相关性评分
    - 可单测：给定多个搜索结果，验证见解提取质量
    - 提高效率：一次处理多个搜索结果
    """
    
    name = "batch_extract_insights_tool"
    description = """从多个网页内容中批量提取与研究查询相关的关键见解，并评估相关性。
输入研究查询和搜索结果列表，输出所有见解的列表（每个见解包含内容和相关性评分）。
相关性评分范围：0.0-1.0，其中 1.0 表示完全相关。
此工具可以高效地批量处理多个搜索结果。"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "研究查询"
            },
            "search_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string", "nullable": True},
                        "content": {"type": "string"}
                    },
                    "required": ["url", "content"]
                },
                "description": "搜索结果列表（包含url、title、content）"
            }
        },
        "required": ["query", "search_results"]
    }
    output_type = "array"
    
    def __init__(self, model_id: str = "qwen3-14b", **kwargs):
        super().__init__(**kwargs)
        self.model = model_manager.get_model(model_id)
        # 使用结构化输出工具（从 deep_researcher 导入）
        self.struct_tool = StructExtractInsightsTool()
    
    async def forward(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> ToolResult:
        """批量提取见解（单步变换）"""
        all_insights = []
        
        try:
            # 对每个搜索结果提取见解
            for result_item in search_results:
                url = result_item.get('url', '')
                title = result_item.get('title')
                content = result_item.get('content', '')
                
                if not content or not url:
                    continue
                
                # 限制内容长度
                content_limited = content[:5000] if len(content) > 5000 else content
                
                prompt = EXTRACT_INSIGHTS_PROMPT.format(
                    query=query,
                    content=content_limited
                )
                messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
                
                try:
                    response = await self.model(
                        messages,
                        tools_to_call_from=[self.struct_tool]
                    )
                    
                    # 解析结构化输出
                    if response.tool_calls and len(response.tool_calls) > 0:
                        arguments = json5.loads(response.tool_calls[0].function.arguments)
                        extracted = arguments.get("insights", [])
                        
                        for insight_data in extracted:
                            all_insights.append({
                                "content": insight_data.get("content", ""),
                                "source_url": url,
                                "source_title": title,
                                "relevance_score": insight_data.get(
                                    "relevance_score", FALLBACK_RELEVANCE_SCORE
                                )
                            })
                    else:
                        # 回退
                        all_insights.append({
                            "content": content[:500],
                            "source_url": url,
                            "source_title": title,
                            "relevance_score": FALLBACK_RELEVANCE_SCORE
                        })
                
                except Exception as e:
                    logger.warning(f"[BatchExtractInsightsTool] Error extracting from {url}: {e}")
                    # 回退见解
                    all_insights.append({
                        "content": content[:500] if content else "",
                        "source_url": url,
                        "source_title": title,
                        "relevance_score": FALLBACK_RELEVANCE_SCORE
                    })
            
            logger.info(f"[BatchExtractInsightsTool] Extracted {len(all_insights)} insights from {len(search_results)} results")
            return ToolResult(output=all_insights)
            
        except Exception as e:
            logger.error(f"[BatchExtractInsightsTool] Error: {e}")
            return ToolResult(output=[], error=str(e))


@TOOL.register_module(name="generate_follow_ups_tool", force=True)
class GenerateFollowUpsTool(AsyncTool):
    """
    生成后续研究查询的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入见解 → 输出查询列表
    - 强约束：最多3个查询，格式明确
    - 可单测：给定见解，验证查询相关性
    """
    
    name = "generate_follow_ups_tool"
    description = """基于已发现的见解生成后续研究查询，以探索空白或相关领域。
输入原始查询、当前查询和见解列表，输出最多3个后续查询。
每个查询应该简洁并专注于研究主题的特定方面。"""
    
    parameters = {
        "type": "object",
        "properties": {
            "original_query": {
                "type": "string",
                "description": "原始研究查询"
            },
            "current_query": {
                "type": "string",
                "description": "当前查询"
            },
            "insights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "已发现的见解列表（字符串格式）"
            }
        },
        "required": ["original_query", "current_query", "insights"]
    }
    output_type = "array"
    
    def __init__(self, model_id: str = "qwen3-14b", **kwargs):
        super().__init__(**kwargs)
        self.model = model_manager.get_model(model_id)
        # 使用结构化输出工具（从 deep_researcher 导入）
        self.struct_tool = StructGenerateFollowUpsTool()
    
    async def forward(
        self,
        original_query: str,
        current_query: str,
        insights: List[str]
    ) -> ToolResult:
        """生成后续查询（单步变换）"""
        try:
            # 格式化见解（只使用前5个）
            insights_text = "\n".join([
                f"- {insight}" for insight in insights[:5]
            ])
            
            prompt = GENERATE_FOLLOW_UPS_PROMPT.format(
                original_query=original_query,
                current_query=current_query,
                insights=insights_text
            )
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            
            response = await self.model(
                messages,
                tools_to_call_from=[self.struct_tool]
            )
            
            # 解析结构化输出
            follow_up_queries = []
            if response.tool_calls and len(response.tool_calls) > 0:
                arguments = json5.loads(response.tool_calls[0].function.arguments)
                follow_up_queries = arguments.get("follow_up_queries", [])
            
            # 限制数量（最多3个）
            follow_up_queries = follow_up_queries[:3]
            
            logger.info(f"[GenerateFollowUpsTool] Generated {len(follow_up_queries)} follow-up queries")
            return ToolResult(output=follow_up_queries)
            
        except Exception as e:
            logger.error(f"[GenerateFollowUpsTool] Error: {e}")
            return ToolResult(output=[], error=str(e))


@TOOL.register_module(name="generate_summary_tool", force=True)
class GenerateSummaryTool(AsyncTool):
    """
    生成最终研究摘要的工具（LLM 单步变换）
    
    特点：
    - 单步变换：输入见解 → 输出摘要
    - 强约束：格式化的摘要
    - 可单测：给定见解，验证摘要质量
    """
    
    name = "generate_summary_tool"
    description = """生成最终研究摘要，综合所有发现的见解。
输入原始查询、见解列表和访问的URL，输出格式化的研究摘要。
摘要应该组织良好，包含关键发现、附加信息和补充信息。"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "原始研究查询"
            },
            "insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "source_url": {"type": "string"},
                        "source_title": {"type": "string", "nullable": True},
                        "relevance_score": {"type": "number"}
                    }
                },
                "description": "发现的见解列表"
            },
            "visited_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "访问的URL列表",
                "nullable": True
            },
            "depth_reached": {
                "type": "integer",
                "description": "达到的研究深度",
                "default": 0,
                "nullable": True
            }
        },
        "required": ["query", "insights"]
    }
    output_type = "string"
    
    def __init__(self, summary_model_id: str = "gpt-4o-research-preview", **kwargs):
        super().__init__(**kwargs)
        self.summary_model = model_manager.get_model(summary_model_id)
    
    async def forward(
        self,
        query: str,
        insights: List[Dict[str, Any]],
        visited_urls: Optional[List[str]] = None,
        depth_reached: int = 0
    ) -> ToolResult:
        """生成研究摘要（单步变换）"""
        try:
            # 转换见解为 ResearchInsight 对象
            research_insights = []
            for insight_data in insights:
                research_insights.append(ResearchInsight(
                    content=insight_data.get("content", ""),
                    source_url=insight_data.get("source_url", ""),
                    source_title=insight_data.get("source_title"),
                    relevance_score=insight_data.get("relevance_score", 0.7)
                ))
            
            # 创建研究摘要
            visited_urls_set = set(visited_urls) if visited_urls else set()
            summary = ResearchSummary(
                query=query,
                insights=sorted(research_insights, key=lambda x: x.relevance_score, reverse=True),
                visited_urls=visited_urls_set,
                depth_reached=depth_reached,
            )
            
            # 使用专门的搜索模型生成最终摘要
            messages = [ChatMessage(role=MessageRole.USER, content=query)]
            response = await self.summary_model(messages)
            summary_content = response.content if response.content else ""
            
            # 组合参考材料和摘要
            final_output = summary.output + "\n\n" + summary_content
            
            logger.info(f"[GenerateSummaryTool] Summary generated with {len(research_insights)} insights")
            return ToolResult(output=final_output)
            
        except Exception as e:
            logger.error(f"[GenerateSummaryTool] Error: {e}")
            return ToolResult(output="", error=str(e))


@TOOL.register_module(name="parallel_research_tool", force=True)
class ParallelResearchTool(AsyncTool):
    """
    并行研究工具 - 并行执行多个后续查询的研究
    
    特点：
    - 接收多个后续查询
    - 并行执行研究周期（搜索、提取见解）
    - 递归处理，直到达到最大深度
    - 返回所有新发现的见解
    
    注意：这是一个组合工具，内部调用其他工具来实现并行递归研究
    """
    
    name = "parallel_research_tool"
    description = """并行执行多个后续查询的深度研究。
输入后续查询列表、当前深度、研究参数等，输出所有新发现的见解列表。
此工具会并行处理多个查询，每个查询会执行完整的研究周期（搜索、提取见解），
并在达到最大深度前递归处理新生成的后续查询。
分支因子限制为最多2个并行分支。"""
    
    parameters = {
        "type": "object",
        "properties": {
            "follow_up_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "后续查询列表（最多2个，用于并行研究）"
            },
            "original_query": {
                "type": "string",
                "description": "原始研究查询"
            },
            "current_depth": {
                "type": "integer",
                "description": "当前研究深度"
            },
            "max_depth": {
                "type": "integer",
                "description": "最大研究深度"
            },
            "filter_year": {
                "type": "integer",
                "description": "年份过滤（可选）",
                "nullable": True
            },
            "deadline": {
                "type": "number",
                "description": "截止时间（Unix时间戳）"
            },
            "max_insights": {
                "type": "integer",
                "description": "最大见解数"
            },
            "existing_insights": {
                "type": "array",
                "items": {"type": "object"},
                "description": "已存在的见解列表（用于共享上下文）"
            },
            "visited_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "已访问的URL列表（用于共享上下文）"
            }
        },
        "required": ["follow_up_queries", "original_query", "current_depth", "max_depth", "deadline"]
    }
    output_type = "array"
    
    def __init__(self, model_id: str = "qwen3-14b", **kwargs):
        super().__init__(**kwargs)
        self.model = model_manager.get_model(model_id)
        self.web_searcher = WebSearcherTool()
        self.web_searcher.fetch_content = True
        # 创建工具实例（使用类名，因为它们在同一个文件中）
        self.extract_tool = ExtractInsightsTool(model_id=model_id)
        self.generate_follow_ups_tool = GenerateFollowUpsTool(model_id=model_id)
    
    async def _research_cycle(
        self,
        query: str,
        depth: int,
        filter_year: Optional[int],
        deadline: float,
        max_depth: int,
        max_insights: int,
        original_query: str,
        shared_insights: List[Dict[str, Any]],
        visited_urls: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        单个研究周期（搜索、提取见解、生成后续查询）
        
        这是递归函数，用于并行执行多个研究分支
        """
        # 检查终止条件
        if time.time() >= deadline or depth >= max_depth:
            return []
        
        logger.info(f"[ParallelResearchTool] Research cycle, depth {depth + 1} - query: {query}")
        new_insights = []
        
        try:
            # 1. 网络搜索
            search_params = {'query': query}
            if filter_year:
                search_params['filter_year'] = filter_year
            
            search_result = await self.web_searcher.forward(**search_params)
            if search_result.error or not search_result.results:
                return []
            
            search_results = search_result.results
            
            # 过滤已访问的 URL
            filtered_results = []
            for result_item in search_results:
                if isinstance(result_item, SearchResult):
                    url = result_item.url
                    title = result_item.title
                    content = result_item.raw_content or result_item.description or ""
                elif isinstance(result_item, dict):
                    url = result_item.get('url', '')
                    title = result_item.get('title', '')
                    content = result_item.get('raw_content') or result_item.get('content', '')
                else:
                    continue
                
                if url and url not in visited_urls and content:
                    filtered_results.append({
                        'url': url,
                        'title': title,
                        'content': content
                    })
                    visited_urls.add(url)
            
            if not filtered_results:
                return []
            
            # 2. 批量提取见解
            for result_item in filtered_results:
                if time.time() >= deadline:
                    break
                
                url = result_item['url']
                title = result_item.get('title')
                content = result_item['content']
                
                # 使用 extract_insights_tool 提取见解
                extract_result = await self.extract_tool.forward(
                    query=original_query,
                    content=content,
                    url=url,
                    title=title
                )
                
                if extract_result.output:
                    if isinstance(extract_result.output, list):
                        new_insights.extend(extract_result.output)
                    else:
                        new_insights.append(extract_result.output)
            
            if not new_insights:
                return []
            
            # 添加到共享见解
            shared_insights.extend(new_insights)
            
            # 限制见解数量
            if len(shared_insights) > max_insights:
                shared_insights.sort(
                    key=lambda x: x.get('relevance_score', 0.7) if isinstance(x, dict) else 0.7,
                    reverse=True
                )
                shared_insights[:] = shared_insights[:max_insights]
            
            # 3. 生成后续查询
            insights_text_list = [
                insight.get('content', '') if isinstance(insight, dict) else str(insight)
                for insight in shared_insights[:5]
            ]
            
            follow_ups_result = await self.generate_follow_ups_tool.forward(
                original_query=original_query,
                current_query=query,
                insights=insights_text_list
            )
            
            follow_up_queries = []
            if follow_ups_result.output and isinstance(follow_ups_result.output, list):
                follow_up_queries = follow_ups_result.output[:2]  # 限制分支因子为2
            
            # 4. 递归：并行执行后续查询
            if follow_up_queries and depth + 1 < max_depth and time.time() < deadline:
                tasks = []
                for follow_up in follow_up_queries:
                    if time.time() >= deadline:
                        break
                    
                    tasks.append(
                        self._research_cycle(
                            query=follow_up,
                            depth=depth + 1,
                            filter_year=filter_year,
                            deadline=deadline,
                            max_depth=max_depth,
                            max_insights=max_insights,
                            original_query=original_query,
                            shared_insights=shared_insights,
                            visited_urls=visited_urls,
                        )
                    )
                
                # 并行执行递归研究调用
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        
        except Exception as e:
            logger.error(f"[ParallelResearchTool] Error in research cycle for '{query}': {e}")
        
        return new_insights
    
    async def forward(
        self,
        follow_up_queries: List[str],
        original_query: str,
        current_depth: int,
        max_depth: int,
        deadline: float,
        filter_year: Optional[int] = None,
        max_insights: int = 20,
        existing_insights: Optional[List[Dict[str, Any]]] = None,
        visited_urls: Optional[List[str]] = None,
    ) -> ToolResult:
        """并行执行多个研究任务"""
        try:
            # 限制分支因子为2
            parallel_queries = follow_up_queries[:2]
            
            # 创建共享状态
            shared_insights = list(existing_insights) if existing_insights else []
            visited_urls_set = set(visited_urls) if visited_urls else set()
            
            # 并行执行研究任务
            tasks = []
            for query in parallel_queries:
                if time.time() >= deadline:
                    break
                
                tasks.append(
                    self._research_cycle(
                        query=query,
                        depth=current_depth,
                        filter_year=filter_year,
                        deadline=deadline,
                        max_depth=max_depth,
                        max_insights=max_insights,
                        original_query=original_query,
                        shared_insights=shared_insights,
                        visited_urls=visited_urls_set,
                    )
                )
            
            # 并行执行所有研究任务
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # 返回新发现的见解（排除原有的）
            existing_count = len(existing_insights) if existing_insights else 0
            new_insights = shared_insights[existing_count:] if len(shared_insights) > existing_count else []
            
            # 返回结果，包含新见解和更新的访问URL列表
            result = {
                'new_insights': new_insights,
                'visited_urls': list(visited_urls_set),  # 返回所有访问的URL（包括新的）
            }
            
            logger.info(f"[ParallelResearchTool] Completed parallel research, found {len(new_insights)} new insights, {len(visited_urls_set)} total URLs")
            return ToolResult(output=result)
            
        except Exception as e:
            logger.error(f"[ParallelResearchTool] Error: {e}")
            return ToolResult(output=[], error=str(e))


import re
from typing import List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.tools import AsyncTool

# LLM 交互的提示词
OPTIMIZE_QUERY_INSTRUCTION = """
您是一位研究助手，帮助优化网络研究的搜索查询。
您的任务是重新表述给定的查询，使其对网络搜索更有效。
使其具体，使用相关关键词，并确保清晰简洁。

原始查询：{query}

提供优化的查询文本，无需任何解释或附加格式。
"""

EXTRACT_INSIGHTS_PROMPT = """
分析以下内容并提取与研究查询相关的关键见解。
对于每个见解，在 0.0 到 1.0 的范围内评估其与查询的相关性。

研究查询：{query}
要分析的内容：
{content}

从此内容中提取最多 3 个最重要的见解。对于每个见解：
1. 提供见解内容
2. 提供相关性评分（0.0-1.0）
"""

GENERATE_FOLLOW_UPS_PROMPT = """
基于迄今为止发现的见解，生成后续研究查询以探索空白或相关领域。
这些应该有助于加深我们对主题的理解。

原始查询：{original_query}
当前查询：{current_query}
迄今为止的关键见解：
{insights}

生成最多 3 个具体的后续查询，以帮助解决我们当前知识中的空白。
每个查询应该简洁并专注于研究主题的特定方面。
"""

# 见解解析的常量
DEFAULT_RELEVANCE_SCORE = 1.0
FALLBACK_RELEVANCE_SCORE = 0.7
FALLBACK_CONTENT_LIMIT = 500
# 检测见解开头的模式（数字.、-、*、•）并捕获内容
INSIGHT_MARKER_PATTERN = re.compile(r"^\s*(?:\d+\.|-|\*|•)\s*(.*)")
# 检测相关性评分的模式，捕获数字（不区分大小写）
RELEVANCE_SCORE_PATTERN = re.compile(r"relevance.*?:.*?(\d\.?\d*)", re.IGNORECASE)

class ResearchInsight(BaseModel):
    """研究期间发现的单个见解。"""

    model_config = ConfigDict(frozen=True)  # 使见解不可变

    content: str = Field(description="见解内容")
    source_url: str = Field(description="发现此见解的 URL")
    source_title: Optional[str] = None
    relevance_score: float = Field(
        default=1.0, description="相关性评分（0.0-1.0）", ge=0.0, le=1.0
    )

    def __str__(self) -> str:
        """将见解格式化为带有来源归属的字符串。"""
        source = self.source_title or self.source_url
        return f"{self.content} [Source: {source}]"

# 结构化输出工具定义（用于 LLM 工具调用）
class ResearchSummary(BaseModel):
    """深度研究结果的综合摘要。"""

    output: str = Field(default="", description="格式化的研究摘要")
    query: str = Field(description="原始研究查询")
    insights: List[ResearchInsight] = Field(default_factory=list, description="发现的关键见解")
    visited_urls: Set[str] = Field(default_factory=set, description="研究期间访问的 URL")
    depth_reached: int = Field(default=0, description="达到的最大研究深度", ge=0)

    @model_validator(mode="after")
    def populate_output(self) -> "ResearchSummary":
        """Populate the output field after validation."""
        # Group and sort insights by relevance
        grouped_insights = {
            "Key Findings": [i for i in self.insights if i.relevance_score >= 0.8],
            "Additional Information": [
                i for i in self.insights if 0.5 <= i.relevance_score < 0.8
            ],
            "Supplementary Information": [
                i for i in self.insights if i.relevance_score < 0.5
            ],
        }

        sections = [
            f"# Research: {self.query}\n",
            f"**Sources**: {len(self.visited_urls)} | **Depth**: {self.depth_reached + 1}\n",
        ]

        for section_title, insights in grouped_insights.items():
            if insights:
                sections.append(f"## {section_title}")
                for i, insight in enumerate(insights, 1):
                    sections.extend(
                        [
                            insight.content,
                            f"> Source: [{insight.source_title or 'Link'}]({insight.source_url})\n",
                        ]
                    )

        # Assign the formatted string to the 'output' field inherited from ToolResult
        self.output = "\n".join(sections)
        return self


class OptimizedQueryTool(AsyncTool):
    """用于生成优化搜索查询的工具。"""

    name: str = "optimize_query"
    description: str = """基于原始查询生成优化的搜索查询。此工具重新表述查询以提高搜索效果。"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要优化的原始查询。",
            },
            "optimized_query": {
                "type": "string",
                "description": "工具生成的优化查询。",
            },
            "filter_year": {
                "type": "integer",
                "description": "（可选）按年份过滤结果（例如，2025）。",
                "nullable": True,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    output_type = "any"

    async def forward(self, query: str,
                      optimized_query: str,
                      filter_year: Optional[int] = None):
        """生成优化的搜索查询。"""
        return query, optimized_query, filter_year

class GenerateFollowUpsTool(AsyncTool):
    """基于见解生成后续查询的工具。"""

    name: str = "generate_follow_ups"
    description: str = """基于研究期间发现的见解生成后续查询。此工具有助于探索研究主题中的空白或相关领域。"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "follow_up_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "后续查询列表（最多3个），有助于解决当前知识中的空白",
                "maxItems": 3,
            },
        },
        "required": ["follow_up_queries"],
        "additionalProperties": False,
    }
    output_type = "any"

    async def forward(self, follow_up_queries: List[str]) -> List[str]:
        """基于见解生成后续查询。"""
        return follow_up_queries

class ExtractInsightsTool(AsyncTool):
    """从内容中提取见解的工具。"""

    name: str = "extract_insights"
    description: str = """基于与研究查询的相关性从内容中提取关键见解。此工具在 0.0 到 1.0 的范围内评估每个见解的相关性。"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "见解内容",
                        },
                        "relevance_score": {
                            "type": "number",
                            "description": "相关性评分，范围在 0.0 到 1.0 之间",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["content", "relevance_score"],
                },
                "description": "从内容中提取的关键见解列表",
                "maxItems": 3,
            }
        },
    }
    output_type = "any"
    async def forward(self, insights: any) -> any:
        """基于与查询的相关性从内容中提取见解。"""
        return insights

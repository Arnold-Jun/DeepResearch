from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
import time
import asyncio

from src.tools.research.web_fetcher import WebFetcherTool
from src.tools.research import (
    FirecrawlSearchEngine,
    DuckDuckGoSearchEngine,
    BaiduSearchEngine,
    BingSearchEngine,
    WebSearchEngine,
    SearchItem
)
from src.tools import AsyncTool, ToolResult
from src.logger import logger
from src.registry import TOOL

_WEB_SEARCHER_DESCRIPTION = """Search the web for real-time information about any topic.
This tool returns comprehensive research results with relevant information, URLs, titles, and descriptions.
If the primary research engine fails, it automatically falls back to alternative engines."""

class SearchResult(BaseModel):
    """Represents a single research result returned by a research engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: int = Field(description="Position in research results")
    url: str = Field(description="URL of the research result")
    title: str = Field(default="", description="Title of the research result")
    description: str = Field(
        default="", description="Description or snippet of the research result"
    )
    source: str = Field(description="The research engine that provided this result")
    raw_content: Optional[str] = None

    def __str__(self) -> str:
        """String representation of a research result."""
        return f"{self.title} ({self.url})"


class SearchMetadata(BaseModel):
    """Metadata about the research operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_results: int = Field(description="Total number of results found")
    language: str = Field(description="Language code used for the research")
    country: str = Field(description="Country code used for the research")


class SearchResponse(ToolResult):
    """Structured response from the web research tool, inheriting ToolResult."""
    query: str = Field(description="The research query that was executed")
    results: List[SearchResult] = Field(default_factory=list, description="List of research results")
    metadata: Optional[SearchMetadata] = None

    @model_validator(mode="after")
    def populate_output(self) -> "SearchResponse":

        if self.error:
            return self

        """Populate output or error fields based on research results."""
        result_text = [f"Search results for '{self.query}':"]

        for i, result in enumerate(self.results, 1):
            # Add title with position number
            title = result.title.strip() or "No title"
            result_text.append(f"\n{i}. {title}")

            # Add URL with proper indentation
            result_text.append(f"   URL: {result.url}")

            # Add description if available
            if result.description.strip():
                result_text.append(f"   Description: {result.description}")

            # Add content preview if available
            if result.raw_content:
                content_preview = result.raw_content.replace("\n", " ").strip()
                result_text.append(f"   Content: {content_preview}")

        # Add metadata at the bottom if available
        if self.metadata:
            result_text.extend(
                [
                    f"\nMetadata:",
                    f"- Total results: {self.metadata.total_results}",
                    f"- Language: {self.metadata.language}",
                    f"- Country: {self.metadata.country}",
                ]
            )

        self.output = "\n".join(result_text)
        return self

@TOOL.register_module(name="web_searcher_tool", force=True)
class WebSearcherTool(AsyncTool):
    """Search the web for information using various research engines."""

    name: str = "web_searcher_tool"
    description: str = _WEB_SEARCHER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The research query to submit to the research engine.",
            },
            "filter_year": {
                "type": "integer",
                "description": "(optional) Filter results by year (e.g., 2025).",
                "nullable": True,
            },
        },
        "required": ["query"],
    }
    output_type = 'any'

    def __init__(self,
                 *args,
                 engine: str = "Firecrawl",
                 fallback_engines=["DuckDuckGo", "Baidu", "Bing"],
                 max_length: int = 4096,
                 retry_delay: int = 10,
                 max_retries: int = 3,
                 lang: str = "en",
                 country: str = "us",
                 num_results: int = 5,
                 fetch_content: bool = False,
                 **kwargs
                 ):
        super(WebSearcherTool, self).__init__()

        self.engine = engine.lower()
        self.fallback_engines = [
            fe.lower() for fe in fallback_engines if fe.lower() != self.engine
        ]

        self.max_length = max_length
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.lang = lang
        self.country = country
        self.num_results = num_results
        self.fetch_content = fetch_content

        self._search_engine: dict[str, WebSearchEngine] = {
            "firecrawl": FirecrawlSearchEngine(),
            "duckduckgo": DuckDuckGoSearchEngine(),
            "baidu": BaiduSearchEngine(),
            "bing": BingSearchEngine(),
        }
        self.content_fetcher: WebFetcherTool = WebFetcherTool()

    async def forward(
        self,
        query: str,
        filter_year: Optional[int] = None,
    ) -> SearchResponse:
        """
        Execute a Web research and return detailed research results.

        Args:
            query: The research query to submit to the research engine
            num_results: The number of research results to return (default: 5)
            lang: Language code for research results (default from config)
            country: Country code for research results (default from config)
            fetch_content: Whether to fetch content from result pages (default: False)

        Returns:
            A structured response containing research results and metadata
        """
        search_params = {"lang": self.lang, "country": self.country}

        if filter_year is not None:
            search_params["filter_year"] = filter_year

        # Try searching with retries when all engines fail
        for retry_count in range(self.max_retries + 1):
            results = await self._try_all_engines(query, self.num_results, search_params)
            if results:
                # Fetch content if requested
                if self.fetch_content:
                    results = await self._fetch_content_for_results(results)

                # Return a successful structured response
                return SearchResponse(
                    query=query,
                    results=results,
                    metadata=SearchMetadata(
                        total_results=len(results),
                        language=self.lang,
                        country=self.country,
                    ),
                )

            if retry_count < self.max_retries:
                # 所有引擎都失败，等待并重试
                res = f"所有搜索引擎都失败。等待 {self.retry_delay} 秒后重试 {retry_count + 1}/{self.max_retries}..."
                logger.warning(res)
                time.sleep(self.retry_delay)
            else:
                res = f"所有搜索引擎在 {self.max_retries} 次重试后都失败。放弃。"
                logger.error(res)
                # 返回错误响应
                return SearchResponse(
                    query=query,
                    error="所有搜索引擎在多次重试后都未能返回结果。",
                    results=[],
                )

    async def _try_all_engines(
        self, query: str, num_results: int, search_params: Dict[str, Any]
    ) -> List[SearchResult]:
        """按配置的顺序尝试所有搜索引擎。"""
        engine_order = self._get_engine_order()
        failed_engines = []

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            logger.info(f"🔎 Attempting research with {engine_name.capitalize()}...")
            try:
                search_items = await self._perform_search_with_engine(
                    engine, query, num_results, search_params
                )

                if not search_items:
                    failed_engines.append(engine_name)
                    logger.warning(f"{engine_name.capitalize()} 返回空结果，尝试下一个引擎...")
                    continue

                if failed_engines:
                    logger.info(
                        f"搜索成功，使用 {engine_name.capitalize()}，在尝试后: {', '.join(failed_engines)}"
                    )

                # 将搜索项转换为结构化结果
                return [
                    SearchResult(
                        position=i + 1,
                        url=item.url,
                        title=item.title
                        or f"结果 {i+1}",  # 确保我们始终有一个标题
                        description=item.description or "",
                        source=engine_name,
                    )
                    for i, item in enumerate(search_items)
                ]
            except Exception as e:
                failed_engines.append(engine_name)
                logger.warning(f"{engine_name.capitalize()} 搜索失败: {str(e)}，尝试下一个引擎...")
                continue

        if failed_engines:
            logger.error(f"所有搜索引擎都失败: {', '.join(failed_engines)}")
        return []

    async def _fetch_content_for_results(
            self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Fetch and add web content to research results."""
        if not results:
            return []

        # Create tasks for each result
        # fetched_results = [await self._fetch_single_result_content(result) for result in results]
        fetched_results = await asyncio.gather(
            *[self._fetch_single_result_content(result) for result in results]
        )

        # Explicit validation of return type
        return [
            (
                result
                if isinstance(result, SearchResult)
                else SearchResult(**result.model_dump())
            )
            for result in fetched_results
        ]

    async def _fetch_single_result_content(self, result: SearchResult) -> SearchResult:
        """Fetch content for a single research result."""
        if result.url:
            res = await self.content_fetcher.forward(result.url)
            content = res.text_content
            if content:
                if len(content) > self.max_length:
                    content = content[: self.max_length] + "..."
                result.raw_content = content
        return result

    def _get_engine_order(self) -> List[str]:
        """Determines the order in which to try research engines."""
        preferred = (
            self.engine if self.engine else "firecrawl"
        )
        fallbacks = [engine for engine in self.fallback_engines]

        # Start with preferred engine, then fallbacks, then remaining engines
        engine_order = [preferred] if preferred in self._search_engine else []
        engine_order.extend(
            [
                fb
                for fb in fallbacks
                if fb in self._search_engine and fb not in engine_order
            ]
        )
        engine_order.extend([e for e in self._search_engine if e not in engine_order])

        return engine_order

    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
        search_params: Dict[str, Any],
    ) -> List[SearchItem]:
        """Execute research with the given engine and parameters."""
        try:
            results = [result
                for result in await engine.perform_search(
                    query,
                    num_results=num_results,
                    lang=search_params.get("lang"),
                    country=search_params.get("country"),
                    filter_year=search_params.get("filter_year"),
                )
            ]
            return results
        except Exception as e:
            # 记录错误但不抛出，让调用者可以尝试下一个引擎
            logger.warning(f"搜索引擎执行失败: {type(e).__name__}: {str(e)}")
            return []

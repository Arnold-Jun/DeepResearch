from __future__ import annotations

from typing import Any, List

from src.models import model_manager
from src.registry import TOOL
from src.tools.tools import Tool
from src.tools.research.research_tools import OptimizeQueryTool
from src.tools.research.web_searcher import WebSearcherTool


@TOOL.register_module(name="smart_search_tool")
class SmartSearchTool(Tool):
    name = "smart_search_tool"
    description = (
        "Perform a smart web search. This tool first optimizes the query using an LLM to "
        "improve search results, then executes the search. Returns the search results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The initial search query.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, model_id: str | None = None, searcher_config: dict[str, Any] | None = None):
        super().__init__()
        self.model_id = model_id or "qwen3-14b"
        self.optimizer = OptimizeQueryTool(model_id=self.model_id)
        
        # Configure internal web searcher
        # If no config provided, use defaults but ensure fetch_content is True as per original agent logic
        if searcher_config is None:
            self.searcher = WebSearcherTool()
            self.searcher.fetch_content = True
        else:
            self.searcher = WebSearcherTool(**searcher_config)

    async def forward(self, query: str) -> Any:
        # 1. Optimize
        opt_res = await self.optimizer.forward(query=query)
        if opt_res.error:
            # Fallback to original query if optimization fails
            optimized_query = query
        else:
            optimized_query = opt_res.output.get("optimized_query") or query
            
        # 2. Search
        search_res = await self.searcher.forward(query=optimized_query)
        
        # 3. Return combined information
        # We return the original search result object but attach optimization info to metadata if possible,
        # or simplified structure. The tool executor reducer will handle the output.
        # Let's return a dict to be explicit.
        
        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "search_results": getattr(search_res, "results", []) or getattr(search_res, "output", []),
            "error": search_res.error
        }

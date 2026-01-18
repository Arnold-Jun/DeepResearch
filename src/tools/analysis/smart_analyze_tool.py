from __future__ import annotations

from typing import Any, List

from src.registry import TOOL
from src.tools.tools import Tool
from src.tools.analysis.analysis_tools import ParallelAnalyzeTool


@TOOL.register_module(name="smart_analyze_tool")
class SmartAnalyzeTool(Tool):
    name = "smart_analyze_tool"
    description = (
        "Perform deep analysis on text content using one or multiple AI models. "
        "Returns structured analysis results."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The analysis task or question.",
            },
            "source": {
                "type": "string",
                "description": "The text content to analyze.",
            },
        },
        "required": ["task", "source"],
    }

    def __init__(self, model_id: str | None = None, analyzer_model_ids: List[str] | None = None):
        super().__init__()
        # If specific analyzer IDs are provided, use them. 
        # Otherwise, fall back to the single model_id provided.
        ids = analyzer_model_ids
        if not ids:
            ids = [model_id or "qwen3-14b"]
            
        self.analyzer = ParallelAnalyzeTool(analyzer_model_ids=ids)

    async def forward(self, task: str, source: str) -> Any:
        # Delegate directly to the parallel analyzer
        return await self.analyzer.forward(task=task, source=source)

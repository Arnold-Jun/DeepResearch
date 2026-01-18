from __future__ import annotations

from typing import Any, List

from src.registry import TOOL
from src.tools.tools import Tool
from src.tools.research.research_tools import BatchExtractInsightsTool, GenerateFollowUpsTool


@TOOL.register_module(name="insight_reflector_tool")
class InsightReflectorTool(Tool):
    name = "insight_reflector_tool"
    description = (
        "Extract insights from search results and generate follow-up questions. "
        "Use this tool after performing a search to process the findings and plan next steps."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query used for the search.",
            },
            "search_results": {
                "type": "array",
                "items": {"type": "object"},
                "description": "The search results to analyze.",
            },
            "original_task": {
                "type": "string",
                "description": "The original user task (for context in generating follow-ups).",
            },
            "existing_insights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of insights already extracted (to avoid duplicates).",
            },
        },
        "required": ["query", "search_results"],
    }

    def __init__(self, model_id: str | None = None):
        super().__init__()
        self.model_id = model_id or "qwen3-14b"
        self.extractor = BatchExtractInsightsTool(model_id=self.model_id)
        self.planner = GenerateFollowUpsTool(model_id=self.model_id)

    async def forward(
        self, 
        query: str, 
        search_results: List[Any], 
        original_task: str | None = None,
        existing_insights: List[str] | None = None
    ) -> Any:
        # 1. Extract Insights
        extract_res = await self.extractor.forward(query=query, search_results=search_results)
        new_insights = extract_res.output or []
        if not isinstance(new_insights, list):
            new_insights = []
            
        # Prepare context for follow-up generation
        # We need to mix new insights with existing ones to give the model full context
        current_insight_texts = []
        for it in new_insights:
             if isinstance(it, dict):
                current_insight_texts.append(it.get("content", ""))
             else:
                current_insight_texts.append(str(it))
        
        all_insight_texts = (existing_insights or []) + current_insight_texts
        
        # 2. Generate Follow-ups
        # We utilize the newly found insights immediately to generate better questions
        followup_res = await self.planner.forward(
            original_query=original_task or query,
            current_query=query,
            insights=all_insight_texts[-10:] if all_insight_texts else [] # limit context
        )
        
        followups = followup_res.output or []
        
        return {
            "insights": new_insights,
            "new_followups": followups,
            "extraction_error": extract_res.error,
            "followup_error": followup_res.error
        }

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DecisionSchemaTool:
    """
    Schema-only tool object used for LLM tool-calling.

    This is intentionally NOT a `src.tools.Tool` to avoid importing heavy/optional dependencies.
    The model layer only needs `name/description/parameters`.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    # Never executed; kept for completeness/debugging.
    def forward(
        self,
        route: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict[str, Any]] = None,
        llm_node: Optional[str] = None,
        llm_args: Optional[dict[str, Any]] = None,
        final_answer: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "route": route,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "llm_node": llm_node,
            "llm_args": llm_args,
            "final_answer": final_answer,
        }


__all__ = ["DecisionSchemaTool"]



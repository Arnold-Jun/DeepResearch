from typing import List, Any, Optional
from pydantic import BaseModel, Field

class SearchItem(BaseModel):
    """Represents a single research result item"""

    title: str = Field(description="The title of the research result")
    url: str = Field(description="The URL of the research result")
    date: Optional[str] = None
    position: Optional[int] = None
    source: Optional[str] = None
    description: Optional[str] = None

    def __str__(self) -> str:
        """String representation of a research result item."""
        return f"{self.title} - {self.url} - {self.description or 'No description available'}"

class WebSearchEngine(BaseModel):
    """Base class for web research engines."""

    model_config = {"arbitrary_types_allowed": True}

    async def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Perform a web research and return a list of research items.

        Args:
            query (str): The research query to submit to the research engine.
            num_results (int, optional): The number of research results to return. Default is 10.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            List[SearchItem]: A list of SearchItem objects matching the research query.
        """
        raise NotImplementedError
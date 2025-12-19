from baidusearch.baidusearch import search

from src.tools.research.base import WebSearchEngine, SearchItem

class BaiduSearchEngine(WebSearchEngine):
    async def perform_search(self, query: str, num_results: int = 10, *args, **kwargs):
        """
        Baidu research engine.

        Returns results formatted according to SearchItem model.
        """
        raw_results = search(query, num_results=num_results)

        # Convert raw results to SearchItem format
        results = []
        for i, item in enumerate(raw_results):
            url = None
            title = None
            description = None

            if isinstance(item, str):
                # If it's just a URL
                url = item
                title = f"Baidu Result {i+1}"
            elif isinstance(item, dict):
                # If it's a dictionary with details
                url = item.get("url", "")
                title = item.get("title", f"Baidu Result {i+1}")
                description = item.get("abstract", None)
            else:
                # Try to get attributes directly
                try:
                    url = getattr(item, "url", "")
                    title = getattr(item, "title", f"Baidu Result {i+1}")
                    description = getattr(item, "abstract", None)
                except Exception:
                    # Fallback to a basic result
                    url = str(item)
                    title = f"Baidu Result {i+1}"

            # 确保 URL 是完整的（如果不是，添加 https://www.baidu.com 前缀）
            if url and not url.startswith(("http://", "https://", "file://", "raw:")):
                if url.startswith("/"):
                    url = "https://www.baidu.com" + url
                else:
                    url = "https://www.baidu.com/" + url

            if url:
                results.append(
                    SearchItem(title=title, url=url, description=description)
                )

        return results
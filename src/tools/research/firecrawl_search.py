import os
import time
import asyncio
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv(verbose=True)

try:
    from firecrawl import Firecrawl
    FIRECRAWL_NEW_API = True
except ImportError:
    try:
        from firecrawl import FirecrawlApp
        FIRECRAWL_NEW_API = False
    except ImportError:
        raise ImportError("Firecrawl package not found. Please install it with: pip install firecrawl-py")

from src.tools.research.base import WebSearchEngine, SearchItem

def search(params):
    logger = logging.getLogger(__name__)
    
    try:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 Firecrawl API key。请设置环境变量 FIRECRAWL_API_KEY。"
                "例如：export FIRECRAWL_API_KEY='your-api-key-here'"
            )
        query = params.get("q", "")
        logger.info(f"Firecrawl searching for: {query}")
        
        app = Firecrawl(api_key=api_key) if FIRECRAWL_NEW_API else FirecrawlApp(api_key=api_key)
        search_params = {"query": query, "limit": params.get("num", 10)}
        
        try:
            if not hasattr(app, 'research'):
                logger.error("Firecrawl API does not have 'research' method")
                return []
            response = app.search(**search_params)
        except Exception as api_error:
            logger.error(f"Firecrawl API call failed: {str(api_error)}", exc_info=True)
            return []

        if not response:
            logger.warning("Firecrawl returned None response")
            return []
        
        data = None
        if isinstance(response, dict):
            data = response.get("web", []) or response.get("data", [])
        elif hasattr(response, 'web'):
            data = response.web
        elif hasattr(response, 'data'):
            data = response.data
        elif hasattr(response, '__dict__'):
            response_dict = response.__dict__
            web_data = response_dict.get("web")
            if web_data is not None:
                if isinstance(web_data, list):
                    data = web_data
                elif isinstance(web_data, dict):
                    data = web_data.get("results", []) or web_data.get("data", []) or web_data.get("items", [])
                elif hasattr(web_data, '__dict__'):
                    web_dict = web_data.__dict__
                    data = web_dict.get("results", []) or web_dict.get("data", []) or web_dict.get("items", [])
                elif hasattr(web_data, 'results'):
                    data = web_data.results
                else:
                    data = []
            else:
                data = response_dict.get("data", [])
        elif isinstance(response, list):
            data = response
        else:
            logger.warning(f"Firecrawl response format unexpected: {type(response)}")
            return []

        if not data or not isinstance(data, list):
            logger.warning(f"Firecrawl returned invalid data for query: {query}")
            return []
        
        results = []
        for i, item in enumerate(data):
            title = url = description = ""
            
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("heading") or item.get("titleText") or ""
                url = item.get("url") or item.get("link") or item.get("href") or item.get("urlLink") or item.get("sourceURL") or ""
                description = item.get("description") or item.get("snippet") or item.get("abstract") or item.get("summary") or item.get("content") or item.get("text") or ""
            elif hasattr(item, 'url') or hasattr(item, '__dict__'):
                title = getattr(item, 'title', "") or getattr(item, 'name', "") or getattr(item, 'heading', "") or ""
                url = getattr(item, 'url', "") or getattr(item, 'link', "") or getattr(item, 'href', "") or ""
                description = getattr(item, 'description', "") or getattr(item, 'snippet', "") or getattr(item, 'abstract', "") or getattr(item, 'summary', "") or ""
                
                if not url and hasattr(item, '__dict__'):
                    item_dict = item.__dict__
                    title = title or item_dict.get("title", "") or item_dict.get("name", "")
                    url = url or item_dict.get("url", "") or item_dict.get("link", "")
                    description = description or item_dict.get("description", "") or item_dict.get("snippet", "")
            else:
                logger.warning(f"Skipping item {i} with unsupported type: {type(item)}")
                continue
            
            if url:
                results.append(SearchItem(title=title, url=url, description=description))

        logger.info(f"Firecrawl found {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Firecrawl research failed for query '{params.get('q', '')}': {str(e)}", exc_info=True)
        return []

class FirecrawlSearchEngine(WebSearchEngine):
    async def perform_search(
        self,
        query: str,
        num_results: int = 10,
        filter_year: int = None,
        *args, **kwargs
    ) -> List[SearchItem]:
        params = {"q": query, "num": num_results}
        return search(params)


if __name__ == '__main__':
    start_time = time.time()
    search_engine = FirecrawlSearchEngine()
    results = asyncio.run(search_engine.perform_search("OpenAI GPT-4", num_results=5))
    for item in results:
        print(f"Title: {item.title}\nURL: {item.url}\nDescription: {item.description}\n")
    print(f"{time.time() - start_time} seconds elapsed for research")
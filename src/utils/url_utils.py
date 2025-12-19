import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv(verbose=True)

from markitdown._base_converter import DocumentConverterResult
from crawl4ai import AsyncWebCrawler
from firecrawl import FirecrawlApp

async def firecrawl_fetch_url(url: str):
    try:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 Firecrawl API key。请设置环境变量 FIRECRAWL_API_KEY。"
                "例如：export FIRECRAWL_API_KEY='your-api-key-here'"
            )
        app = FirecrawlApp(api_key=api_key)
        response = app.scrape_url(url)
        return response.markdown
    except Exception:
        return None

async def fetch_crawl4ai_url(url: str):
    try:
        async with AsyncWebCrawler() as crawler:
            response = await crawler.arun(url=url)
            return response.markdown if response else None
    except Exception:
        return None

async def fetch_url(url: str) -> Optional[DocumentConverterResult]:
    try:
        firecrawl_result = await firecrawl_fetch_url(url)
        if firecrawl_result:
            return DocumentConverterResult(markdown=firecrawl_result, title=f"Fetched content from {url}")

        crawl4ai_result = await fetch_crawl4ai_url(url)
        if crawl4ai_result:
            return DocumentConverterResult(markdown=crawl4ai_result, title=f"Fetched content from {url}")
    except Exception:
        return None

if __name__ == '__main__':
    import asyncio
    url = "https://www.google.com/"
    result = asyncio.run(firecrawl_fetch_url(url))
    print(result)
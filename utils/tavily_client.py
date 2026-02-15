"""
Tavily web search client — simple, synchronous, robust.

Returns results in a consistent dict format.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TavilySearchClient:
    """Simple Tavily web search wrapper."""

    def __init__(self) -> None:
        self._api_key = os.getenv("TAVILY_API_KEY", "")
        self._client = None

        if not self._api_key:
            logger.warning("TAVILY_API_KEY not set — web searches will fail.")
            return

        try:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self._api_key)
        except ImportError:
            logger.error("tavily-python not installed.")

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Run a web search. Returns list of result dicts."""
        if not self._client:
            logger.error("Tavily client unavailable.")
            return []

        try:
            logger.info("Tavily search: '%s' (max=%d)", query[:60], max_results)
            raw = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )
            results = []
            for item in raw.get("results", []):
                results.append({
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:2000],
                    "published_date": item.get("published_date", ""),
                    "relevance_score": item.get("score", 0.5),
                    "source_type": "web",
                })
            logger.info("Tavily returned %d results for '%s'", len(results), query[:40])
            return results

        except Exception as e:
            logger.error("Tavily search failed for '%s': %s", query[:40], e)
            return []

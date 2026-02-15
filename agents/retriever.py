"""
Contextual Retriever Agent — fetches web sources for sub-queries.

Agent 2 of 5 in the Deep Researcher pipeline.
Uses Tavily for web search. Purely synchronous — no asyncio.
"""

from __future__ import annotations

import logging
from typing import Any

from utils.tavily_client import TavilySearchClient

logger = logging.getLogger(__name__)
_tavily = TavilySearchClient()


def retrieve_sources(
    sub_queries: list[str],
    tavily_per_query: int = 5,
    arxiv_per_query: int = 0,
    uploaded_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve web sources for each sub-query.

    Simple synchronous loop — searches each sub-query via Tavily,
    merges uploaded docs, deduplicates by URL.

    Args:
        sub_queries: List of sub-query strings.
        tavily_per_query: Max web results per query.
        arxiv_per_query: Ignored (Arxiv disabled).
        uploaded_docs: Optional list of pre-parsed uploaded documents.

    Returns:
        List of source dicts with title, url, content, source_type, etc.
    """
    effective = sub_queries[:5]  # cap queries
    logger.info("Retrieving sources for %d sub-queries.", len(effective))

    all_results: list[dict[str, Any]] = []

    # Merge uploaded documents first (highest priority)
    if uploaded_docs:
        all_results.extend(uploaded_docs)
        logger.info("Added %d uploaded document(s).", len(uploaded_docs))

    # Search each sub-query
    for query in effective:
        try:
            results = _tavily.search(query, max_results=tavily_per_query)
            all_results.extend(results)
            logger.info("Got %d results for: '%s'", len(results), query[:50])
        except Exception as e:
            logger.error("Search failed for '%s': %s", query[:50], e)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for doc in all_results:
        url = doc.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(doc)

    logger.info("Retrieved %d unique sources total.", len(deduped))
    return deduped

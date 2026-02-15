"""
Query Decomposer Agent — breaks a broad research topic into focused sub-queries.

Agent 1 of 5 in the Deep Researcher pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
_llm = LLMClient()

SYSTEM_PROMPT = """You are a research query decomposer. Given a broad research topic,
break it down into 3-5 specific, focused sub-questions that together cover the topic
comprehensively.

Return ONLY a JSON array of strings, like:
["sub-question 1", "sub-question 2", "sub-question 3"]

Rules:
- Each sub-question should be specific and searchable
- Cover different angles of the topic
- Use plain language suitable for web search
- Return ONLY the JSON array, no other text"""


def decompose_query(topic: str) -> list[str]:
    """Break a research topic into 3-5 sub-queries.

    Args:
        topic: Broad research topic string.

    Returns:
        List of focused sub-query strings.
    """
    try:
        response = _llm.call(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Decompose this research topic: {topic}"},
            ],
            model_tier="low_cost",
            task_name="query_decomposition",
        )

        # Try to parse JSON
        text = response.strip()
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        queries = json.loads(text)
        if isinstance(queries, list) and len(queries) > 0:
            logger.info("Decomposed topic into %d sub-queries.", len(queries))
            return queries[:5]  # Cap at 5

    except Exception as e:
        logger.warning("Query decomposition failed: %s. Using topic as-is.", e)

    # Fallback: return topic as a single query
    return [topic]

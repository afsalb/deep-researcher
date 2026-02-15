"""
Critical Analysis Agent — summarizes, validates, and scores retrieved sources.

Agent 3 of 5 in the Deep Researcher pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
_llm = LLMClient()

SYSTEM_PROMPT = """You are a critical research analyst. Given a set of source documents,
you must:
1. Summarize each source briefly (1-2 sentences)
2. Score each source's credibility (0.0 to 1.0)
3. Identify any contradictions between sources
4. Filter out low-credibility sources

Return ONLY a JSON object with this structure:
{
  "summaries": [
    {"title": "...", "summary": "...", "credibility": 0.9}
  ],
  "contradictions": [
    "Source A says X, but Source B says Y"
  ]
}

Rules:
- Be objective and evidence-based
- Score credibility based on source reliability and content quality
- Only flag genuine contradictions, not minor differences
- Return ONLY the JSON object, no other text"""


def analyze_sources(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze and validate retrieved sources.

    Args:
        docs: List of source dicts from the retriever.

    Returns:
        Dict with summaries, contradictions, and credible_docs.
    """
    if not docs:
        return {"summaries": [], "contradictions": [], "credible_docs": []}

    # Build a text representation of sources for the LLM
    source_text = ""
    for i, doc in enumerate(docs[:15], 1):  # cap at 15
        source_text += f"\n--- Source {i} ---\n"
        source_text += f"Title: {doc.get('title', 'Unknown')}\n"
        source_text += f"URL: {doc.get('url', 'N/A')}\n"
        source_text += f"Type: {doc.get('source_type', 'unknown')}\n"
        source_text += f"Content: {doc.get('content', '')[:500]}\n"

    try:
        response = _llm.call(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze these sources:\n{source_text}"},
            ],
            model_tier="low_cost",
            task_name="source_analysis",
        )

        # Parse JSON
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        analysis = json.loads(text)

        # Filter credible docs (score >= 0.4)
        credible_docs = []
        summaries = analysis.get("summaries", [])
        for i, doc in enumerate(docs[:len(summaries)]):
            score = summaries[i].get("credibility", 0.5) if i < len(summaries) else 0.5
            if score >= 0.4:
                credible_docs.append(doc)

        # If filtering removed everything, keep all docs
        if not credible_docs:
            credible_docs = list(docs)

        result = {
            "summaries": summaries,
            "contradictions": analysis.get("contradictions", []),
            "credible_docs": credible_docs,
        }

        logger.info(
            "Analysis: %d summaries, %d contradictions, %d credible docs.",
            len(result["summaries"]),
            len(result["contradictions"]),
            len(result["credible_docs"]),
        )
        return result

    except Exception as e:
        logger.error("Analysis failed: %s. Returning all docs as credible.", e)
        return {
            "summaries": [],
            "contradictions": [],
            "credible_docs": list(docs),
        }

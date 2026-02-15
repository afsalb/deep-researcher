"""
Insight Generation Agent — produces hypotheses, trends, and research gaps.

Agent 4 of 5 in the Deep Researcher pipeline.
Uses the high-reasoning (expensive) model for deeper analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
_llm = LLMClient()

SYSTEM_PROMPT = """You are an advanced research insight generator. Given an analysis
of research sources, you must produce deep insights.

Return ONLY a JSON object with this structure:
{
  "insights": ["key insight 1", "key insight 2", ...],
  "hypotheses": ["testable hypothesis 1", ...],
  "trends": ["emerging trend 1", ...],
  "gaps": ["research gap 1", ...]
}

Rules:
- Insights should be non-obvious conclusions from the data
- Hypotheses should be testable and specific
- Trends should be forward-looking
- Gaps should identify what research is missing
- Provide 2-4 items per category
- Return ONLY the JSON object, no other text"""


def generate_insights(
    analysis: dict[str, Any],
    topic: str,
) -> dict[str, Any]:
    """Generate deep insights from the analysis.

    Args:
        analysis: Analysis dict from the analyzer (summaries, contradictions).
        topic: Original research topic string.

    Returns:
        Dict with insights, hypotheses, trends, gaps.
    """
    # Build context from analysis
    context_parts = [f"Research Topic: {topic}\n"]

    summaries = analysis.get("summaries", [])
    if summaries:
        context_parts.append("Source Summaries:")
        for s in summaries:
            context_parts.append(f"- {s.get('title', '?')}: {s.get('summary', 'N/A')}")

    contradictions = analysis.get("contradictions", [])
    if contradictions:
        context_parts.append("\nContradictions Found:")
        for c in contradictions:
            context_parts.append(f"- {c}")

    context = "\n".join(context_parts)

    try:
        response = _llm.call(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate insights for:\n{context}"},
            ],
            model_tier="high_reasoning",
            task_name="insight_generation",
        )

        # Parse JSON
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        logger.info(
            "Generated: %d insights, %d hypotheses, %d trends, %d gaps.",
            len(result.get("insights", [])),
            len(result.get("hypotheses", [])),
            len(result.get("trends", [])),
            len(result.get("gaps", [])),
        )
        return result

    except Exception as e:
        logger.error("Insight generation failed: %s", e)
        return {
            "insights": ["Analysis could not be completed due to an error."],
            "hypotheses": [],
            "trends": [],
            "gaps": [],
        }

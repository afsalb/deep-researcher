"""
Follow-up Agent — generates suggested follow-up questions.

After each chat response, this agent suggests 3 relevant follow-up questions
to keep the research conversation going. Captures the user's curiosity loop.
"""

from __future__ import annotations

import json
import logging

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
_llm = LLMClient()

SUGGESTION_PROMPT = """You are a research assistant. Given the user's last question
and the assistant's answer, suggest 3 relevant follow-up questions.

User Question: "{question}"
Assistant Answer: "{answer}"

Rules:
- Questions should dig deeper into the topic.
- Questions should be specific, not generic.
- Return ONLY a JSON array of strings: ["Question 1?", "Question 2?", "Question 3?"]
"""


def suggest_followups(
    question: str,
    answer: str,
) -> list[str]:
    """Generate 3 follow-up suggestions based on the last turn."""
    try:
        response = _llm.call(
            messages=[
                {"role": "system", "content": "You are a suggestion engine."},
                {"role": "user", "content": SUGGESTION_PROMPT.format(
                    question=question,
                    answer=answer[:1000]  # truncate answer for context
                )},
            ],
            model_tier="classification",  # Use free/cheap model
            task_name="followup_suggestions",
        )

        # Parse JSON
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
            
        suggestions = json.loads(text)
        if isinstance(suggestions, list) and len(suggestions) > 0:
            return suggestions[:3]

    except Exception as e:
        logger.warning("Follow-up suggestion failed: %s", e)
    
    return []

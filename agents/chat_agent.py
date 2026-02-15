"""
Chat Agent — handles conversational follow-up questions.

This agent is responsible for:
1. Classifying user intent (answer from context vs. new search).
2. Answering questions using existing research context (cheap/fast).
3. Synthesizing answers from new research findings.

It acts as the "router" for the chat phase of the application.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
_llm = LLMClient()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """You are a research assistant router. Classify the user's
follow-up question based on the research context provided.

Context: Report on "{topic}".
User Question: "{question}"
Has Attachments: {has_attachments}

Determine the intent:
- "answer_from_context": Question can be answered using the existing report/findings.
- "needs_web_search": Question asks for NEW information not in the report.
- "needs_attachment_analysis": User uploaded a new file that must be analyzed.

Return ONLY the classification string.
"""

ANSWER_PROMPT = """You are a helpful research assistant. Answer the user's follow-up
question based ONLY on the provided research context.

Research Context:
{context}

User Question: {question}

Rules:
- Be concise and direct.
- Cite sources if possible (e.g., [Source 1]).
- If the answer is not in the context, admit it and suggest a new search.
- Do not hallucinate information not in the context.
"""

SYNTHESIS_PROMPT = """You are a research assistant. Synthesize a response to the user's
question using the NEW findings from the latest search/analysis.

Original Topic: {topic}
User Question: {question}

New Findings:
{new_findings}

Provide a clear, evidence-based answer.
"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def classify_intent(
    question: str,
    topic: str,
    has_attachments: bool,
) -> str:
    """Determine how to handle the follow-up question.

    Args:
        question: User's follow-up question.
        topic: Original research topic.
        has_attachments: True if user uploaded files with this message.

    Returns:
        "answer_from_context" | "needs_web_search" | "needs_attachment_analysis"
    """
    if has_attachments:
        return "needs_attachment_analysis"

    try:
        response = _llm.call(
            messages=[
                {"role": "system", "content": "You are a classifier."},
                {"role": "user", "content": CLASSIFICATION_PROMPT.format(
                    topic=topic,
                    question=question,
                    has_attachments=has_attachments,
                )},
            ],
            model_tier="classification",  # Use free/cheap model
            task_name="chat_classification",
        )
        intent = response.strip().lower().replace('"', '').replace("'", "")
        
        valid = {"answer_from_context", "needs_web_search", "needs_attachment_analysis"}
        if intent not in valid:
            logger.warning("Invalid classification '%s', defaulting to web search", intent)
            return "needs_web_search"
            
        logger.info("Chat intent classified as: %s", intent)
        return intent

    except Exception as e:
        logger.error("Classification failed: %s", e)
        return "needs_web_search"


def answer_from_report(
    question: str,
    report_text: str,
    source_summaries: list[dict],
) -> str:
    """Answer using existing research context (cheap)."""
    # Compress context
    context = f"Report:\n{report_text[:10000]}\n\nSource Summaries:\n"
    for s in source_summaries[:10]:
        context += f"- {s.get('title', '?')}: {s.get('summary', '')}\n"

    try:
        return _llm.call(
            messages=[
                {"role": "system", "content": ANSWER_PROMPT.format(
                    context=context,
                    question=question
                )},
            ],
            model_tier="low_cost",
            task_name="chat_answer_context",
        )
    except Exception as e:
        logger.error("Context answer failed: %s", e)
        return "I'm sorry, I couldn't answer that from the report. Please try searching."


def synthesize_answer(
    question: str,
    topic: str,
    new_analysis: dict[str, Any],
) -> str:
    """Synthesize answer from NEW search/analysis findings."""
    # Format new findings
    findings = ""
    for s in new_analysis.get("summaries", []):
        findings += f"- {s.get('title', '?')}: {s.get('summary', '')}\n"
    
    for c in new_analysis.get("contradictions", []):
        findings += f"Contradiction: {c}\n"

    try:
        return _llm.call(
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT.format(
                    topic=topic,
                    question=question,
                    new_findings=findings,
                )},
            ],
            model_tier="low_cost",
            task_name="chat_answer_synthesis",
        )
    except Exception as e:
        logger.error("Synthesis answer failed: %s", e)
        return "I found new information but failed to summarize it. Please check the sources."


RELEVANCE_PROMPT = """You are a research relevance filter. Determine if the provided document
content is relevant to the research topic.

Topic: "{topic}"
Document Content (snippet):
"{content}"

Return ONLY "YES" if relevant or "NO" if irrelevant.
"""

def check_relevance(topic: str, content: str) -> bool:
    """Check if document content is relevant to the topic."""
    # Quick check for short content
    if len(content) < 50:
        return False
        
    try:
        response = _llm.call(
            messages=[
                {"role": "system", "content": "You are a relevance filter."},
                {"role": "user", "content": RELEVANCE_PROMPT.format(
                    topic=topic, 
                    content=content[:2000] # Check first 2k chars
                )},
            ],
            model_tier="classification",
            task_name="relevance_check",
        )
        return "YES" in response.strip().upper()
    except Exception as e:
        logger.warning("Relevance check failed: %s", e)
        return True # Default to allow if check fails

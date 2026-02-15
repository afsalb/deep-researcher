"""
Report Builder Agent — compiles all findings into a structured Markdown report.

Agent 5 of 5 in the Deep Researcher pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)
_llm = LLMClient()

FULL_REPORT_PROMPT = """You are a professional research report writer. Given the research
topic, source analysis, and generated insights, compile a comprehensive, detailed, and
long-form Markdown report.

The report MUST be VERY DETAILED (explore every angle) and include these sections with Markdown headers:

# [Topic Name]

## Executive Summary
Briefly summarize the report (1 paragraph), but this is NOT the main summary. The main summary is generated separately. This is just an intro.

## Key Findings
Detailed findings from the source analysis, organized by theme.
Ideally 500+ words.

## Analysis
Critical analysis of the sources, including contradictions and credibility assessment.
Compare and contrast different viewpoints.

## Insights & Hypotheses
The generated insights, testable hypotheses, emerging trends, and research gaps.
Elaborate on each point.

## Conclusions
Summary conclusions and recommendations.

## References
Numbered list of all sources used.

Rules:
- Write in a professional, academic tone.
- Cite sources by number [1], [2], etc.
- AIM FOR MAXIMUM DETAIL. Do not summarize; expand on findings.
- This is the FULL REPORT. It should be lengthy and comprehensive.
"""
SUMMARY_PROMPT = """You are an executive assistant. You are given a FULL RESEARCH REPORT.
Your task is to write a standalone EXECUTIVE SUMMARY (2-3 paragraphs) based *only* on the report.

Research Report:
{report_text}

Rules:
- High-level overview only.
- No headers, just paragraphs.
- Capture the most important findings.
- Do NOT say "See full report". Write the actual summary.
"""

BIBTEX_PROMPT = """Generate BibTeX citations for these web sources. Return ONLY the raw
BibTeX entries, no other text. Use @misc for web pages.

Sources:
{sources}"""


def build_full_report(
    topic: str,
    analysis: dict[str, Any],
    insights: dict[str, Any],
    docs: list[dict[str, Any]],
) -> dict[str, str]:
    """Compile the FULL Detailed Research Report only.

    Args:
        topic: Original research topic.
        analysis: Analysis dict.
        insights: Insights dict.
        docs: List of source documents.

    Returns:
        Dict with 'markdown' (full report) and 'bibtex'.
    """
    # Build the context for report generation
    context_parts = [f"# Research Topic: {topic}\n"]
    
    # Source summaries
    summaries = analysis.get("summaries", [])
    if summaries:
        context_parts.append("## Source Analysis:")
        for i, s in enumerate(summaries, 1):
            context_parts.append(
                f"[{i}] {s.get('title', '?')} — {s.get('summary', 'N/A')} "
                f"(Credibility: {s.get('credibility', 'N/A')})"
            )

    # Contradictions
    contradictions = analysis.get("contradictions", [])
    if contradictions:
        context_parts.append("\n## Contradictions:")
        for c in contradictions:
            context_parts.append(f"- {c}")

    # Insights
    for key in ["insights", "hypotheses", "trends", "gaps"]:
        items = insights.get(key, [])
        if items:
            context_parts.append(f"\n## {key.title()}:")
            for item in items:
                context_parts.append(f"- {item}")

    # Sources list
    context_parts.append("\n## Sources Used:")
    for i, doc in enumerate(docs, 1):
        context_parts.append(
            f"[{i}] {doc.get('title', 'Unknown')} — {doc.get('url', 'N/A')}"
        )

    context = "\n".join(context_parts)

    # 1. Generate FULL REPORT
    full_report = ""
    try:
        full_report = _llm.call(
            messages=[
                {"role": "system", "content": FULL_REPORT_PROMPT},
                {"role": "user", "content": context},
            ],
            model_tier="low_cost",
            task_name="report_generation_full",
        )
        logger.info("Full Report generated: %d characters.", len(full_report))
    except Exception as e:
        logger.error("Full Report generation failed: %s", e)
        full_report = _build_fallback_report(topic, analysis, insights, docs)

    # Generate BibTeX
    bibtex = _generate_bibtex(docs)

    return {"markdown": full_report, "bibtex": bibtex}


def generate_summary(full_report: str) -> str:
    """Generate Executive Summary from the Full Report."""
    try:
        # Truncate to avoid context limit (approx 15k chars)
        summary_context = full_report[:15000] 
        
        summary = _llm.call(
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT.format(report_text=summary_context)},
            ],
            model_tier="low_cost",
            task_name="report_generation_summary",
        )
        logger.info("Summary generated: %d characters.", len(summary))
        return summary
    except Exception as e:
        logger.error("Summary generation failed: %s", e)
        return (
            "> [!WARNING]\n"
            "> **Summary Generation Failed**\n"
            "> The AI could not generate a summary. Please check the 'Full Report' tab for details, or verify your API key/connection.\n"
        )


def _generate_bibtex(docs: list[dict[str, Any]]) -> str:
    """Generate BibTeX citations from source docs."""
    if not docs:
        return ""

    source_lines = []
    for i, doc in enumerate(docs, 1):
        source_lines.append(
            f"[{i}] Title: {doc.get('title', '?')}, "
            f"URL: {doc.get('url', 'N/A')}, "
            f"Date: {doc.get('published_date', 'N/A')}"
        )

    try:
        bibtex = _llm.call(
            messages=[
                {"role": "user", "content": BIBTEX_PROMPT.format(
                    sources="\n".join(source_lines)
                )},
            ],
            model_tier="low_cost",
            task_name="bibtex_generation",
        )
        return bibtex
    except Exception as e:
        logger.error("BibTeX generation failed: %s", e)
        return ""


def _build_fallback_report(
    topic: str,
    analysis: dict[str, Any],
    insights: dict[str, Any],
    docs: list[dict[str, Any]],
) -> str:
    """Build a basic report without LLM if generation fails."""
    lines = [
        f"# Research Report: {topic}\n",
        "> [!WARNING]",
        "> **Report Generation Failed**",
        "> The AI failed to generate the full detailed report. This is often due to an invalid API key, rate limits, or network issues.",
        "> Below is a structured dump of the gathered data.\n",
        "## Gathered Findings (Fallback)\n",
        f"This report presents findings from {len(docs)} sources on the topic of {topic}.\n",
    ]

    summaries = analysis.get("summaries", [])
    if summaries:
        lines.append("## Key Findings (Raw)\n")
        for s in summaries:
            lines.append(f"- **{s.get('title', '?')}**: {s.get('summary', 'N/A')}\n")

    for key in ["insights", "hypotheses", "trends", "gaps"]:
        items = insights.get(key, [])
        if items:
            lines.append(f"\n## {key.title()}\n")
            for item in items:
                lines.append(f"- {item}\n")

    lines.append("\n## References\n")
    for i, doc in enumerate(docs, 1):
        lines.append(f"{i}. [{doc.get('title', '?')}]({doc.get('url', '')})\n")

    return "\n".join(lines)

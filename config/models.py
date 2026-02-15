"""
Configuration — model tiers, pricing, and constants.

Defines the LLM model tiers used by the research pipeline
and per-model pricing for the cost tracker.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model Tiers — each has a primary and fallback model
# ---------------------------------------------------------------------------

MODEL_TIERS: dict[str, dict] = {
    "low_cost": {
        "primary": "google/gemini-2.0-flash-001",
        "fallback": "meta-llama/llama-3.1-8b-instruct:free",
        "max_tokens": 1500,
        "temperature": 0.3,
        "use_cases": ["decomposition", "summarization", "formatting"],
    },
    "high_reasoning": {
        "primary": "anthropic/claude-3.5-sonnet",
        "fallback": "openai/gpt-4o-mini",
        "max_tokens": 2500,
        "temperature": 0.7,
        "use_cases": ["insight_generation", "hypothesis_formation"],
    },
    "classification": {
        "primary": "meta-llama/llama-3.1-8b-instruct:free",
        "fallback": "google/gemini-2.0-flash-001",
        "max_tokens": 500,
        "temperature": 0.1,
        "use_cases": ["routing", "intent_classification", "suggestion_generation"],
    },
}

# ---------------------------------------------------------------------------
# Cost Tracking — approximate USD per 1M tokens (via OpenRouter)
# ---------------------------------------------------------------------------

MODEL_COSTS: dict[str, dict[str, float]] = {
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.30},
    "meta-llama/llama-3.1-8b-instruct:free": {"input": 0.0, "output": 0.0},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUERY_MAX_LENGTH: int = 500           # max chars for a research query (UI)
CONTENT_TRUNCATION_LIMIT: int = 2000  # max chars kept per source document

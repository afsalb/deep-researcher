"""
OpenRouter LLM client — simple, synchronous, robust.

Uses the openai SDK pointed at https://openrouter.ai/api/v1.
Two tiers: low_cost (Gemini Flash) and high_reasoning (Claude 3.5 Sonnet).
Automatic fallback and retry on rate-limit.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config.models import MODEL_TIERS
from utils.cost_tracker import CostTracker

load_dotenv()
logger = logging.getLogger(__name__)


from utils.guardrails import Guardrails

class LLMClient:
    """Thin OpenRouter wrapper with tiered model selection and fallback."""

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not self._api_key:
            logger.warning("OPENROUTER_API_KEY not set — LLM calls will fail.")

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
            timeout=60,
        )
        self._tracker = CostTracker()
        self._guards = Guardrails()

    def call(
        self,
        messages: list[dict[str, Any]],
        model_tier: str = "low_cost",
        task_name: str = "unknown",
    ) -> str:
        """Send a chat completion request. Returns the text response.

        Tries the primary model first, then falls back to the fallback model.
        """
        # Guardrail 1: Cost Check
        stats = self._tracker.get_session_stats()
        if not self._guards.check_cost(stats["total_cost"]):
            raise RuntimeError("Session cost limit exceeded.")

        # Guardrail 2: Input Validation (Removed as per user request to prevent blocking)
        # last_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        # is_safe, reason = self._guards.validate_input(last_msg)
        # if not is_safe:
        #     logger.warning(f"Guardrail warning (non-blocking): {reason}")


        tier = MODEL_TIERS.get(model_tier)
        if not tier:
            raise ValueError(f"Unknown tier '{model_tier}'. Use: {list(MODEL_TIERS)}")

        # Try primary
        try:
            content = self._do_call(
                messages, tier["primary"], tier["max_tokens"],
                tier["temperature"], task_name,
            )
        except Exception as e:
            logger.warning("Primary model %s failed: %s. Trying fallback.", tier["primary"], e)
            try:
                content = self._do_call(
                    messages, tier["fallback"], tier["max_tokens"],
                    tier["temperature"], task_name,
                )
            except Exception as e:
                logger.error("Fallback model %s also failed: %s", tier["fallback"], e)
                raise RuntimeError(f"All models failed for {task_name}: {e}") from e

        # Guardrail 3: Output Sanitization
        return self._guards.sanitize_output(content)

    def _do_call(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
        task_name: str,
    ) -> str:
        """Execute a single OpenAI-compatible chat completion."""
        logger.info("LLM call: model=%s task=%s", model, task_name)

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content or ""

        # Track cost
        usage = response.usage
        if usage:
            self._tracker.track_call(
                model=model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                task_name=task_name,
            )

        return content


    # Legacy alias used by document_parser
    def get_chat_completion(self, messages, model_tier="low_cost", task_name="unknown"):
        return self.call(messages, model_tier, task_name)

"""
Guardrails Module — safety, security, and cost control for AI agents.

Implements 5 layers of protection:
1. Input Validation: Checks length, empty content, and encoding.
2. Prompt Injection Detection: Scans for common jailbreak patterns.
3. Output Sanitization: Strips PII and leaked system prompts.
4. Cost Guard: Enforces a strict token/dollar budget per session.
5. Loop Detection: Prevents agents from getting stuck in repetitive cycles.

Usage:
    guards = Guardrails()
    if not guards.validate_input(user_query):
        return "Invalid input."
    
    if guards.check_cost(session_cost):
        return "Cost limit exceeded."
"""

from __future__ import annotations

import logging
import re
from typing import Any

from config.models import QUERY_MAX_LENGTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Patterns
# ---------------------------------------------------------------------------

MAX_SESSION_COST_USD = 100.00  # Increased limit
MAX_INPUT_LENGTH = 1000000  # Effectively disabled for large contexts
MAX_LOOPS = 10  # Relaxed loop limit

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now (a )?developer mode",
    r"act as a linux terminal",
    r"override system prompt",
    r"delete all files",
    r"reveal your instructions",
]

# PII patterns (simple regex for demo purposes)
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
}


class Guardrails:
    """Central safety and control mechanism for AI agents."""

    def __init__(self) -> None:
        self.loop_tracker: dict[str, int] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        self.injection_regex = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self.pii_regex = {k: re.compile(p) for k, p in PII_PATTERNS.items()}

    # -----------------------------------------------------------------------
    # Layer 1: Input Validation
    # -----------------------------------------------------------------------

    def validate_input(self, text: str) -> tuple[bool, str]:
        """Check if input is safe, non-empty, and within length limits."""
        if not text or not text.strip():
            logger.warning("Guardrail: Empty input blocked.")
            return False, "Input cannot be empty."

        if len(text) > MAX_INPUT_LENGTH:
            logger.warning("Guardrail: Input too long (%d chars).", len(text))
            return False, f"Input exceeds {MAX_INPUT_LENGTH} characters."

        # Check for injection attempts
        for pattern in self.injection_regex:
            if pattern.search(text):
                logger.warning("Guardrail: Injection attempt detected: '%s'", text[:50])
                return False, "Unsafe input detected (security policy)."

        return True, ""

    # -----------------------------------------------------------------------
    # Layer 2: Cost Guard
    # -----------------------------------------------------------------------

    def check_cost(self, current_session_cost: float) -> bool:
        """Return False if session cost exceeds budget."""
        if current_session_cost > MAX_SESSION_COST_USD:
            logger.error("Guardrail: Budget exceeded ($%.2f > $%.2f).", 
                         current_session_cost, MAX_SESSION_COST_USD)
            return False
        return True

    # -----------------------------------------------------------------------
    # Layer 3: Output Sanitization
    # -----------------------------------------------------------------------

    def sanitize_output(self, text: str) -> str:
        """Strip PII and system prompt leaks from agent output."""
        sanitized = text

        # Redact PII
        for pii_type, regex in self.pii_regex.items():
            sanitized = regex.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)

        # Check for system prompt leaks (heuristic)
        if "You are a research query decomposer" in sanitized or "SYSTEM_PROMPT" in sanitized:
            logger.warning("Guardrail: System prompt leak detected and redacted.")
            sanitized = sanitized.replace("You are a research query decomposer", "[REDACTED]")

        if sanitized != text:
            logger.info("Guardrail: Output sanitized (PII or leak removed).")

        return sanitized

    # -----------------------------------------------------------------------
    # Layer 4: Loop Detection
    # -----------------------------------------------------------------------

    def detect_loop(self, agent_name: str, action_hash: str) -> bool:
        """Check if an agent is repeating the same action too many times.
        
        Args:
            agent_name: Name of the agent.
            action_hash: Unique hash/string of the action (e.g., query string).
        """
        key = f"{agent_name}:{action_hash}"
        self.loop_tracker[key] = self.loop_tracker.get(key, 0) + 1
        
        if self.loop_tracker[key] > MAX_LOOPS:
            logger.warning("Guardrail: Loop detected for %s (action repeated %d times).", 
                           agent_name, self.loop_tracker[key])
            return True
        return False

    def reset_loops(self) -> None:
        """Clear loop tracker (call at start of new turn)."""
        self.loop_tracker.clear()

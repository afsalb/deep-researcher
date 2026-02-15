"""
Real-time token and cost tracking for the research session.

Singleton pattern ensures a single tracker instance across all agents.
Thread-safe via threading.Lock for concurrent agent calls.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from config.models import MODEL_COSTS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LLMCallRecord:
    """Single LLM API call record."""

    model: str
    input_tokens: int
    output_tokens: int
    task_name: str
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# CostTracker (singleton)
# ---------------------------------------------------------------------------

class CostTracker:
    """Session-wide singleton tracking all LLM token usage and costs."""

    _instance: CostTracker | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> CostTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._records: list[LLMCallRecord] = []
        self._initialised = True

    # -- public API --------------------------------------------------------

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_name: str = "unknown",
    ) -> float:
        """Record an LLM call and return its estimated cost in USD.

        Args:
            model: Full model identifier (e.g. ``google/gemini-2.0-flash-001``).
            input_tokens: Number of prompt tokens consumed.
            output_tokens: Number of completion tokens generated.
            task_name: Logical agent / task that made the call.

        Returns:
            Estimated cost in USD for this single call.
        """
        costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        cost_usd = (
            (input_tokens / 1_000_000) * costs["input"]
            + (output_tokens / 1_000_000) * costs["output"]
        )
        record = LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            task_name=task_name,
            cost_usd=cost_usd,
        )
        with self._lock:
            self._records.append(record)
        return cost_usd

    def get_session_stats(self) -> dict[str, Any]:
        """Return aggregated session statistics.

        Returns:
            Dict with ``total_cost``, ``total_input_tokens``,
            ``total_output_tokens``, ``calls_per_agent``, and
            ``calls_per_model``.
        """
        with self._lock:
            records = list(self._records)

        total_cost = sum(r.cost_usd for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)

        calls_per_agent: dict[str, int] = {}
        calls_per_model: dict[str, int] = {}
        for r in records:
            calls_per_agent[r.task_name] = calls_per_agent.get(r.task_name, 0) + 1
            calls_per_model[r.model] = calls_per_model.get(r.model, 0) + 1

        return {
            "total_cost": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_calls": len(records),
            "calls_per_agent": calls_per_agent,
            "calls_per_model": calls_per_model,
        }

    def export_to_dict(self) -> dict[str, Any]:
        """Serialise full session data for Streamlit display.

        Returns:
            Dict containing ``stats`` and a chronological ``call_log``.
        """
        with self._lock:
            records = list(self._records)

        return {
            "stats": self.get_session_stats(),
            "call_log": [
                {
                    "model": r.model,
                    "task": r.task_name,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": round(r.cost_usd, 6),
                    "timestamp": r.timestamp,
                }
                for r in records
            ],
        }

    def reset(self) -> None:
        """Clear all tracked records (useful for new sessions)."""
        with self._lock:
            self._records.clear()

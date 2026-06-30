"""Deterministic mock LLM (default, CI-safe).

Stands in for a real model: same interface, deterministic output, no network.
Its "judgment" is a lightweight keyword heuristic — the authoritative detection
lives in the rule layer (packs/ugc/classify.py, moderate.py); the mock is the
second opinion so the fusion logic is exercised without a model.
"""

from __future__ import annotations

from recolens.providers.base import CategoryVerdict, LLMProvider, SafetyVerdict

_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard above",
    "system prompt",
    "you are now",
    "forget your instructions",
)


class DeterministicMockLLM(LLMProvider):
    name = "mock"

    def classify_category(self, text: str, labels: list[str]) -> CategoryVerdict:
        low = text.lower()
        scores = {label: low.count(label.lower()) for label in labels}
        best = max(scores, key=lambda k: (scores[k], k)) if labels else "other"
        total = sum(scores.values())
        conf = (scores.get(best, 0) / total) if total else 0.0
        return CategoryVerdict(label=best if conf > 0 else "other", confidence=conf)

    def assess_safety(self, text: str) -> SafetyVerdict:
        low = text.lower()
        hits = [m for m in _INJECTION_MARKERS if m in low]
        if hits:
            return SafetyVerdict(True, ("prompt_injection",), f"marker: {hits[0]}")
        return SafetyVerdict(False)

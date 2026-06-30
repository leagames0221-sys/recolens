"""LLM provider abstraction (C-3).

classify / moderate fuse a deterministic rule layer with an LLM layer. The LLM
layer goes through this ABC so the default (mock, deterministic) keeps CI offline
and reproducible, while Ollama is an env-swap (``LLM_PROVIDER=ollama``). If no
provider is available, callers run the rule layer alone (R-LLM-3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SafetyVerdict:
    malicious: bool
    categories: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class CategoryVerdict:
    label: str
    confidence: float = 0.0
    fields: dict = field(default_factory=dict)


class LLMProvider(ABC):
    name: str = "base"

    def available(self) -> bool:
        return True

    @abstractmethod
    def classify_category(self, text: str, labels: list[str]) -> CategoryVerdict: ...

    @abstractmethod
    def assess_safety(self, text: str) -> SafetyVerdict: ...

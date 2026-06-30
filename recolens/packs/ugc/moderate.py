"""Abuse / spam / prompt-injection filtering (R-LLM-2, R-LLM-4).

A deterministic rule layer is authoritative and always runs; an optional LLM
provider adds a second opinion. Maps to OWASP LLM Top-10 LLM01 (Prompt
Injection): user-generated content is treated as data, and injection-shaped
strings are flagged so they can never reach a downstream model as instructions.

Negative cases (benign content must NOT be flagged) are mandatory in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# LLM01 prompt-injection shaped patterns.
INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?(the\s+)?previous\s+(instructions|prompts?)",
    r"disregard\s+(the\s+)?(above|previous)",
    r"forget\s+(your\s+)?(previous\s+)?instructions",
    r"you\s+are\s+now\s+",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"system\s+prompt\s*[:=]",
    r"<\|im_(start|end)\|>",
    r"###\s*instruction",
    r"act\s+as\s+(an?\s+)?(unrestricted|jailbroken|developer\s+mode)",
)
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass(frozen=True)
class ModerationResult:
    action: str  # "allow" | "flag" | "block"
    malicious: bool
    injection: bool
    spam: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    source: str = "rule"


def _detect_injection(text: str) -> list[str]:
    return [rx.pattern for rx in _INJECTION_RE if rx.search(text)]


def _detect_spam(text: str) -> list[str]:
    reasons: list[str] = []
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        reasons.append("excessive_caps")
    if re.search(r"(.)\1{6,}", text):
        reasons.append("char_repetition")
    if text.lower().count("http") >= 4:
        reasons.append("link_spam")
    return reasons


def moderate(text: str, provider=None) -> ModerationResult:
    inj = _detect_injection(text)
    spam = _detect_spam(text)
    reasons = [f"injection:{p}" for p in inj] + [f"spam:{r}" for r in spam]
    source = "rule"

    if provider is not None and getattr(provider, "available", lambda: True)():
        v = provider.assess_safety(text)
        if v.malicious:
            reasons.append(f"llm:{v.reason or 'flagged'}")
            source = "rule+llm"

    injection = bool(inj)
    spammy = bool(spam)
    llm_flag = source == "rule+llm"
    malicious = injection or spammy or llm_flag
    if injection:
        action = "block"  # injection-shaped content never passes through
    elif spammy or llm_flag:
        action = "flag"
    else:
        action = "allow"
    return ModerationResult(
        action=action,
        malicious=malicious,
        injection=injection,
        spam=spammy,
        reasons=tuple(reasons),
        source=source,
    )

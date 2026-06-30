"""Content classification + article-quality scoring (R-LLM-1).

Rule layer (deterministic, keyword lexicon + heuristics) is authoritative and
always runs. An optional LLM provider gives a second opinion that can override
the category when confident. Works LLM-free (R-LLM-3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CATEGORY_LEXICON: dict[str, tuple[str, ...]] = {
    "fantasy": ("dragon", "wizard", "kingdom", "magic", "sword", "elf", "quest"),
    "romance": ("love", "heart", "kiss", "wedding", "longing", "promise"),
    "scifi": ("spaceship", "android", "galaxy", "quantum", "cyborg", "reactor", "orbit"),
    "mystery": ("detective", "clue", "murder", "alibi", "suspect", "case"),
    "essay": ("argument", "society", "reflection", "culture", "ethics", "theory"),
    "howto": ("step", "guide", "install", "configure", "tutorial", "setup"),
    "horror": ("ghost", "blood", "haunted", "scream", "curse", "monster"),
    "comedy": ("joke", "laugh", "prank", "witty", "silly", "punchline"),
    "history": ("empire", "ancient", "war", "dynasty", "revolution", "century"),
    "poetry": ("verse", "rhythm", "metaphor", "stanza", "moonlight", "echo"),
}
LABELS = list(CATEGORY_LEXICON.keys())
_WORD = re.compile(r"[0-9A-Za-z']+")


@dataclass(frozen=True)
class ClassifyResult:
    category: str
    confidence: float
    quality: float
    source: str  # "rule" | "llm"


def rule_category(text: str) -> tuple[str, float]:
    low = text.lower()
    scores = {c: sum(low.count(w) for w in words) for c, words in CATEGORY_LEXICON.items()}
    total = sum(scores.values())
    if total == 0:
        return "other", 0.0
    best = max(scores, key=lambda k: (scores[k], k))
    return best, scores[best] / total


def quality_score(text: str) -> float:
    """Heuristic article quality in [0,1]: length adequacy x lexical diversity,
    penalized for character-run spam."""
    words = _WORD.findall(text.lower())
    n = len(words)
    if n == 0:
        return 0.0
    length_ok = min(1.0, n / 20.0)  # saturates around 20+ words
    diversity = len(set(words)) / n
    repetition_penalty = 1.0 if not re.search(r"(.)\1{4,}", text) else 0.5
    return round(length_ok * diversity * repetition_penalty, 4)


def classify(text: str, provider=None) -> ClassifyResult:
    label, conf = rule_category(text)
    source = "rule"
    if provider is not None and getattr(provider, "available", lambda: True)():
        v = provider.classify_category(text, LABELS)
        if v.confidence > conf and v.label in CATEGORY_LEXICON:
            label, conf, source = v.label, v.confidence, "llm"
    return ClassifyResult(
        category=label, confidence=round(conf, 4), quality=quality_score(text), source=source
    )

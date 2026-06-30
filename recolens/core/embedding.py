"""Embedding providers.

``DeterministicHashEmbed`` is a zero-dependency hashing-trick embedder so the
default pipeline runs offline and reproducibly (R-SEARCH-2 fallback, R-CORE-4).
The real local embedder (sentence-transformers e5/BGE) is added in P3 behind
the ``[embed]`` extra, implementing the same ABC (C-3).
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

# Tokenizes ASCII words and CJK (hiragana / katakana / kanji) runs.
_TOKEN = re.compile(r"[0-9A-Za-z]+|[぀-ヿ一-鿿]")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed query-side text. Default: same as embed(); models that need a
        distinct query encoding (e.g. e5's 'query:' prefix) override this."""
        return self.embed(texts)


class DeterministicHashEmbed(EmbeddingProvider):
    """Stable hashing-trick embedder (no model, no network)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in _tokens(text):
                # md5 for a stable, process-independent hash (Python's hash() is salted).
                bucket = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
                vec[bucket] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

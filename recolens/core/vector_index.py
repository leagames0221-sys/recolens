"""Vector index abstraction + in-memory fallback.

``InMemoryFlatIndex`` is an exact brute-force cosine index with no dependencies,
used by default and as the automatic fallback when Qdrant is unavailable
(R-SEARCH-3). The Qdrant-backed index (P3, ``[vector]`` extra) implements the
same ABC (C-3).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence


class VectorIndex(ABC):
    @abstractmethod
    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None: ...

    @abstractmethod
    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def __len__(self) -> int: ...


def _l2_normalize(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class InMemoryFlatIndex(VectorIndex):
    """Exact cosine top-K over normalized vectors."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        self._ids.extend(ids)
        self._vecs.extend(_l2_normalize(v) for v in vectors)

    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]:
        q = _l2_normalize(query)
        scored = [
            (id_, sum(a * b for a, b in zip(q, v, strict=True)))
            for id_, v in zip(self._ids, self._vecs, strict=True)
        ]
        scored.sort(key=lambda t: (-t[1], t[0]))  # deterministic tie-break by id
        return scored[:k]

    def __len__(self) -> int:
        return len(self._ids)

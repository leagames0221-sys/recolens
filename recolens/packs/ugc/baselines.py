"""Baseline rankers for the eval harness (P2).

These give the eval harness real systems to compare before the richer P4
recommenders land. Both implement the ``Ranker`` protocol (core/evaluate.py).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from recolens.core.embedding import DeterministicHashEmbed
from recolens.core.schema import Interaction, Item, item_text
from recolens.core.vector_index import InMemoryFlatIndex


class PopularityRanker:
    """Rank every item by training popularity, excluding items the user saw."""

    name = "popularity"

    def __init__(self) -> None:
        self._ranked: list[str] = []
        self._seen: dict[str, set[str]] = defaultdict(set)

    def fit(self, items: Sequence[Item], train: Sequence[Interaction]) -> None:
        counts = Counter(t.item_id for t in train)
        all_ids = [it.item_id for it in items]
        # popular first; deterministic tie-break by id; zero-count items appended sorted
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self._ranked = [i for i, _ in ranked] + sorted(i for i in all_ids if i not in counts)
        self._seen = defaultdict(set)
        for t in train:
            self._seen[t.user_id].add(t.item_id)

    def rank(self, user_id: str, k: int) -> list[str]:
        seen = self._seen.get(user_id, set())
        return [i for i in self._ranked if i not in seen][:k]


class ContentRanker:
    """Embedding kNN: score candidates by similarity to the user's history mean."""

    name = "content"

    def __init__(self, dim: int = 64, embedder=None) -> None:
        # Default = zero-dep deterministic embedder; inject a real one (e.g.
        # LocalSentenceTransformerEmbed) to evaluate real-embedding quality.
        self._embedder = embedder if embedder is not None else DeterministicHashEmbed(dim=dim)
        self._vec: dict[str, list[float]] = {}
        self._index = InMemoryFlatIndex()
        self._seen: dict[str, set[str]] = defaultdict(set)

    def fit(self, items: Sequence[Item], train: Sequence[Interaction]) -> None:
        vecs = self._embedder.embed([item_text(it) for it in items])
        self._vec = {it.item_id: v for it, v in zip(items, vecs, strict=True)}
        self._index = InMemoryFlatIndex()
        self._index.add(list(self._vec.keys()), list(self._vec.values()))
        self._seen = defaultdict(set)
        for t in train:
            self._seen[t.user_id].add(t.item_id)

    def _profile(self, items: set[str]) -> list[float]:
        vecs = [self._vec[i] for i in items if i in self._vec]
        dim = self._embedder.dim
        n = len(vecs)
        return [sum(v[j] for v in vecs) / n for j in range(dim)]

    def rank(self, user_id: str, k: int) -> list[str]:
        seen = self._seen.get(user_id, set())
        if not seen:
            return []  # content alone cannot serve cold-start; hybrid (P4) handles it
        profile = self._profile(seen)
        hits = self._index.search(profile, k + len(seen))
        return [i for i, _ in hits if i not in seen][:k]

"""Item-item collaborative filtering (T-29).

Classic co-occurrence CF: items co-interacted by the same users are related.
A user is scored on candidates by summing co-occurrence with their history.
Falls back to popularity ordering when co-occurrence is sparse.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from recolens.core.schema import Interaction, Item


class CollaborativeRanker:
    name = "collab"

    def __init__(self) -> None:
        self._co: dict[str, Counter[str]] = defaultdict(Counter)
        self._seen: dict[str, set[str]] = defaultdict(set)
        self._pop: list[str] = []

    def fit(self, items: Sequence[Item], train: Sequence[Interaction]) -> None:
        user_items: dict[str, set[str]] = defaultdict(set)
        for t in train:
            user_items[t.user_id].add(t.item_id)

        co: dict[str, Counter[str]] = defaultdict(Counter)
        for its in user_items.values():
            seq = sorted(its)
            for a in seq:
                for b in seq:
                    if a != b:
                        co[a][b] += 1
        self._co = co
        self._seen = {u: set(s) for u, s in user_items.items()}

        counts = Counter(t.item_id for t in train)
        self._pop = [i for i, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        self._pop += sorted(it.item_id for it in items if it.item_id not in counts)

    def genuine_rank(self, user_id: str, k: int) -> list[str]:
        """Top-k by *real* co-occurrence only (no popularity padding).

        Used by the hybrid so collaborative's confident, sparse predictions are
        not diluted by popularity fill during fusion.
        """
        seen = self._seen.get(user_id, set())
        scores: Counter[str] = Counter()
        for h in seen:
            for cand, c in self._co.get(h, {}).items():
                if cand not in seen:
                    scores[cand] += c
        ranked = [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        return ranked[:k]

    def rank(self, user_id: str, k: int) -> list[str]:
        seen = self._seen.get(user_id, set())
        ranked = self.genuine_rank(user_id, k)
        if len(ranked) < k:
            taken = set(ranked) | seen
            ranked += [i for i in self._pop if i not in taken]
        return ranked[:k]

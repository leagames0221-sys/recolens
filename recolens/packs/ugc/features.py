"""Item feature extraction + a metadata (tag) signal (T-27).

Separates structured features (tags / length / author / recency) from the raw
text used by the embedding ranker. The tag-Jaccard signal is a cheap,
explainable content signal that the hybrid ranker fuses alongside the embedding
and collaborative signals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from recolens.core.schema import Item


@dataclass(frozen=True)
class ItemFeatures:
    item_id: str
    tags: frozenset[str]
    body_len: int
    author_id: str
    recency_rank: int  # 0 = newest


def extract_features(items: Sequence[Item]) -> dict[str, ItemFeatures]:
    by_recency = sorted(items, key=lambda it: (-it.created_ts, it.item_id))
    rank = {it.item_id: i for i, it in enumerate(by_recency)}
    return {
        it.item_id: ItemFeatures(
            item_id=it.item_id,
            tags=frozenset(it.tags),
            body_len=len(it.body),
            author_id=it.author_id,
            recency_rank=rank[it.item_id],
        )
        for it in items
    }


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def tag_jaccard_ranking(
    user_tags: frozenset[str],
    features: dict[str, ItemFeatures],
    exclude: set[str],
    k: int,
) -> list[str]:
    """Rank candidate items by tag-set Jaccard similarity to the user profile."""
    if not user_tags:
        return []
    scored = [
        (iid, _jaccard(user_tags, f.tags))
        for iid, f in features.items()
        if iid not in exclude
    ]
    scored = [s for s in scored if s[1] > 0]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [iid for iid, _ in scored[:k]]

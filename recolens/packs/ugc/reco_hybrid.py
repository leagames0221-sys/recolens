"""Hybrid recommender via Reciprocal Rank Fusion (T-30).

Fuses three signals — embedding content (kNN), collaborative (item-item), and
tag-Jaccard metadata — with RRF (Cormack et al. 2009: score = Σ 1/(c+rank),
c=60). Cold-start users (no history) fall back to popularity (R-RECO-2), so the
recommender never errors on a new user.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from recolens.core.schema import Interaction, Item
from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
from recolens.packs.ugc.features import extract_features, tag_jaccard_ranking
from recolens.packs.ugc.reco_collab import CollaborativeRanker

RRF_C = 60
# Weighted fusion (ADR-0009). The collaborative co-read signal is sharper than
# diffuse content similarity, so equal-weight fusion dilutes it; we up-weight
# collaborative and fuse only its genuine co-occurrence hits.
W_CONTENT = 1.0
W_COLLAB = 3.0
W_TAG = 0.5


class HybridRanker:
    name = "hybrid"

    def __init__(self, dim: int = 64, embedder=None) -> None:
        self._content = ContentRanker(dim=dim, embedder=embedder)
        self._collab = CollaborativeRanker()
        self._pop = PopularityRanker()
        self._features: dict = {}
        self._seen: dict[str, set[str]] = defaultdict(set)
        self._user_tags: dict[str, frozenset[str]] = {}

    def fit(self, items: Sequence[Item], train: Sequence[Interaction]) -> None:
        self._content.fit(items, train)
        self._collab.fit(items, train)
        self._pop.fit(items, train)
        self._features = extract_features(items)

        self._seen = defaultdict(set)
        for t in train:
            self._seen[t.user_id].add(t.item_id)
        # user tag profile = union of tags of items in their history
        self._user_tags = {
            u: frozenset(
                tag for iid in seen for tag in self._features[iid].tags if iid in self._features
            )
            for u, seen in self._seen.items()
        }

    def rank(self, user_id: str, k: int) -> list[str]:
        seen = self._seen.get(user_id, set())
        if not seen:
            return self._pop.rank(user_id, k)  # cold-start fallback (R-RECO-2)

        depth = k * 5
        weighted_lists = [
            (W_CONTENT, self._content.rank(user_id, depth)),
            (W_COLLAB, self._collab.genuine_rank(user_id, depth)),
            (
                W_TAG,
                tag_jaccard_ranking(
                    self._user_tags.get(user_id, frozenset()), self._features, seen, depth
                ),
            ),
        ]
        fused: Counter[str] = Counter()
        for weight, lst in weighted_lists:
            for rank, item in enumerate(lst):
                fused[item] += weight / (RRF_C + rank + 1)
        ranked = [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
        return ranked[:k]

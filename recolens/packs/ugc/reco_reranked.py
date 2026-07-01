"""Two-stage learned-reranker recommender (P9).

Stage 1 (retrieval): the existing signals (content kNN, item-item collaborative,
tag-Jaccard, popularity) each propose candidates.
Stage 2 (ranking): a learned model scores every (user, candidate) from the
signals' rank-reciprocal features and orders by predicted relevance.

Training uses a *within-train* time split so the reranker never sees the eval
test slice: signals are fit on the earlier part, labels come from the later
part, then the signals are refit on the full train set for serving. This is the
standard offline learning-to-rank setup — no test leakage. See ADR-0010.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from recolens.core.evaluate import time_split
from recolens.core.rank import LogisticReranker
from recolens.core.schema import Interaction, Item
from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
from recolens.packs.ugc.features import extract_features, tag_jaccard_ranking
from recolens.packs.ugc.reco_collab import CollaborativeRanker

RRF_C = 60
DEPTH = 50  # candidate depth per signal (stage-1 retrieval breadth)
LABEL_RATIO = 0.3  # within-train slice used as LTR labels


def _user_tags(seen: dict[str, set[str]], features: dict) -> dict[str, frozenset[str]]:
    return {
        u: frozenset(tag for iid in s for tag in features[iid].tags if iid in features)
        for u, s in seen.items()
    }


def _candidate_features(content, collab, pop, features, user_id, seen, user_tags, depth):
    """item -> [f_content, f_collab, f_tag, f_popularity] (rank-reciprocal, seen excluded)."""
    lists = (
        content.rank(user_id, depth),
        collab.genuine_rank(user_id, depth),
        tag_jaccard_ranking(user_tags, features, seen, depth),
        pop.rank(user_id, depth),
    )
    feats: dict[str, list[float]] = {}
    for idx, lst in enumerate(lists):
        for rank, item in enumerate(lst):
            if item in seen:
                continue
            feats.setdefault(item, [0.0, 0.0, 0.0, 0.0])[idx] = 1.0 / (RRF_C + rank + 1)
    return feats


class RerankedRanker:
    """Learned two-stage reranker. Default model = dependency-free logistic LTR;
    inject a LightGBM LambdaMART (``[rank]`` extra) for the production-grade variant."""

    name = "reranked"

    def __init__(self, dim: int = 64, embedder=None, model=None) -> None:
        self._dim = dim
        self._embedder = embedder
        self._model = model if model is not None else LogisticReranker()
        self._content = ContentRanker(dim=dim, embedder=embedder)
        self._collab = CollaborativeRanker()
        self._pop = PopularityRanker()
        self._features: dict = {}
        self._seen: dict[str, set[str]] = defaultdict(set)
        self._user_tags: dict[str, frozenset[str]] = {}

    def _fit_signals(self, items, interactions):
        content = ContentRanker(dim=self._dim, embedder=self._embedder)
        collab = CollaborativeRanker()
        pop = PopularityRanker()
        content.fit(items, interactions)
        collab.fit(items, interactions)
        pop.fit(items, interactions)
        return content, collab, pop

    def _build_training(self, items, fit_part, label_part):
        content, collab, pop = self._fit_signals(items, fit_part)
        features = extract_features(items)
        seen: dict[str, set[str]] = defaultdict(set)
        for t in fit_part:
            seen[t.user_id].add(t.item_id)
        tags = _user_tags(seen, features)
        pos: dict[str, set[str]] = defaultdict(set)
        for t in label_part:
            pos[t.user_id].add(t.item_id)

        X: list[list[float]] = []
        y: list[int] = []
        groups: list[int] = []
        for u, s in seen.items():
            cand = _candidate_features(
                content, collab, pop, features, u, s, tags.get(u, frozenset()), DEPTH
            )
            if not cand:
                continue
            g = 0
            for item, fv in cand.items():
                X.append(fv)
                y.append(1 if item in pos.get(u, ()) else 0)
                g += 1
            groups.append(g)
        return X, y, groups

    def fit(self, items: Sequence[Item], train: Sequence[Interaction]) -> RerankedRanker:
        # 1) within-train time split (no eval-test leakage) -> LTR training data
        fit_part, label_part = time_split(list(train), test_ratio=LABEL_RATIO)
        if not label_part or not fit_part:
            fit_part, label_part = list(train), list(train)
        X, y, groups = self._build_training(items, fit_part, label_part)
        if X and sum(y) == 0:
            # degenerate: no positive landed in a candidate pool -> label on full train
            X, y, groups = self._build_training(items, list(train), list(train))
        if X and sum(y) > 0:
            self._model.fit(X, y, groups)
        # 2) refit signals on FULL train for serving
        self._content, self._collab, self._pop = self._fit_signals(items, train)
        self._features = extract_features(items)
        self._seen = defaultdict(set)
        for t in train:
            self._seen[t.user_id].add(t.item_id)
        self._user_tags = _user_tags(self._seen, self._features)
        return self

    def rank(self, user_id: str, k: int) -> list[str]:
        seen = self._seen.get(user_id, set())
        if not seen:
            return self._pop.rank(user_id, k)  # cold-start fallback (R-RECO-2)
        depth = max(k * 5, DEPTH)
        cand = _candidate_features(
            self._content, self._collab, self._pop, self._features,
            user_id, seen, self._user_tags.get(user_id, frozenset()), depth,
        )
        if not cand:
            return self._pop.rank(user_id, k)
        scored = sorted(
            ((self._model.predict(fv), item) for item, fv in cand.items()),
            key=lambda t: (-t[0], t[1]),
        )
        return [item for _, item in scored][:k]

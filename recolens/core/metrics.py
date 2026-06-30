"""Ranking & recsys metrics (zero-dep).

Definitions follow the primary sources recorded in
docs/evidence/metrics_definitions.md (ir_measures / trec_eval for IR metrics,
RecBole for coverage / novelty). The IR metrics are cross-checked against
``ir_measures`` in tests (independent two-implementation agreement), so this
module does not invent its own definitions.

Conventions:
- A ranking is an ordered list of item ids (best first).
- ``relevance`` is a mapping item_id -> graded relevance (int). An item is
  "relevant" iff relevance >= ``rel`` (default 1). Absent => 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _relevant_set(relevance: Mapping[str, float], rel: int) -> set[str]:
    return {k for k, v in relevance.items() if v >= rel}


def recall_at_k(
    ranking: Sequence[str], relevance: Mapping[str, float], k: int, rel: int = 1
) -> float:
    rels = _relevant_set(relevance, rel)
    if not rels:
        return 0.0
    hit = sum(1 for d in ranking[:k] if d in rels)
    return hit / len(rels)


def precision_at_k(
    ranking: Sequence[str], relevance: Mapping[str, float], k: int, rel: int = 1
) -> float:
    if k <= 0:
        return 0.0
    rels = _relevant_set(relevance, rel)
    hit = sum(1 for d in ranking[:k] if d in rels)
    return hit / k


def average_precision(
    ranking: Sequence[str], relevance: Mapping[str, float], rel: int = 1
) -> float:
    rels = _relevant_set(relevance, rel)
    if not rels:
        return 0.0
    hits = 0
    acc = 0.0
    for i, d in enumerate(ranking, start=1):
        if d in rels:
            hits += 1
            acc += hits / i
    return acc / len(rels)


def reciprocal_rank(ranking: Sequence[str], relevance: Mapping[str, float], rel: int = 1) -> float:
    rels = _relevant_set(relevance, rel)
    for i, d in enumerate(ranking, start=1):
        if d in rels:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    # DCG@k = sum_{i=1..k} rel(i) / log2(i+1)   (linear gains, dcg='log2')
    return sum(
        float(relevance.get(d, 0.0)) / math.log2(i + 1) for i, d in enumerate(ranking[:k], start=1)
    )


def ndcg_at_k(ranking: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    dcg = dcg_at_k(ranking, relevance, k)
    ideal_order = sorted(relevance.values(), reverse=True)
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_order[:k], start=1) if g > 0)
    return dcg / idcg if idcg > 0 else 0.0


# --- recsys-level metrics over a set of recommendation lists ---


def item_coverage(rec_lists: Sequence[Sequence[str]], n_items: int) -> float:
    """|union of recommended items| / |I|  (RecBole ItemCoverage)."""
    if n_items <= 0:
        return 0.0
    seen: set[str] = set()
    for lst in rec_lists:
        seen.update(lst)
    return len(seen) / n_items


def novelty(
    rec_lists: Sequence[Sequence[str]],
    item_user_count: Mapping[str, int],
    n_users: int,
) -> float:
    """Mean self-information log2(M / d_j) over recommended items (RecBole Novelty).

    Items with no recorded interactions are treated as maximally novel (d_j=1).
    """
    if n_users <= 0:
        return 0.0
    total = 0.0
    count = 0
    for lst in rec_lists:
        for item in lst:
            d_j = max(1, item_user_count.get(item, 0))
            total += math.log2(n_users / d_j)
            count += 1
    return total / count if count else 0.0


# --- aggregation over multiple queries (mean), used by `recolens eval` ---


# Per-query IR metric callables (ranking, relevance, k) -> float.
def aggregate_ir(
    qrels: Mapping[str, Mapping[str, float]],
    runs: Mapping[str, Sequence[str]],
    ks: Sequence[int] = (5, 10),
    rel: int = 1,
) -> dict[str, float]:
    """Mean over queries of the standard IR metrics. Keys like 'nDCG@10'."""
    out: dict[str, list[float]] = {}

    def push(name: str, value: float) -> None:
        out.setdefault(name, []).append(value)

    for qid, ranking in runs.items():
        relevance = qrels.get(qid, {})
        push("AP", average_precision(ranking, relevance, rel))
        push("RR", reciprocal_rank(ranking, relevance, rel))
        for k in ks:
            push(f"R@{k}", recall_at_k(ranking, relevance, k, rel))
            push(f"P@{k}", precision_at_k(ranking, relevance, k, rel))
            push(f"nDCG@{k}", ndcg_at_k(ranking, relevance, k))

    return {name: (sum(vs) / len(vs) if vs else 0.0) for name, vs in out.items()}

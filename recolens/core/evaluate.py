"""Offline evaluation harness: temporal split + ranking evaluation (R-EVAL-1).

A ``Ranker`` is any object that can ``fit(items, train_interactions)`` and
``rank(user_id, k)``. This harness is system-agnostic: the P4 recommenders
(content / collaborative / hybrid) plug into the same protocol, so eval compares
them on identical ground truth.

Split policy: per-user temporal hold-out — each user's latest ``test_ratio`` of
interactions (by ts) are the held-out ground truth; the rest is history.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from recolens.core.metrics import aggregate_ir
from recolens.core.schema import Interaction, Item


@runtime_checkable
class Ranker(Protocol):
    name: str

    def fit(self, items: Sequence[Item], train: Sequence[Interaction]) -> None: ...

    def rank(self, user_id: str, k: int) -> list[str]: ...


def time_split(
    interactions: Sequence[Interaction], test_ratio: float = 0.3
) -> tuple[list[Interaction], list[Interaction]]:
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for it in interactions:
        by_user[it.user_id].append(it)

    train: list[Interaction] = []
    test: list[Interaction] = []
    for _uid, lst in by_user.items():
        ordered = sorted(lst, key=lambda x: (x.ts, x.item_id))
        if len(ordered) < 2:
            train.extend(ordered)  # not enough history to hold out
            continue
        n_test = max(1, round(len(ordered) * test_ratio))
        train.extend(ordered[:-n_test])
        test.extend(ordered[-n_test:])
    return train, test


def build_qrels(test: Sequence[Interaction]) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for it in test:
        qrels[it.user_id][it.item_id] = 1.0
    return qrels


def run_eval(
    items: Sequence[Item],
    train: Sequence[Interaction],
    test: Sequence[Interaction],
    rankers: Sequence[Ranker],
    ks: Sequence[int] = (5, 10),
) -> tuple[
    Mapping[str, Mapping[str, float]], dict[str, dict[str, float]], dict[str, dict[str, list[str]]]
]:
    """Return (qrels, {ranker_name: metrics}, {ranker_name: {user: ranking}})."""
    qrels = build_qrels(test)
    users = sorted(qrels.keys())
    results: dict[str, dict[str, float]] = {}
    runs_by_ranker: dict[str, dict[str, list[str]]] = {}

    for ranker in rankers:
        ranker.fit(items, train)
        runs = {u: ranker.rank(u, max(ks)) for u in users}
        results[ranker.name] = aggregate_ir(qrels, runs, ks)
        runs_by_ranker[ranker.name] = runs

    return qrels, results, runs_by_ranker

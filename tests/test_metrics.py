"""Metric tests.

The IR metrics are cross-checked against ir_measures (trec_eval family) on the
same qrels/run — independent two-implementation agreement, not a confirmatory
test. Edge/negative cases (empty relevant set, no hit) are included.
"""

from __future__ import annotations

import pytest

from recolens.core import metrics

# --- cross-check vs ir_measures -------------------------------------------------


def _calc_ir_measures(qrels_graded, run):
    ir_measures = pytest.importorskip("ir_measures")
    from ir_measures import AP, RR, P, R, nDCG

    measures = [nDCG @ 3, nDCG @ 5, AP, RR, R @ 3, R @ 5, P @ 3, P @ 5]
    agg = ir_measures.calc_aggregate(measures, qrels_graded, run)
    # normalize keys to our naming
    return {str(m): float(v) for m, v in agg.items()}


def test_ir_metrics_match_ir_measures_binary():
    # binary relevance, two queries
    qrels = {
        "q1": {"a": 1, "b": 0, "c": 1, "d": 1},
        "q2": {"x": 1, "y": 1, "z": 0},
    }
    run = {
        "q1": {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5},
        "q2": {"z": 0.9, "x": 0.8, "y": 0.1},
    }
    ref = _calc_ir_measures(qrels, run)

    runs_ranked = {
        q: [d for d, _ in sorted(s.items(), key=lambda t: -t[1])] for q, s in run.items()
    }
    ours = metrics.aggregate_ir(qrels, runs_ranked, ks=(3, 5), rel=1)

    for name in ["AP", "RR", "R@3", "R@5", "P@3", "P@5", "nDCG@3", "nDCG@5"]:
        assert name in ref, f"{name} not produced by ir_measures"
        assert abs(ours[name] - ref[name]) < 1e-9, f"{name}: ours={ours[name]} ir={ref[name]}"


def test_ndcg_matches_ir_measures_graded():
    # graded relevance exercises the log2 DCG / IDCG path
    qrels = {"q1": {"a": 3, "b": 1, "c": 2, "d": 0}}
    run = {"q1": {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.4}}  # deliberately suboptimal order
    ref = _calc_ir_measures(qrels, run)
    ranked = [d for d, _ in sorted(run["q1"].items(), key=lambda t: -t[1])]
    for k in (3, 5):
        ours = metrics.ndcg_at_k(ranked, qrels["q1"], k)
        assert abs(ours - ref[f"nDCG@{k}"]) < 1e-9


# --- direct unit checks ---------------------------------------------------------


def test_recall_precision_basic():
    ranking = ["a", "b", "c", "d"]
    relevance = {"a": 1, "c": 1}  # 2 relevant
    assert metrics.recall_at_k(ranking, relevance, 2) == 0.5  # only 'a' in top-2
    assert metrics.precision_at_k(ranking, relevance, 2) == 0.5


def test_reciprocal_rank_first_relevant():
    assert metrics.reciprocal_rank(["x", "a", "b"], {"a": 1}) == pytest.approx(0.5)
    assert metrics.reciprocal_rank(["a"], {"a": 1}) == 1.0


def test_ndcg_perfect_is_one():
    rel = {"a": 3, "b": 2, "c": 1}
    assert metrics.ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)


# --- edge / negative cases ------------------------------------------------------


def test_no_relevant_returns_zero():
    assert metrics.recall_at_k(["a"], {}, 5) == 0.0
    assert metrics.average_precision(["a"], {}) == 0.0
    assert metrics.reciprocal_rank(["a"], {}) == 0.0
    assert metrics.ndcg_at_k(["a"], {}, 5) == 0.0


def test_no_hit_returns_zero():
    assert metrics.reciprocal_rank(["x", "y"], {"a": 1}) == 0.0
    assert metrics.recall_at_k(["x", "y"], {"a": 1}, 2) == 0.0


# --- coverage / novelty ---------------------------------------------------------


def test_item_coverage():
    rec = [["a", "b"], ["b", "c"]]
    assert metrics.item_coverage(rec, n_items=10) == pytest.approx(0.3)  # {a,b,c}/10
    assert metrics.item_coverage([], n_items=0) == 0.0


def test_novelty_popular_lower_than_rare():
    counts = {"pop": 100, "rare": 1}
    n_users = 100
    pop = metrics.novelty([["pop"]], counts, n_users)
    rare = metrics.novelty([["rare"]], counts, n_users)
    assert rare > pop
    assert pop == pytest.approx(0.0)  # log2(100/100)=0

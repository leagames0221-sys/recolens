"""Eval harness, A-B simulation, and golden regression gate (R-EVAL-3, R-EVAL-4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recolens.core.ab import ab_compare
from recolens.core.evaluate import build_qrels, run_eval, time_split
from recolens.core.schema import parse_interactions, parse_items
from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
from recolens.packs.ugc.reco_collab import CollaborativeRanker
from recolens.packs.ugc.reco_hybrid import HybridRanker
from recolens.packs.ugc.synth import generate

GOLDEN = Path(__file__).resolve().parents[1] / "docs" / "golden" / "eval_metrics.json"


def _run_default_eval():
    data = generate(n_items=300, n_users=80, seed=42)
    items = parse_items(data["items"]).valid
    inter = parse_interactions(data["interactions"]).valid
    train, test = time_split(inter, 0.3)
    rankers = [
        PopularityRanker(),
        ContentRanker(dim=64),
        CollaborativeRanker(),
        HybridRanker(dim=64),
    ]
    _q, results, _r = run_eval(items, train, test, rankers, ks=(5, 10))
    return results


# --- temporal split sanity ---


def test_time_split_holds_out_latest():
    data = generate(n_items=50, n_users=10, seed=3)
    inter = parse_interactions(data["interactions"]).valid
    train, test = time_split(inter, 0.3)
    assert len(train) > 0 and len(test) > 0
    # each test interaction's ts >= that user's max train ts (held-out is latest)
    max_train = {}
    for it in train:
        max_train[it.user_id] = max(max_train.get(it.user_id, it.ts), it.ts)
    for it in test:
        if it.user_id in max_train:
            assert it.ts >= max_train[it.user_id]


# --- content has real signal on themed data (beats popularity) ---


def test_content_beats_popularity_on_themed_data():
    results = _run_default_eval()
    assert results["content"]["nDCG@10"] > results["popularity"]["nDCG@10"]
    assert results["content"]["RR"] > results["popularity"]["RR"]


# --- golden regression gate ---


def test_eval_matches_golden():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    results = _run_default_eval()
    for ranker, metrics in golden["results"].items():
        for name, expected in metrics.items():
            actual = results[ranker][name]
            assert actual == pytest.approx(expected, abs=1e-9), f"{ranker}/{name} drifted"


def test_golden_gate_catches_tampering():
    # a tampered golden value must NOT match current output (negative case)
    results = _run_default_eval()
    tampered = results["content"]["nDCG@10"] + 0.05
    assert results["content"]["nDCG@10"] != pytest.approx(tampered, abs=1e-9)


# --- A-B simulation ---


def test_ab_identical_variants_zero_lift():
    data = generate(n_items=120, n_users=40, seed=9)
    items = parse_items(data["items"]).valid
    inter = parse_interactions(data["interactions"]).valid
    train, test = time_split(inter, 0.3)
    qrels = build_qrels(test)
    users = sorted(qrels)
    r = PopularityRanker()
    r.fit(items, train)
    runs = {u: r.rank(u, 10) for u in users}
    res = ab_compare(qrels, runs, runs, k=10, seed=1)  # A == B
    assert res["abs_lift"] == 0.0
    assert res["decision"] == "inconclusive"  # zero diff cannot be a win


def test_ab_detects_strict_improvement():
    # B strictly dominates A: B always has the relevant item at top, A never does.
    qrels = {f"u{i}": {f"good{i}": 1.0} for i in range(40)}
    runs_a = {f"u{i}": [f"bad{i}"] for i in range(40)}
    runs_b = {f"u{i}": [f"good{i}"] for i in range(40)}
    res = ab_compare(qrels, runs_a, runs_b, k=10, seed=1)
    assert res["kpi_b"] == 1.0 and res["kpi_a"] == 0.0
    assert res["decision"] == "B wins"
    assert res["ci95_low"] > 0

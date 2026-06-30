"""Phase 4 recommender tests: collab, hybrid, features, cold-start (R-RECO-2)."""

from __future__ import annotations

from recolens.core.evaluate import run_eval, time_split
from recolens.core.schema import Item, parse_interactions, parse_items
from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
from recolens.packs.ugc.features import extract_features, tag_jaccard_ranking
from recolens.packs.ugc.reco_collab import CollaborativeRanker
from recolens.packs.ugc.reco_hybrid import HybridRanker
from recolens.packs.ugc.synth import generate


def _data(n_items=200, n_users=60, seed=42):
    data = generate(n_items=n_items, n_users=n_users, seed=seed)
    items = parse_items(data["items"]).valid
    inter = parse_interactions(data["interactions"]).valid
    return items, inter


# --- features ---


def test_extract_features_recency_rank():
    items = [
        Item(item_id="old", created_ts=100, tags=("a",)),
        Item(item_id="new", created_ts=200, tags=("a", "b")),
    ]
    f = extract_features(items)
    assert f["new"].recency_rank == 0  # newest first
    assert f["old"].recency_rank == 1
    assert f["new"].tags == frozenset({"a", "b"})


def test_tag_jaccard_ranking_orders_by_overlap_and_excludes():
    items = [
        Item(item_id="x", tags=("fantasy", "romance")),
        Item(item_id="y", tags=("fantasy",)),
        Item(item_id="z", tags=("scifi",)),
    ]
    f = extract_features(items)
    ranked = tag_jaccard_ranking(frozenset({"fantasy", "romance"}), f, exclude={"x"}, k=5)
    assert ranked[0] == "y"  # shares fantasy
    assert "z" not in ranked  # no overlap
    assert "x" not in ranked  # excluded


def test_tag_jaccard_empty_profile_returns_empty():
    items = [Item(item_id="x", tags=("a",))]
    f = extract_features(items)
    assert tag_jaccard_ranking(frozenset(), f, exclude=set(), k=5) == []


# --- collaborative ---


def test_collab_excludes_seen_and_returns_k():
    items, inter = _data()
    train, _test = time_split(inter, 0.3)
    r = CollaborativeRanker()
    r.fit(items, train)
    user = train[0].user_id
    recs = r.rank(user, 10)
    assert len(recs) == 10
    seen = {t.item_id for t in train if t.user_id == user}
    assert not (set(recs) & seen)  # never recommends already-seen items


# --- hybrid + cold-start ---


def test_hybrid_cold_start_falls_back_not_errors():
    items, inter = _data()
    train, _test = time_split(inter, 0.3)
    r = HybridRanker(dim=64)
    r.fit(items, train)
    # a user with no history must not error and should still get recommendations
    recs = r.rank("brand-new-user", 5)
    assert len(recs) == 5  # popularity fallback


def _ndcg_by_method(n_items, n_users, seed):
    items, inter = _data(n_items=n_items, n_users=n_users, seed=seed)
    train, test = time_split(inter, 0.3)
    rankers = [
        PopularityRanker(),
        ContentRanker(dim=64),
        CollaborativeRanker(),
        HybridRanker(dim=64),
    ]
    _q, results, _r = run_eval(items, train, test, rankers, ks=(5, 10))
    return {m: results[m]["nDCG@10"] for m in ("popularity", "content", "collab", "hybrid")}


def test_method_ranking_is_stable_and_honest():
    """The harness yields a STABLE ordering across seeds (no cherry-picking):
    collab > hybrid > content > popularity.

    Note the deliberately-reported negative result: the hybrid does NOT beat
    collaborative on this co-read-heavy workload — the co-read signal is sharp
    enough that fusing in lower-precision content can only dilute it. The hybrid's
    value is robustness (it always beats content and popularity), not peak score.
    """
    for seed in (42, 7, 123, 2026):
        r = _ndcg_by_method(300, 80, seed)
        # learned methods clear the popularity baseline by a wide margin
        assert r["collab"] > r["popularity"], (seed, r)
        assert r["content"] > r["popularity"], (seed, r)
        # collaborative is the strongest single method here
        assert r["collab"] > r["content"], (seed, r)
        # hybrid is a robust all-rounder: beats content & popularity...
        assert r["hybrid"] > r["content"], (seed, r)
        assert r["hybrid"] > r["popularity"], (seed, r)
        # ...but does NOT beat the sharp collaborative signal (honest finding)
        assert r["hybrid"] < r["collab"], (seed, r)


def test_hybrid_does_not_collapse_to_worst():
    """Regression guard: fusion must never be the worst method (that would mean
    a broken blend, not a trade-off)."""
    for seed in (42, 7, 123):
        r = _ndcg_by_method(300, 80, seed)
        assert r["hybrid"] > min(r.values()), (seed, r)


def test_recommenders_are_deterministic():
    items, inter = _data()
    train, _test = time_split(inter, 0.3)
    a, b = HybridRanker(dim=64), HybridRanker(dim=64)
    a.fit(items, train)
    b.fit(items, train)
    assert a.rank(train[0].user_id, 10) == b.rank(train[0].user_id, 10)

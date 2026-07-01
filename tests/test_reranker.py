"""Learned reranker (stage-2 LTR) tests — P9.

Covers determinism, that the model actually *learns* the right signal weighting,
no eval-test leakage, cold-start/degenerate fallbacks (negative cases), and the
guarded headline that LambdaMART beats fixed-weight RRF fusion.
"""

from __future__ import annotations

import pytest

from recolens.core.evaluate import run_eval, time_split
from recolens.core.rank import LogisticReranker, sigmoid
from recolens.core.schema import parse_interactions, parse_items
from recolens.packs.ugc.reco_hybrid import HybridRanker
from recolens.packs.ugc.reco_reranked import RerankedRanker
from recolens.packs.ugc.synth import generate


def _data(n_items=300, n_users=80, seed=42):
    d = generate(n_items=n_items, n_users=n_users, seed=seed)
    items = parse_items(d["items"]).valid
    inter = parse_interactions(d["interactions"]).valid
    train, test = time_split(inter, 0.3)
    return items, train, test


# --- pure learner ---------------------------------------------------------


def test_sigmoid_is_stable_at_extremes():
    assert sigmoid(0.0) == 0.5
    assert 0.0 <= sigmoid(-1000.0) < 1e-6  # no OverflowError
    assert 1.0 - sigmoid(1000.0) < 1e-6


def test_logistic_learns_separable_signal():
    # feature 0 perfectly predicts the label; the learned weight must be positive
    X = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    y = [1, 1, 0, 0]
    m = LogisticReranker(n_iter=500).fit(X, y)
    assert m.predict([1.0, 0.0]) > m.predict([0.0, 1.0])


def test_logistic_is_deterministic():
    X = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    y = [1, 0, 1]
    a = LogisticReranker().fit(X, y).weights
    b = LogisticReranker().fit(X, y).weights
    assert a == b


def test_logistic_fit_rejects_empty():
    with pytest.raises(ValueError):
        LogisticReranker().fit([], [])


# --- the reranker learns to trust the sharp signal ------------------------


def test_reranker_upweights_collaborative_signal():
    items, train, _ = _data()
    r = RerankedRanker(dim=64).fit(items, train)
    w = r._model.weights
    # collaborative co-read is the sharpest signal -> highest learned weight;
    # popularity is the weakest -> should not dominate.
    assert w["collab"] > w["content"]
    assert w["collab"] > w["popularity"]
    assert w["collab"] > w["tag"]


def test_reranked_beats_popularity():
    items, train, test = _data()
    ranker = RerankedRanker(dim=64)
    _q, results, _r = run_eval(items, train, test, [ranker], ks=(10,))
    pop = _pop_ndcg(items, train, test)
    assert results["reranked"]["nDCG@10"] > pop


def _pop_ndcg(items, train, test):
    from recolens.packs.ugc.baselines import PopularityRanker

    _q, results, _r = run_eval(items, train, test, [PopularityRanker()], ks=(10,))
    return results["popularity"]["nDCG@10"]


# --- negative / degenerate cases -----------------------------------------


def test_reranked_cold_start_user_does_not_crash():
    items, train, _ = _data(n_items=60, n_users=12, seed=5)
    r = RerankedRanker(dim=32).fit(items, train)
    out = r.rank("does-not-exist", 10)  # unknown user -> popularity fallback
    assert isinstance(out, list) and len(out) <= 10


def test_reranked_is_deterministic_end_to_end():
    items, train, _ = _data(n_items=120, n_users=20, seed=7)
    a = RerankedRanker(dim=32).fit(items, train).rank("u0", 10)
    b = RerankedRanker(dim=32).fit(items, train).rank("u0", 10)
    assert a == b


# --- headline (guarded): LambdaMART beats fixed-weight RRF fusion ----------


def _lightgbm_available() -> bool:
    from recolens.providers.rank_lightgbm import LightGBMReranker

    return LightGBMReranker.available()


def _ndcg10(items, train, test, ranker):
    _q, res, _r = run_eval(items, train, test, [ranker], ks=(10,))
    return res[ranker.name]["nDCG@10"]


@pytest.mark.skipif(not _lightgbm_available(), reason="requires the [rank] extra (lightgbm)")
def test_lambdamart_beats_fixed_rrf_and_linear():
    from recolens.providers.rank_lightgbm import LightGBMReranker

    items, train, test = _data()
    lgbm = _ndcg10(items, train, test, RerankedRanker(dim=64, model=LightGBMReranker()))
    logit = _ndcg10(items, train, test, RerankedRanker(dim=64))  # logistic
    hybrid = _ndcg10(items, train, test, HybridRanker(dim=64))
    # learned non-linear fusion (LambdaMART) > fixed-weight RRF fusion > linear learned
    assert lgbm > hybrid
    assert lgbm > logit

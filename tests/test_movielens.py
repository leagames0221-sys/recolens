"""Real-data validation on MovieLens 100k (skipped unless the data is present).

Locks the headline claim that — on real data where no signal is oracle — the
learned reranker beats every single signal and the fixed-weight fusion. Uses the
dependency-free logistic reranker, so no [rank] extra is needed. The data is not
vendored (GroupLens license); run `python scripts/fetch_movielens.py` to enable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recolens.core.evaluate import run_eval, time_split
from recolens.core.schema import parse_interactions, parse_items
from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
from recolens.packs.ugc.reco_collab import CollaborativeRanker
from recolens.packs.ugc.reco_hybrid import HybridRanker
from recolens.packs.ugc.reco_reranked import RerankedRanker

_ML_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-100k"


def _available() -> bool:
    return (_ML_DIR / "u.data").exists() and (_ML_DIR / "u.item").exists()


pytestmark = pytest.mark.skipif(
    not _available(), reason="MovieLens 100k not fetched (run scripts/fetch_movielens.py)"
)


def _real_eval():
    from recolens.datasets.movielens import load

    data = load()
    items = parse_items(data["items"]).valid
    inter = parse_interactions(data["interactions"]).valid
    train, test = time_split(inter, 0.3)
    rankers = [
        PopularityRanker(),
        ContentRanker(dim=64),
        CollaborativeRanker(),
        HybridRanker(dim=64),
        RerankedRanker(dim=64),  # logistic (zero-dep)
    ]
    _q, results, _r = run_eval(items, train, test, rankers, ks=(10,))
    return results


def test_learned_reranker_wins_on_real_data():
    r = _real_eval()
    # the learned reranker beats the best single signal (collaborative) and the
    # fixed-weight fusion — the opposite of the near-oracle synthetic fixture.
    assert r["reranked"]["nDCG@10"] > r["collab"]["nDCG@10"]
    assert r["reranked"]["nDCG@10"] > r["hybrid"]["nDCG@10"]
    assert r["collab"]["nDCG@10"] > r["popularity"]["nDCG@10"]


def test_real_data_is_nontrivial_scale():
    from recolens.datasets.movielens import load

    data = load()
    assert len(data["items"]) > 1000
    assert len({x["user_id"] for x in data["interactions"]}) > 500

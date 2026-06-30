"""Phase 6 tests: benchmark structure + config reproducibility (R-PERF, R-CORE-4)."""

from __future__ import annotations

from recolens.core.bench import _percentile, bench_config
from recolens.core.embedding import DeterministicHashEmbed
from recolens.core.evaluate import run_eval, time_split
from recolens.core.schema import parse_interactions, parse_items
from recolens.core.vector_index import InMemoryFlatIndex
from recolens.packs.ugc.baselines import ContentRanker, PopularityRanker
from recolens.packs.ugc.synth import generate


def test_percentile_basic():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(xs, 50) == 3.0
    assert _percentile(xs, 95) == 5.0
    assert _percentile([], 50) == 0.0


def test_bench_config_reports_expected_keys():
    items = parse_items(generate(n_items=80, n_users=1, seed=1)["items"]).valid
    r = bench_config(
        items, DeterministicHashEmbed(dim=64), lambda _d: InMemoryFlatIndex(), n_queries=20, k=10
    )
    for key in (
        "dim",
        "embed_ms_per_text",
        "build_ms",
        "search_p50_ms",
        "search_p95_ms",
        "tag_match_at_k",
        "index_mem_kb_est",
    ):
        assert key in r
    assert 0.0 <= r["tag_match_at_k"] <= 1.0
    assert r["dim"] == 64


def test_config_reproducibility_same_seed_same_metrics():
    # R-CORE-4 / R-OPS: identical config -> identical metrics across runs
    def run():
        data = generate(n_items=200, n_users=50, seed=123)
        items = parse_items(data["items"]).valid
        inter = parse_interactions(data["interactions"]).valid
        train, test = time_split(inter, 0.3)
        _q, res, _r = run_eval(
            items, train, test, [PopularityRanker(), ContentRanker(dim=64)], ks=(5, 10)
        )
        return res

    assert run() == run()

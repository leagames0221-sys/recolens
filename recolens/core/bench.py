"""Cost x latency x quality benchmark for embed/index configs (R-PERF-1).

For each (embedder, index) configuration: corpus embed throughput, index build
time, search p50/p95 latency, an index memory estimate, and a tag-overlap
quality proxy. This is the "how fast / how cheap, at what quality" tradeoff the
case asks for (performance & cost improvement). Runs on CPU (R-PERF-2).

Latency is wall-clock, so absolute numbers vary run to run; the report is
regenerated, not golden-tested.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from recolens.core.embedding import EmbeddingProvider
from recolens.core.schema import Item, item_text
from recolens.core.vector_index import VectorIndex


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[idx]


def bench_config(
    items: Sequence[Item],
    embedder: EmbeddingProvider,
    index_factory: Callable[[int], VectorIndex],
    *,
    n_queries: int = 50,
    k: int = 10,
) -> dict[str, float]:
    ids = [it.item_id for it in items]
    texts = [item_text(it) for it in items]

    t0 = time.perf_counter()
    vectors = embedder.embed(texts)
    embed_s = time.perf_counter() - t0

    index = index_factory(embedder.dim)
    t0 = time.perf_counter()
    index.add(ids, vectors)
    build_s = time.perf_counter() - t0

    query_items = list(items)[:n_queries]
    qvecs = embedder.embed_queries([item_text(it) for it in query_items])
    tags_by_id = {it.item_id: set(it.tags) for it in items}

    latencies: list[float] = []
    matched = 0
    retrieved_total = 0
    for qit, qvec in zip(query_items, qvecs, strict=True):
        t0 = time.perf_counter()
        res = index.search(qvec, k + 1)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        retrieved = [iid for iid, _ in res if iid != qit.item_id][:k]
        qtags = tags_by_id[qit.item_id]
        matched += sum(1 for r in retrieved if tags_by_id.get(r, set()) & qtags)
        retrieved_total += len(retrieved)

    return {
        "dim": embedder.dim,
        "embed_ms_per_text": round(embed_s / max(1, len(texts)) * 1000.0, 3),
        "build_ms": round(build_s * 1000.0, 2),
        "search_p50_ms": round(_percentile(latencies, 50), 3),
        "search_p95_ms": round(_percentile(latencies, 95), 3),
        "tag_match_at_k": round(matched / retrieved_total, 4) if retrieved_total else 0.0,
        "index_mem_kb_est": round(len(items) * embedder.dim * 4 / 1024.0, 1),
    }

# Benchmark (generated)

Corpus = 300 items, 50 queries, k=10, CPU.
Latency is wall-clock (varies run to run); regenerate with `recolens bench --all`.

| config | dim | embed ms/text | build ms | search p50 ms | search p95 ms | tag-match@10 | index mem (KB est) |
|---|---|---|---|---|---|---|---|
| hash + memory | 64 | 0.026 | 1.03 | 0.83 | 0.973 | 0.956 | 75.0 |
| e5-small + memory | 384 | 7.238 | 6.85 | 5.888 | 6.102 | 1.0 | 450.0 |
| e5-small + qdrant | 384 | 6.976 | 62.43 | 0.45 | 1.015 | 1.0 | 450.0 |
| hash + qdrant | 64 | 0.035 | 15.54 | 0.43 | 0.533 | 0.956 | 75.0 |

**Reading it**: the hash embedder is near-zero cost and dependency-free (good default / fast iteration); e5-small trades higher embed cost for better semantic quality (see docs/evidence/embedding_spike_b1.md). Qdrant adds a small indexing overhead but scales beyond in-memory. Pick per the quality/latency/cost budget.

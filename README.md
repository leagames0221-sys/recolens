# recolens

> Local, free content recommendation & search mini-platform with a first-class
> **evaluation harness** — KPI / A-B / cost-latency — for the work a *platform*
> engineer actually owns.

![ci](https://img.shields.io/badge/ci-green-brightgreen)
![tests](https://img.shields.io/badge/tests-88%20passing-brightgreen)
![free](https://img.shields.io/badge/cost-%240-brightgreen)
![no-card](https://img.shields.io/badge/credit_card-not_required-brightgreen)
![local-llm](https://img.shields.io/badge/LLM-local%2Foptional-blue)
![license](https://img.shields.io/badge/license-MIT-blue)

Design decisions live in [`docs/adr/`](docs/adr/); prior-art and measured evidence
in [`docs/evidence/`](docs/evidence/).

## Demo

A real, offline session (reproduce with `python scripts/gen_demo_svg.py`):

![recolens demo](docs/demo/recolens-demo.svg)

The KPI / A-B report is a self-contained page: [`docs/demo-viewer/index.html`](docs/demo-viewer/index.html)
(regenerate with `python scripts/gen_report.py`).

## Why

Most recommender portfolios stop at *"I trained a model."* A platform engineer's
job is the rest of the loop: turning behavior logs + content into features,
serving search and recommendations over a vector index, and — critically —
**measuring** quality and cost with KPIs and A-B tests, reproducibly. recolens
demonstrates that whole loop, locally, for free, with no credit card.

## What it does (approach)

A two-layer design — domain-agnostic `core/` + `packs/ugc/` — wired as a pipeline
`ingest → feature → embed → index → serve → eval`, driven from one config. Nine CLI
commands:

| command | what |
|---|---|
| `ingest` | load Item/User/Interaction via a Protocol Buffers schema (rejects malformed records with counts) |
| `index` / `search` | embed items, build a vector index, kNN search |
| `recommend` | content / collaborative / hybrid / **reranked** top-N (with cold-start fallback) |
| `eval` | time-split offline metrics: Recall@K / nDCG / MRR / MAP / Coverage / Novelty |
| `ab` | offline A-B simulation with a KPI (hit-rate) + bootstrap 95% CI + decision |
| `bench` | embedding/ANN cost × latency × memory trade-off report |
| `classify` / `moderate` | LLM-as-judge category & quality; abuse / spam / prompt-injection filter |

Embeddings, Qdrant, and Ollama are **optional layers**. The default install has
**zero runtime dependencies** and runs deterministically offline.

## Results (measured, reproducible)

- **Metric arithmetic is correct by construction**: `core/metrics.py` agrees with
  **ir_measures** to **< 1e-9** on identical (qrels, run) input. (That proves the
  formulas are right — *not* that a benchmark is trustworthy; the latter is
  established by real-data validation, below.)
  ([evidence](docs/evidence/metrics_definitions.md))
- **Validated on real data (MovieLens 100k), not just a simulator.** Running the
  *identical* harness on the standard public benchmark (1,682 items / 942 users), on
  **nDCG@10** the learned reranker beats the best single signal (collaborative) at
  every tested hold-out ratio — LambdaMART by **+13% (ratio 0.3)**, and the zero-dep
  logistic default is positive across ratios {0.2, 0.3, 0.5} (the more robust winner).
  The margin narrows as the hold-out grows, and collaborative edges the reranker at
  rank-1 — so this is an nDCG@10 result, reported with its sensitivity, not a
  cherry-picked cell. ([evidence + sensitivity table](docs/evidence/real_data_movielens.md);
  reproduce with `python scripts/fetch_movielens.py` then `recolens eval --dataset movielens`)
- **The synthetic set is a controlled fixture, not the credibility source.**
  Default `recolens eval` (n=300/80, seed 42, stable across seeds {42, 7, 123, 99,
  2026}, golden-locked) plants a near-oracle co-read signal, so collaborative wins
  *by construction* — which is exactly what makes it a good unit test ("does the
  harness recover a known-planted signal, and refuse to let blind fusion beat an
  oracle?"). It is **not** evidence that a method is good; the real-data table above
  is.

  | method | nDCG@10 | RR | role (on the fixture) |
  |---|---|---|---|
  | collaborative | **0.469** | 0.772 | planted near-oracle signal |
  | reranked · LambdaMART | 0.279 | 0.480 | learned stage-2 rank (`[rank]` extra) |
  | hybrid | 0.235 | 0.324 | fixed-weight RRF fusion |
  | reranked · logistic | 0.207 | 0.315 | learned linear fusion (default, zero-dep) |
  | content | 0.201 | 0.325 | complementary signal |
  | popularity | 0.027 | 0.045 | baseline |

- **The industry-standard fix, implemented and measured across both regimes.** The
  2025-2026 answer to "fixed fusion can't beat a sharp signal" is a **two-stage
  retrieve → learned-rank** pipeline (signals retrieve, a model *learns* to rank).
  recolens ships it — a zero-dep logistic reranker by default, **LightGBM LambdaMART**
  (the production GBDT learning-to-rank workhorse) via `--reranker lightgbm`.
  - **On real data (MovieLens), on nDCG@10: learned reranking wins** — LambdaMART
    beats the best single signal by +13% (ratio 0.3), positive across ratios; the
    zero-dep logistic is the more robust winner (see sensitivity table).
  - **On the near-oracle fixture**: it beats fixed RRF fusion (+18.7% nDCG@10 / +48% RR)
    but honestly *cannot* beat the planted oracle — fusion helps for complementary
    signals, not against an oracle. Both regimes reported; neither tuned to a desired
    outcome. ([ADR-0010](docs/adr/ADR-0010-two-stage-learned-reranking.md),
    [evidence](docs/evidence/ltr_industry_standard_2026.md))
- **A-B simulation agrees**: `recolens ab --a content --b collab` →
  hit-rate@10 0.70 → 0.90, **+28.6%**, 95% CI [+0.09, +0.33], decision "B wins"
  (and identical variants correctly return *inconclusive*).
- **Cost/perf**: Qdrant local mode cut 384-dim search **p50 ~5.9ms → ~0.45ms
  (~13×)** vs brute force; local embedding throughput ~**310 texts/s on CPU**.
  ([report](docs/BENCHMARK.generated.md))
- **Safety**: `moderate` blocks prompt-injection inputs, flags spam, allows benign
  — verified with **negative examples**, not only happy-path tests.

## Quickstart

```bash
uv sync                 # zero runtime deps; deterministic, offline
uv run recolens ingest  # synthesize + load data (no external services)
uv run recolens eval    # offline metrics table + run manifest under runs/
uv run recolens ab --a content --b collab   # A-B with KPI + 95% CI + decision
```

Optional layers (off by default, still no credit card):

```bash
uv sync --extra embed   # real local embeddings (sentence-transformers e5/BGE)
uv sync --extra vector  # Qdrant backend (local mode, no docker needed); auto-fallback to in-memory
uv sync --extra llm     # local Ollama for classify / moderate; falls back to deterministic rules
uv sync --extra rank    # LightGBM LambdaMART stage-2 reranker; falls back to the logistic reranker

# then, e.g., compare the learned reranker against the fixed-weight fusion:
uv run --extra rank recolens eval --reranker lightgbm
```

Real-data validation (MovieLens 100k — downloaded on demand, not vendored):

```bash
python scripts/fetch_movielens.py                       # official host + SHA-256 pin
uv run --extra rank recolens eval --dataset movielens --reranker lightgbm
```

> **Note (protobuf):** run via `uv run` (or `uv sync` first). The committed
> generated code targets the protobuf runtime pinned in `uv.lock`; an older
> system-wide protobuf raises a version error by design — see
> [ADR-0007](docs/adr/ADR-0007-protobuf-contract.md).

## Design & decisions

> **Scope:** the **evaluation harness** is the load-bearing claim (metric correctness +
> real-data validation + method selection). The embedding/Qdrant/LLM/Protobuf layers
> are capability demos wired through the same contracts — deliberately optional, not
> the thesis.

- 2-layer `core/` + `packs/ugc/`; deterministic core, optional heavy layers.
- **Two-stage retrieve → learned rank** (industry-standard): signals retrieve
  candidates, a learned reranker (logistic default / LightGBM LambdaMART) orders
  them; trained on a within-train split with no test leakage ([ADR-0010](docs/adr/ADR-0010-two-stage-learned-reranking.md)).
- Metrics follow **BEIR / ir_measures** definitions (no home-grown metrics).
- Provider ABCs with env swap (`EMBED_PROVIDER` / `VECTOR_BACKEND` / `LLM_PROVIDER`).
- All choices recorded as ADRs: [`docs/adr/`](docs/adr/). Prior art:
  [`docs/evidence/prior_art_catalog.md`](docs/evidence/prior_art_catalog.md).

## Limitations (honest)

- **Default data is a synthetic *fixture*, not the credibility source.** The seeded
  generator plants a known signal so the harness has a deterministic unit test (it
  must recover the planted signal, and refuse to let blind fusion beat an oracle).
  Real-world claims rest on the **MovieLens 100k** validation (see Results / evidence),
  run through the identical harness. No real *user* data is vendored — MovieLens is
  downloaded on demand and never redistributed (GroupLens license).
- **Synthetic data cannot fairly rate *semantic* embeddings.** Its content signal
  is literal themed-token overlap, which actually favors the deterministic hash
  embedder over semantic e5 (we measured e5 ≈ −11% nDCG@10 here). So we do **not**
  headline an "embeddings beat hash" number — the harness supports real e5/BGE via
  `--extra embed`, but a fair semantic comparison needs real text.
- **Scale.** Designed and benchmarked at laptop scale (hundreds–thousands of
  items). It demonstrates the *patterns* (two-tower-style retrieval, ANN, offline
  eval, A-B) that scale on Databricks/Spark/Snowflake — it is not those systems.
- **A-B is offline simulation**, not a live experiment; it reports a KPI proxy with
  confidence intervals and says *inconclusive* rather than overclaiming.
- **LLM layer defaults to a deterministic mock** so CI is free and hermetic. Real
  Ollama accuracy on Japanese (model selection for shipping) is a separate,
  weight-download step, intentionally not run in CI.
- **"Two-tower" here is a reduced content/collaborative retrieval**, not a trained
  dual-encoder. The eval harness is the headline, not SOTA ranking accuracy.
- **The learned reranker wins on real data, not on the fixture — shown honestly.**
  On MovieLens it beats every single signal and fixed fusion; on the near-oracle
  synthetic fixture it beats fusion but cannot beat the planted oracle (expected).
  Features are currently the four signals' rank-reciprocals; richer features
  (recency, author, session, embedding score) would widen the real-data margin. See
  ADR-0010.

## Security posture

Zero default runtime deps; license allowlist (MIT/Apache-2.0/BSD/ISC) enforced by
`scripts/audit_deps.py`; pinned model revision + `safetensors` + telemetry off;
`uv.lock` pinning; pre-commit gitleaks + private-path sweep. See
[ADR-0006](docs/adr/ADR-0006-supply-chain-defense.md).

**Known dependency caveat:** the **default install has zero runtime dependencies
and audits clean** (`pip-audit`). The optional `embed` extra pulls `torch` (via
sentence-transformers), which currently carries 2 known advisories with no published
fix at any version:

- **PYSEC-2026-139** (High, local) — unsafe deserialization when loading **untrusted**
  model/checkpoint files. recolens loads a single, **revision-pinned** model with
  **`safetensors` enforced** (pickle refused) and never deserializes user-supplied
  model files, so this path is not exercised.
- **CVE-2025-3000** (Medium, local) — memory corruption in `torch.jit.script`, which
  recolens does not call.

Both are local-only, opt-in (absent from the default install), and disclosed here
rather than hidden. The pins will be bumped once upstream ships a fix.

## License

MIT — see [LICENSE](LICENSE).

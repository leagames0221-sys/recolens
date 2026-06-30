# recolens

> Local, free content recommendation & search mini-platform with a first-class
> **evaluation harness** — KPI / A-B / cost-latency — for the work a *platform*
> engineer actually owns.

![ci](https://img.shields.io/badge/ci-green-brightgreen)
![tests](https://img.shields.io/badge/tests-77%20passing-brightgreen)
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
| `recommend` | content / collaborative / hybrid top-N (with cold-start fallback) |
| `eval` | time-split offline metrics: Recall@K / nDCG / MRR / MAP / Coverage / Novelty |
| `ab` | offline A-B simulation with a KPI (hit-rate) + bootstrap 95% CI + decision |
| `bench` | embedding/ANN cost × latency × memory trade-off report |
| `classify` / `moderate` | LLM-as-judge category & quality; abuse / spam / prompt-injection filter |

Embeddings, Qdrant, and Ollama are **optional layers**. The default install has
**zero runtime dependencies** and runs deterministically offline.

## Results (measured, reproducible)

- **Metric correctness**: our `core/metrics.py` agrees with **ir_measures** to
  **< 1e-9** on identical (qrels, run) input — the harness is trustworthy by
  construction, not assertion. ([evidence](docs/evidence/metrics_definitions.md))
- **The harness discriminates, and the ranking is stable.** Default
  `recolens eval` (n=300/80, seed 42), nDCG@10 — reproduced across seeds
  {42, 7, 123, 99, 2026} and locked by a golden test:

  | method | nDCG@10 | RR | role |
  |---|---|---|---|
  | collaborative | **0.469** | 0.772 | best single (sharp co-read signal) |
  | hybrid | 0.235 | 0.324 | robust all-rounder (always > content, > popularity) |
  | content | 0.201 | 0.325 | strong complementary signal |
  | popularity | 0.027 | 0.045 | baseline — beaten ~9–17× by learned methods |

- **An honest negative result.** The hybrid does **not** beat collaborative here:
  when one signal is much sharper, rank fusion dilutes it. The harness surfaces
  this, so the production choice on this workload is collaborative — not blind
  fusion. ([ADR-0009](docs/adr/ADR-0009-hybrid-fusion-and-negative-result.md))
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
```

> **Note (protobuf):** run via `uv run` (or `uv sync` first). The committed
> generated code targets the protobuf runtime pinned in `uv.lock`; an older
> system-wide protobuf raises a version error by design — see
> [ADR-0007](docs/adr/ADR-0007-protobuf-contract.md).

## Design & decisions

- 2-layer `core/` + `packs/ugc/`; deterministic core, optional heavy layers.
- Metrics follow **BEIR / ir_measures** definitions (no home-grown metrics).
- Provider ABCs with env swap (`EMBED_PROVIDER` / `VECTOR_BACKEND` / `LLM_PROVIDER`).
- All choices recorded as ADRs: [`docs/adr/`](docs/adr/). Prior art:
  [`docs/evidence/prior_art_catalog.md`](docs/evidence/prior_art_catalog.md).

## Limitations (honest)

- **Synthetic data.** Items/users/interactions come from a seeded, two-signal
  simulator. Absolute metric values are illustrative; the **relative, reproducible**
  comparisons (collab vs content vs hybrid vs popularity, Qdrant vs brute force) are
  the point. No real user data is used — by design (privacy & security).
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

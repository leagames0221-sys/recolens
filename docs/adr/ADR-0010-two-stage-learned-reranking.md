# ADR-0010 — Two-stage retrieve → learned rank (the industry-standard fusion)

- **Status**: Accepted
- **Date**: 2026-07-01

## Context
ADR-0009 recorded a real limitation: fixed-weight rank fusion (RRF) cannot beat a
sharp single signal, because blending only dilutes it. The industry-standard
answer — used across large-scale search/recommendation in 2025-2026 — is not a
better fixed rule, but a **two-stage architecture**: cheap signals *retrieve*
candidates, then a **single model learns to rank** them from interaction data,
replacing hand-set weights with learned ones (and, with trees, non-linear
interactions). See `docs/evidence/ltr_industry_standard_2026.md` for cited
sources.

## Decision
1. **Add a stage-2 learned reranker** (`packs/ugc/reco_reranked.py`, method
   `reranked`). Stage 1 = the existing four signals produce candidates; stage 2 =
   a model scores each (user, candidate) from the signals' rank-reciprocal
   features and orders by predicted relevance.
2. **No test leakage.** The reranker trains on a *within-train* time split
   (signals fit on the earlier slice, labels from the later slice), then refits
   signals on the full train set for serving. The eval test slice is never seen.
3. **Two models behind one interface** (`core/rank.py`):
   - **Default = `LogisticReranker`** — a dependency-free, deterministic pointwise
     learned combiner. Keeps the default install zero-dep and CI hermetic.
   - **Optional = `LightGBMReranker`** (LambdaMART, `objective="lambdarank"`) behind
     the `[rank]` extra — the production-grade GBDT learning-to-rank workhorse
     (LightGBM, Microsoft, **MIT**). Falls back to logistic when unavailable.

## Measured result (seed 42, n=300/80, `recolens eval`)
| method | nDCG@10 | RR |
|---|---|---|
| collaborative (near-oracle here) | 0.469 | 0.772 |
| **reranked — LightGBM LambdaMART** | **0.279** | **0.480** |
| hybrid (fixed-weight RRF) | 0.235 | 0.324 |
| reranked — logistic (linear) | 0.207 | 0.315 |

- **Learned non-linear fusion beats fixed fusion**: LambdaMART > RRF hybrid by
  **+18.7% nDCG@10 / +48% RR**, and beats the linear learned combiner by +34%
  nDCG@10 — exactly the expected ordering (non-linear listwise > fixed > linear).
- **Honest nuance**: no combiner beats *collaborative* on this workload, because
  the synthetic co-read signal is **near-oracle** (positives are generated from
  co-read cohorts). Fusion helps when signals are *complementary*, not when one is
  already near-oracle. On real, noisier data no single signal is oracle, which is
  precisely where learned reranking pays off. We report this rather than tuning
  the data to manufacture a win (cf. ADR-0009).

## Alternatives considered
- **A better fixed rule (weighted RRF tuning).** Rejected as the "fix": it is
  still hand-set and cannot adapt per-query; ADR-0009 already shows its ceiling.
- **Make LightGBM a hard dependency.** Rejected — it would break the zero-dep
  default and CI hermeticity (C-1). It is an opt-in extra with a pure-Python
  fallback, matching how embed/vector/llm are handled.
- **Neural ranker (DLRM/DCN).** Overkill for laptop scale and a heavy dep; GBDT
  LambdaMART is the standard, efficient choice under these constraints.

## Consequences
- `reranked` (logistic) is locked by the golden regression test; the LambdaMART
  headline is guarded by a `skipif`-gated test that runs under the `[rank]` extra
  (`tests/test_reranker.py::test_lambdamart_beats_fixed_rrf_and_linear`).
- The portfolio now demonstrates the *current* industry architecture end-to-end
  (retrieve → learned rank) and measures it honestly, including where it helps and
  where it does not.

# ADR-0005 — Metric definitions follow IR/RecSys standards (no home-grown metrics)

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
The evaluation harness is the headline deliverable. Subtly wrong metric formulas
(e.g. a non-standard nDCG discount, or Recall divided by the wrong denominator)
would invalidate every downstream claim and read as amateur to a reviewer.

## Decision
- Implement Recall@K / nDCG@K / MRR / MAP / Coverage / Novelty in `core/metrics.py`
  following the definitions used by **ir_measures / trec_eval** and **RecBole**.
- Pin those definitions in `docs/evidence/metrics_definitions.md` with primary
  source URLs, recorded before implementation.
- A conformance test feeds identical (qrels, run) data to **ir_measures** and to
  our implementation and asserts agreement to **< 1e-9** for the metrics
  ir_measures covers; ranking-list metrics carry negative-example tests.

## Alternatives considered
- **Home-grown formulas**: faster to write, but unverifiable and risky. Rejected.
- **Depend on ir_measures at runtime**: pulls trec_eval (separate license) and a
  C build into the default path. Rejected — ir_measures stays a **dev/test** extra
  used as the oracle, keeping the runtime dependency-free (see ADR-0006).

## Consequences
- Numbers are trustworthy and reproducible; the cross-check is itself evidence.
- Adding a metric requires extending the conformance test, preventing drift.

Sources: https://ir-measur.es/ , https://github.com/terrierteam/ir_measures , https://recbole.io/

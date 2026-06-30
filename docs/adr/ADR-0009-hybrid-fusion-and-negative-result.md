# ADR-0009 — Hybrid fusion design, and an honest negative result

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
recolens ships four rankers (popularity, content, collaborative, hybrid) and an
eval harness whose whole purpose is to say *which method to run*. An earlier draft
claimed "the hybrid wins." Re-running the default `recolens eval` across multiple
seeds showed that claim was **not reproducible** — on the weak-signal synthetic
data the method ranking flipped seed-to-seed, and on one seed the popularity
baseline beat everything. A portfolio whose headline is "trustworthy evaluation"
cannot ship a cherry-picked, non-reproducible result.

## Decision
1. **Make the benchmark discriminating and stable.** The synthetic generator now
   encodes two strong, disjoint signals (`packs/ugc/synth.py`): a per-user
   single-tag *content* signal and a small-cohort *co-read* collaborative signal.
   This produces a **stable ranking across seeds and sizes**.
2. **Improve the hybrid to weighted RRF over genuine collaborative hits**
   (`reco_hybrid.py`): collaborative is up-weighted and fused from real
   co-occurrence only (no popularity padding) so its sharp signal is not diluted.
3. **Report the result honestly, including the negative finding.** Measured
   nDCG@10 ordering, stable across seeds {42, 7, 123, 99, 2026} and sizes:

   | method | nDCG@10 (seed 42, n=300/80) | role |
   |---|---|---|
   | collaborative | **0.469** | best single — the co-read signal is sharp |
   | hybrid | 0.235 | robust all-rounder (always > content, > popularity) |
   | content | 0.201 | strong complementary signal |
   | popularity | 0.027 | baseline, beaten ~9–17× by learned methods |

   **The hybrid does NOT beat collaborative on this workload.** When one signal is
   much sharper than the others, rank fusion can only dilute it — a well-known
   property, here surfaced by the harness. The production choice on this workload
   is collaborative, or a *supervised* combiner; blind fusion is not a free lunch.

## Alternatives considered
- **Tune the data until the hybrid "wins."** Rejected — manufacturing a
  predetermined headline is dishonest and defeats the point of the harness.
- **Equal-weight RRF.** Rejected as the hybrid's default: it underperforms even
  content-aware weighting and is further from collaborative.
- **Drop the hybrid.** Rejected — its robustness (never worst, always beats
  content/popularity) is a real, testable property worth showing.

## Consequences
- The headline numbers are reproducible from the documented command and locked by
  a golden regression test (`tests/test_eval_ab_golden.py`).
- The ranking is asserted across multiple seeds, not one (`test_recommenders.py::
  test_method_ranking_is_stable_and_honest`), so the claim cannot silently rot.
- The portfolio demonstrates measurement-driven method selection and the
  willingness to report a negative result — the judgment the harness exists for.

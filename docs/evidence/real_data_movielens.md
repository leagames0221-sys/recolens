# Real-data validation — MovieLens 100k

> Answers the sharpest critique of a synthetic benchmark: *"does any of this hold
> on data you didn't author?"* The **identical harness** runs on a real, public
> dataset where no signal is oracle. Verified 2026-07-01.

## Why this matters
The synthetic generator plants a near-oracle co-read signal, so collaborative wins
there *by construction* (see ADR-0009/0010). That makes synthetic a good **fixture**
("does the harness recover a known-planted signal?") but a poor credibility source.
MovieLens 100k is the standard IR/recsys benchmark; here no method is oracle, so the
ranking is earned, not authored.

## Dataset (not vendored — GroupLens forbids redistribution)
- MovieLens 100k: **1,682 items, 942 users, 55,375 positive interactions** (rating ≥ 4).
- Downloaded on demand by `scripts/fetch_movielens.py` (pinned official HTTPS host +
  SHA-256 `50d2a982…3229`), extracted to a gitignored `data/`. CI never uses it.
- Citation: F. M. Harper & J. A. Konstan, *The MovieLens Datasets: History and
  Context*, ACM TiiS 5(4), 2015. https://doi.org/10.1145/2827872

## Result — same harness, real data
Reproduce: `python scripts/fetch_movielens.py` then
`uv run --extra rank recolens eval --dataset movielens --reranker lightgbm`

| method | nDCG@10 | RR |
|---|---|---|
| **reranked · LightGBM LambdaMART** | **0.168** | **0.323** |
| reranked · logistic (zero-dep default) | 0.160 | 0.315 |
| collaborative | 0.149 | 0.297 |
| hybrid (fixed-weight RRF) | 0.146 | 0.280 |
| popularity | 0.109 | 0.239 |
| content (genre-tag only) | 0.025 | 0.058 |

## What it establishes
- **On real data, learned reranking beats every single signal *and* fixed fusion.**
  LambdaMART > best single signal (collaborative) by **+12.9% nDCG@10 / +8.8% RR**;
  > fixed-weight RRF hybrid by **+15.5% nDCG@10**. Even the dependency-free logistic
  reranker beats collaborative (+7.6% nDCG@10). This is the industry-standard result
  the synthetic near-oracle setup could not show.
- **The harness discriminates consistently across both regimes** — it recovers the
  planted signal on the fixture *and* ranks methods sensibly on real data. That is
  the actual evidence that the evaluation is trustworthy, not the metric arithmetic
  alone.
- **Honest cross-regime reading**: fusion cannot beat a near-oracle single signal
  (synthetic), but *does* beat every signal when signals are complementary and noisy
  (real). Both are reported; neither is tuned to a desired outcome.

## Scope note
Content is weak here because the only content signal from ml-100k is coarse genre
tags (19 genres) over a hash embedder — not semantic text. This is a property of the
dataset's features, not the harness; richer item text would lift the content and
embedding rows.

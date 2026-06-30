# ADR-0001 — Decomposed prior-art seeds (what we imitate vs. build)

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
recolens is a local, free recommendation & search mini-platform. Rather than
generating everything from scratch, we extract the proven core ideas from
established, permissively-licensed projects and reassemble them. No single
template fits the whole tool, so we take a *decomposed* prior-art approach.

## Decision
From the seven references catalogued in
[prior_art_catalog.md](../evidence/prior_art_catalog.md), we adopt:
- **Skeleton** = Microsoft Recommenders' five tasks (Prepare / Model / Evaluate /
  Select-Optimize / Operationalize).
- **Config-driven reproducibility** = RecBole's style.
- **Vector search** = Qdrant (optional layer) + an in-memory fallback.
- **Embeddings** = sentence-transformers (e5 / BGE, local).
- **Eval metrics** = BEIR / ir_measures standard definitions (no home-grown metrics).
- **Candidate generation** = a reduced form of the two-tower design (content-based
  first; a simple two-tower as a stretch).

The connective tissue — pipeline, CLI, eval harness, A-B simulation, cost/perf
benchmark — is written from scratch.

## Alternatives considered
- **Build everything from scratch**: wasteful where world standards exist
  (metrics, retrieval). Rejected.
- **Fork RecBole / MS Recommenders wholesale and modify**: heavyweight modification
  of a <80%-fit template accrues more debt than reuse. Rejected; extract the core
  instead.
- **Implement metrics ourselves**: diverges from the primary standard
  (BEIR / ir_measures) and undermines the harness's credibility. Rejected.

## Consequences
- Every adopted source is permissive (MIT / Apache-2.0) and passed a dependency
  security check.
- Heavy dependencies (torch / sentence-transformers / qdrant-client / ollama) are
  isolated into optional extras; the default install runs deterministically with
  zero runtime dependencies.
- This ADR makes "what we imitated vs. built" explicit and auditable.

# ADR-0004 — Vector backend: Qdrant with in-memory fallback

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
The target role names **Qdrant** as the managed vector store. The tool must also
run with zero external services on a fresh clone (no docker, no credit card).

## Decision
- `core/vector_index.py` defines a `VectorIndex` ABC. The **default** is
  `InMemoryFlatIndex` (exact brute-force kNN, deterministic) — no dependency.
- `providers/vector_qdrant.py` wraps Qdrant via the **local mode** (`:memory:` /
  embedded), so the demo needs no docker daemon; the same code path scales to a
  hosted Qdrant by changing the connection string.
- If `qdrant-client` is absent or a server is unreachable, search **auto-falls
  back** to `InMemoryFlatIndex` without losing functionality (R-SEARCH-3). A test
  asserts both backends return the same top-K on identical input.

## Alternatives considered
- **FAISS**: fast, but heavier to install on Windows/CPU and not what the role
  uses. Kept out to match the job stack and the zero-friction constraint.
- **pgvector**: needs a Postgres instance; violates zero-service default.

## Consequences
- Recruiter sees the exact tool from the posting, exercised in code.
- A clean clone still runs end-to-end with the in-memory backend.
- Benchmark (ADR-linked `docs/BENCHMARK.generated.md`) measured Qdrant local mode
  cutting 384-dim search p50 ~5.7ms → ~0.54ms vs the brute-force baseline.

Sources: https://github.com/qdrant/qdrant , https://qdrant.tech/documentation/quickstart/

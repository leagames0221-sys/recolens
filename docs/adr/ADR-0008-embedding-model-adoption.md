# ADR-0008 — Local embedding model adoption + supply-chain hardening

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
Phase 3 needs real (non-hash) embeddings for content understanding and vector
search. Pulling a model + ML runtime is an external-source adoption, which must
pass a supply-chain security gate (audit + explicit approval) before use.

## Decision
- **Model**: `intfloat/multilingual-e5-small` — 384-dim, multilingual (incl. JP),
  free, public (no auth / no credit card). Swappable to base / large / BGE-m3 via config.
- **Revision pinned** to commit SHA `614241f622f53c4eeff9890bdc4f31cfecc418b3`
  (verified via HF API 2026-06-30). A moving tag could be repointed to a malicious
  revision; pinning removes that vector.
- **safetensors enforced** (`use_safetensors=True`). The repo also ships
  `pytorch_model.bin` (pickle, can execute arbitrary code on load) — we never load it.
- **Optional extra** `[embed]`: torch (BSD), transformers (Apache-2.0),
  sentence-transformers (Apache-2.0). Isolated from the default install — core
  stays zero-dependency and runs on the deterministic hash embedder offline.
- Telemetry disabled; runs on CPU; offline after first download.

## Audit record (security gate)
- License: model = MIT; runtime libs = BSD / Apache-2.0; verified by `scripts/audit_deps.py`
  (one transitive `certifi` = MPL-2.0, used unmodified — acceptable).
- Weights: safetensors only (no pickle execution path).
- Source: official Hugging Face repo, pinned revision.
- No API key / credit card / sign-up required.

## Alternatives considered
- **BGE-m3 / e5-large** as default: ~2.2GB each — heavier download/RAM. Kept as
  config-swap options; small is the default for laptop footprint.
- **Hosted embedding API**: needs key / billing — violates the free / no-card constraint.
- **Unpinned `main` revision**: simpler but exposes a supply-chain repoint vector. Rejected.

## Consequences
- Real embeddings improve ranking quality (B-1: nDCG@10 +25%, R@10 +37% vs hash;
  see docs/evidence/embedding_spike_b1.md).
- Default (no extra) behaviour is unchanged and dependency-free.

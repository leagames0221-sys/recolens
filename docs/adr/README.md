# Architecture Decision Records

Canonical 4-field (Context / Decision / Alternatives / Consequences).

| ADR | Decision |
|---|---|
| [0001](ADR-0001-decomposed-prior-art.md) | Decomposed prior-art reuse (what we imitated vs built) |
| [0002](ADR-0002-language-and-packaging.md) | Python + uv (rye(uv backend) compatible; rye is frozen) |
| [0003](ADR-0003-llm-provider-default.md) | Default LLM = deterministic mock + optional local Ollama |
| [0004](ADR-0004-vector-backend.md) | Qdrant with in-memory fallback |
| [0005](ADR-0005-metric-conformance.md) | Metrics follow ir_measures / RecBole (cross-checked < 1e-9) |
| [0006](ADR-0006-supply-chain-defense.md) | Zero-runtime-dep + license allowlist + pinned/verified weights |
| [0007](ADR-0007-protobuf-contract.md) | Protocol Buffers as the cross-service contract |
| [0008](ADR-0008-embedding-model-adoption.md) | Embedding model adoption (pinned rev, safetensors, telemetry off) |
| [0009](ADR-0009-hybrid-fusion-and-negative-result.md) | Hybrid fusion design + honest negative result (collab beats the hybrid here) |

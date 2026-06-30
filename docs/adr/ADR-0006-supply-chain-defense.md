# ADR-0006 — Supply-chain defense & zero-runtime-dependency posture

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
The tool downloads model weights and pulls Python packages. A portfolio that is
careless about provenance undercuts its own credibility for a platform role, and
real supply-chain attacks (malicious package versions, weight tampering) are a
live threat.

## Decision
- **Zero runtime dependencies by default.** Heavy/optional layers (`embed`,
  `vector`, `llm`, `dev`) live in extras; the default install runs deterministically.
- **License allowlist**: `scripts/audit_deps.py` permits only MIT / Apache-2.0 /
  BSD / ISC and exits non-zero otherwise; it ships with negative-example tests.
- **Pinned, verified models**: embedding weights are loaded by a pinned revision,
  `safetensors` is forced (no arbitrary pickle execution), and model telemetry is
  disabled (see ADR-0008).
- **Lockfile**: `uv.lock` pins exact versions; CI installs from the lock.
- **Secret/path hygiene**: pre-commit runs gitleaks + a private-path sweep; a
  wordlist sweep is mandatory before any public release.

## Alternatives considered
- **Unpinned installs / arbitrary licenses**: convenient, but exactly the exposure
  this project is meant to demonstrate guarding against. Rejected.

## Consequences
- A reviewer can audit exactly what runs and under what license.
- The default path has no third-party runtime surface to attack.

Sources: https://protobuf.dev/support/cross-version-runtime-guarantee , https://docs.astral.sh/uv/

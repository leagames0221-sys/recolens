# ADR-0002 — Language & packaging: Python + uv

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
The target role specifies a Python ML stack and `rye (uv backend)` for packaging.
The tool must `clone → one command → run` on a consumer laptop, free.

## Decision
- **Python** (>=3.10), packaged with **uv** + hatchling. Runtime dependencies are
  empty; heavy optional layers live in extras (`[embed]`, `[vector]`, `[llm]`, `[dev]`).
- `rye (uv backend)` is satisfied by adopting **uv directly**: as of 2026 rye is
  frozen (no further updates, including security) and Astral positions uv as its
  successor, sharing the backend. README states this explicitly.
- Tests run `ruff` + `pytest` together (lint/format as part of the test gate).

## Alternatives considered
- **rye directly**: matches the wording literally, but it is frozen — adopting a
  no-longer-maintained tool is a liability. Rejected; uv is the maintained path.
- **poetry / pip-tools**: slower, heavier; uv is the current fast standard.

## Consequences
- One lockfile, fast installs, zero-dep default install.
- Demonstrates current ecosystem awareness (uv over frozen rye) rather than
  cargo-culting the job-post wording.

Sources: https://rye.astral.sh/ , https://docs.astral.sh/uv/

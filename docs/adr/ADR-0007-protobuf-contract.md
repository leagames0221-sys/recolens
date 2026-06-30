# ADR-0007 — Protocol Buffers as the cross-service data contract

- **Status**: Accepted
- **Date**: 2026-06-30

## Context
The target role states that inter-service communication uses **Protocol Buffers**.
The platform's records (Item / User / Interaction / Recommendation) cross the
ingest → feature → index → serve → eval boundary and benefit from a typed,
versioned schema.

## Decision
- Define the core records in `proto/recolens.proto` and generate `recolens_pb2.py`
  via `scripts/gen_proto.sh`.
- Use protobuf as the **serialization/wire contract** at boundaries; in-process
  logic uses thin dataclass wrappers (`core/schema.py`) for ergonomics and
  validation (rejecting records missing required fields, with counts — R-CORE-2).
- A round-trip codec test asserts serialize→deserialize equality.

## Alternatives considered
- **Dataclasses / JSON only**: simpler, but does not demonstrate the typed,
  versioned cross-service contract the role asks for. Used internally, not as the
  boundary contract.

## Consequences
- Matches the posting's stack; schema evolution is explicit.
- **Version pin caveat**: generated code (gencode) and the protobuf runtime must
  be compatible. `uv.lock` resolves a runtime matching the committed gencode
  (6.33.x); running under an older system protobuf raises a VersionError. The
  canonical path is `uv run`, and CI uses the lock. Documented in README.

Sources: https://protobuf.dev/ , https://protobuf.dev/support/cross-version-runtime-guarantee

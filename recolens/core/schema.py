"""Canonical in-memory data model + validation (zero-dep).

The wire contract is defined in ``recolens/proto/recolens.proto`` (Protocol
Buffers). The local pipeline operates on these dataclasses so that default
operation needs no third-party runtime dependency (C-1). ``proto/codec.py``
bridges the two when a protobuf boundary is exercised.

Validation rejects records missing required fields and reports the reason —
records are never silently dropped (R-CORE-2).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    item_id: str
    title: str = ""
    body: str = ""
    tags: tuple[str, ...] = ()
    author_id: str = ""
    created_ts: int = 0


@dataclass(frozen=True)
class User:
    user_id: str
    interests: tuple[str, ...] = ()
    created_ts: int = 0


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    event: str = "view"
    ts: int = 0
    weight: float = 1.0


@dataclass(frozen=True)
class ScoredItem:
    item_id: str
    score: float


@dataclass
class ParseResult:
    valid: list = field(default_factory=list)
    rejected: list[tuple[dict, str]] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return len(self.valid)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def _require(record: dict, keys: tuple[str, ...]) -> str | None:
    """Return a reason string if any required key is missing/empty, else None."""
    for k in keys:
        v = record.get(k)
        if v is None or (isinstance(v, str) and v == ""):
            return f"missing required field: {k}"
    return None


def parse_items(records: Iterable[dict]) -> ParseResult:
    res = ParseResult()
    for r in records:
        reason = _require(r, ("item_id",))
        if reason:
            res.rejected.append((r, reason))
            continue
        res.valid.append(
            Item(
                item_id=str(r["item_id"]),
                title=str(r.get("title", "")),
                body=str(r.get("body", "")),
                tags=tuple(r.get("tags", ()) or ()),
                author_id=str(r.get("author_id", "")),
                created_ts=int(r.get("created_ts", 0)),
            )
        )
    return res


def parse_users(records: Iterable[dict]) -> ParseResult:
    res = ParseResult()
    for r in records:
        reason = _require(r, ("user_id",))
        if reason:
            res.rejected.append((r, reason))
            continue
        res.valid.append(
            User(
                user_id=str(r["user_id"]),
                interests=tuple(r.get("interests", ()) or ()),
                created_ts=int(r.get("created_ts", 0)),
            )
        )
    return res


def parse_interactions(records: Iterable[dict]) -> ParseResult:
    res = ParseResult()
    for r in records:
        reason = _require(r, ("user_id", "item_id", "ts"))
        if reason:
            res.rejected.append((r, reason))
            continue
        res.valid.append(
            Interaction(
                user_id=str(r["user_id"]),
                item_id=str(r["item_id"]),
                event=str(r.get("event", "view")),
                ts=int(r["ts"]),
                weight=float(r.get("weight", 1.0)),
            )
        )
    return res


def item_text(item: Item) -> str:
    """Text used for embedding an item (title + body + tags)."""
    return " ".join([item.title, item.body, " ".join(item.tags)]).strip()

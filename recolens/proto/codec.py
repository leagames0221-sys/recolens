"""Protobuf wire codec (the system-to-system contract boundary).

Bridges the zero-dep dataclasses (core/schema.py) and the protobuf wire format
(recolens.proto). Importing this module requires the generated bindings +
protobuf runtime (the ``[dev]`` / ``[proto]`` extra); the default pipeline does
not import it, keeping core dependency-free (C-1).

This is where another system would (de)serialize recolens records off the wire.
"""

from __future__ import annotations

from recolens.core.schema import Item, ScoredItem, User
from recolens.proto import recolens_pb2 as pb


def item_to_proto(item: Item) -> pb.Item:
    return pb.Item(
        item_id=item.item_id,
        title=item.title,
        body=item.body,
        tags=list(item.tags),
        author_id=item.author_id,
        created_ts=item.created_ts,
    )


def item_from_proto(msg: pb.Item) -> Item:
    return Item(
        item_id=msg.item_id,
        title=msg.title,
        body=msg.body,
        tags=tuple(msg.tags),
        author_id=msg.author_id,
        created_ts=msg.created_ts,
    )


def item_to_bytes(item: Item) -> bytes:
    return item_to_proto(item).SerializeToString()


def item_from_bytes(data: bytes) -> Item:
    msg = pb.Item()
    msg.ParseFromString(data)
    return item_from_proto(msg)


def user_to_bytes(user: User) -> bytes:
    return pb.User(
        user_id=user.user_id,
        interests=list(user.interests),
        created_ts=user.created_ts,
    ).SerializeToString()


def scored_items_to_proto(user_id: str, scored: list[ScoredItem]) -> pb.Recommendation:
    return pb.Recommendation(
        user_id=user_id,
        items=[pb.ScoredItem(item_id=s.item_id, score=s.score) for s in scored],
    )

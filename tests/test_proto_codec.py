"""Proto wire-contract round-trip (requires [dev] extra: protobuf bindings)."""

from __future__ import annotations

import pytest

pytest.importorskip("google.protobuf")
pytest.importorskip("recolens.proto.recolens_pb2")

from recolens.core.schema import Item  # noqa: E402
from recolens.proto import codec  # noqa: E402


def test_item_roundtrip_bytes():
    item = Item(
        item_id="item-0001",
        title="Winter Night Journey",
        body="a long body of words",
        tags=("fantasy", "poetry"),
        author_id="author-003",
        created_ts=1_700_000_000,
    )
    restored = codec.item_from_bytes(codec.item_to_bytes(item))
    assert restored == item


def test_item_bytes_are_nonempty_and_stable():
    item = Item(item_id="i", title="t")
    b1 = codec.item_to_bytes(item)
    b2 = codec.item_to_bytes(item)
    assert b1 == b2 and len(b1) > 0

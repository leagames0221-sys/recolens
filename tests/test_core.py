"""Phase 1 tests: schema validation, embedding, index, pipeline, proto codec.

Negative cases are mandatory for validation/guard paths (confirmatory-test 禁止).
"""

from __future__ import annotations

from recolens.core.embedding import DeterministicHashEmbed
from recolens.core.pipeline import build_item_index
from recolens.core.schema import (
    Item,
    item_text,
    parse_interactions,
    parse_items,
    parse_users,
)
from recolens.core.vector_index import InMemoryFlatIndex
from recolens.packs.ugc.synth import generate

# --- schema validation: positive + negative (missing required fields) ---


def test_parse_items_accepts_valid():
    res = parse_items([{"item_id": "i1", "title": "hello", "tags": ["a"]}])
    assert res.n_valid == 1 and res.n_rejected == 0
    assert res.valid[0].item_id == "i1"


def test_parse_items_rejects_missing_id_with_reason():
    res = parse_items([{"title": "no id"}, {"item_id": "", "title": "empty id"}])
    # 落ちるべき入力は reject され理由が付く(silent drop しない)
    assert res.n_valid == 0
    assert res.n_rejected == 2
    assert all("item_id" in reason for _, reason in res.rejected)


def test_parse_interactions_requires_user_item_ts():
    rows = [
        {"user_id": "u1", "item_id": "i1", "ts": 10},  # ok
        {"user_id": "u1", "item_id": "i1"},  # missing ts -> reject
        {"item_id": "i1", "ts": 10},  # missing user_id -> reject
    ]
    res = parse_interactions(rows)
    assert res.n_valid == 1
    assert res.n_rejected == 2


def test_parse_users_rejects_missing_id():
    res = parse_users([{"interests": ["x"]}])
    assert res.n_rejected == 1 and res.n_valid == 0


# --- embedding ---


def test_embedding_is_deterministic_and_normalized():
    emb = DeterministicHashEmbed(dim=32)
    a = emb.embed(["winter night journey"])
    b = emb.embed(["winter night journey"])
    assert a == b  # deterministic across calls
    norm = sum(x * x for x in a[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-9 or norm == 0.0
    assert len(a[0]) == 32


def test_embedding_distinguishes_texts():
    emb = DeterministicHashEmbed(dim=64)
    v1, v2 = emb.embed(["fantasy dragon", "tax accounting spreadsheet"])
    cos = sum(x * y for x, y in zip(v1, v2, strict=True))
    assert cos < 0.99  # different texts are not identical


# --- vector index ---


def test_inmemory_index_topk_and_selfmatch():
    emb = DeterministicHashEmbed(dim=64)
    items = [
        Item(item_id="a", title="winter night"),
        Item(item_id="b", title="summer beach"),
        Item(item_id="c", title="winter snow night"),
    ]
    idx = build_item_index(items, emb, InMemoryFlatIndex())
    assert len(idx) == 3
    q = emb.embed(["winter night"])[0]
    top = idx.search(q, k=2)
    assert top[0][0] == "a"  # exact match ranks first
    assert len(top) == 2


def test_index_length_mismatch_raises():
    idx = InMemoryFlatIndex()
    try:
        idx.add(["a", "b"], [[1.0, 0.0]])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


# --- synth + reproducibility ---


def test_synth_is_reproducible():
    d1 = generate(n_items=30, n_users=10, seed=7)
    d2 = generate(n_items=30, n_users=10, seed=7)
    assert d1 == d2
    assert len(d1["items"]) == 30
    assert len(d1["users"]) == 10
    assert len(d1["interactions"]) > 0


def test_synth_all_items_pass_validation():
    data = generate(n_items=50, n_users=10, seed=1)
    res = parse_items(data["items"])
    assert res.n_rejected == 0 and res.n_valid == 50


def test_item_text_combines_fields():
    t = item_text(Item(item_id="x", title="T", body="B", tags=("a", "b")))
    assert "T" in t and "B" in t and "a" in t and "b" in t

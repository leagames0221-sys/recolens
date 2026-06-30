"""Provider tests for the optional [embed] / [vector] extras.

These importorskip the heavy deps, so they run locally (when the extras are
installed) and are skipped in CI (which installs only [dev]). The default,
always-tested paths live in test_core.py / test_metrics.py.
"""

from __future__ import annotations

import pytest

from recolens.core.schema import Item, item_text


def test_qdrant_index_matches_inmemory_topk():
    pytest.importorskip("qdrant_client")
    from recolens.core.embedding import DeterministicHashEmbed
    from recolens.core.vector_index import InMemoryFlatIndex
    from recolens.providers.vector_qdrant import QdrantIndex

    emb = DeterministicHashEmbed(dim=64)
    items = [
        Item(item_id="a", title="winter night journey"),
        Item(item_id="b", title="summer beach holiday"),
        Item(item_id="c", title="winter snow night cold"),
    ]
    ids = [it.item_id for it in items]
    vecs = emb.embed([item_text(it) for it in items])

    mem = InMemoryFlatIndex()
    mem.add(ids, vecs)
    qd = QdrantIndex(dim=64)
    qd.add(ids, vecs)
    assert len(qd) == 3

    q = emb.embed(["winter night"])[0]
    top_mem = [i for i, _ in mem.search(q, 3)]
    top_qd = [i for i, _ in qd.search(q, 3)]
    # same top-1 (exact match) and same retrieved set
    assert top_qd[0] == top_mem[0] == "a"
    assert set(top_qd) == set(top_mem)


def test_local_embedder_dim_and_normalized():
    pytest.importorskip("sentence_transformers")
    from recolens.providers.embed_local import LocalSentenceTransformerEmbed

    emb = LocalSentenceTransformerEmbed()
    assert emb.dim == 384
    v = emb.embed(["a fantasy story about a dragon"])[0]
    assert len(v) == 384
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-3  # normalize_embeddings=True


def test_local_embedder_query_vs_passage_differ():
    pytest.importorskip("sentence_transformers")
    from recolens.providers.embed_local import LocalSentenceTransformerEmbed

    emb = LocalSentenceTransformerEmbed()
    text = "dragons and wizards"
    p = emb.embed([text])[0]
    q = emb.embed_queries([text])[0]
    # e5 applies different prefixes -> embeddings are not identical
    cos = sum(a * b for a, b in zip(p, q, strict=True))
    assert cos < 0.9999

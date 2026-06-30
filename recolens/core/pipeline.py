"""Pipeline glue: build an item vector index from parsed items.

ingest -> feature(item_text) -> embed -> index. Deterministic and offline by
default (R-CORE-3). Richer feature extraction lives in packs/ugc/features.py (P4).
"""

from __future__ import annotations

from collections.abc import Sequence

from recolens.core.embedding import EmbeddingProvider
from recolens.core.schema import Item, item_text
from recolens.core.vector_index import VectorIndex


def build_item_index(
    items: Sequence[Item],
    embedder: EmbeddingProvider,
    index: VectorIndex,
) -> VectorIndex:
    texts = [item_text(it) for it in items]
    vectors = embedder.embed(texts)
    index.add([it.item_id for it in items], vectors)
    return index

"""Qdrant-backed vector index (optional ``[vector]`` extra).

Implements the VectorIndex ABC (C-3). Defaults to Qdrant's embedded local mode
(``location=":memory:"``) so it runs with no Docker and no network; pass a
``url`` to target a running Qdrant server. If qdrant-client is not installed or
a server is unreachable, callers fall back to InMemoryFlatIndex (R-SEARCH-3).

String item ids are mapped to integer point ids (Qdrant requires int/UUID ids);
the original id is preserved in the point payload.
"""

from __future__ import annotations

from collections.abc import Sequence

from recolens.core.vector_index import VectorIndex


class QdrantIndex(VectorIndex):
    def __init__(
        self,
        dim: int,
        *,
        location: str = ":memory:",
        url: str | None = None,
        collection: str = "recolens",
    ) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        self._client = QdrantClient(url=url) if url else QdrantClient(location=location)
        self._collection = collection
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)
        self._client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self._ids: list[str] = []
        self._next = 0

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        points = []
        for id_, vec in zip(ids, vectors, strict=True):
            points.append(
                self._models.PointStruct(id=self._next, vector=list(vec), payload={"item_id": id_})
            )
            self._next += 1
        if points:
            self._client.upsert(self._collection, points=points)
            self._ids.extend(ids)

    def search(self, query: Sequence[float], k: int) -> list[tuple[str, float]]:
        res = self._client.query_points(
            self._collection, query=list(query), limit=k, with_payload=True
        ).points
        return [(p.payload["item_id"], float(p.score)) for p in res]

    def __len__(self) -> int:
        return len(self._ids)

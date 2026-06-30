"""Local real embeddings via sentence-transformers (optional ``[embed]`` extra).

Implements the same EmbeddingProvider ABC as DeterministicHashEmbed, so it is a
drop-in swap via ``EMBED_PROVIDER=local`` (C-3). Runs on CPU, offline after the
first download. No API key / no credit card (public Hugging Face model).

Supply-chain hardening (K-4 / supply-chain security gate):
- model **revision pinned** to a commit SHA (a moving tag could be repointed).
- **safetensors enforced** (``use_safetensors=True``) — never load pickle weights
  (pytorch_model.bin) which can execute arbitrary code on load.
- telemetry disabled.
See docs/adr/ADR-0008-embedding-model-adoption.md for the audit record.
"""

from __future__ import annotations

import os

from recolens.core.embedding import EmbeddingProvider

# Pinned model identity (verified 2026-06-30 via HF API).
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # commit SHA, MIT, safetensors
DEFAULT_DIM = 384


class LocalSentenceTransformerEmbed(EmbeddingProvider):
    """e5/BGE local embedder. e5 needs 'query:' / 'passage:' prefixes."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        *,
        batch_size: int = 32,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # Lazy import so the package imports without the [embed] extra installed.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model,
            revision=revision,
            model_kwargs={"use_safetensors": True},  # refuse pickle weights
        )
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        # method was renamed across sentence-transformers versions
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self._dim = int(get_dim())

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        prefixed = [prefix + t for t in texts]
        vecs = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in vecs]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Items are passages.
        return self._encode(texts, self._passage_prefix)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self._query_prefix)

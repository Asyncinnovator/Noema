"""Baseline retrieval systems to compare against noema.KnowledgeBase:
plain-chunk RAG, naive full-context stuffing, and a text-only prompt
compressor.

All are intentionally simple, dependency-light reference implementations
-- not a specific commercial product -- so the benchmark compares *shapes*
of approach (raw chunks vs. the whole corpus vs. compressed chunks vs.
canonical graph facts) rather than one library's tuning against another's.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from noema.compress import count_tokens

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_TOKENS = 60
DEFAULT_TOP_K = 5

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _chunk_document(text: str, chunk_tokens: int) -> list[str]:
    """Group consecutive sentences into chunks of up to `chunk_tokens`
    tokens each, the standard fixed-size chunking a plain-text RAG pipeline
    would use -- no sentence is split, chunks just stop growing once the
    next sentence would exceed the budget.
    """
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > chunk_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


class NaiveChunkRAG:
    """Fixed-size chunk + embed + top-k retrieval, the traditional RAG
    baseline: chunks are embedded whole and returned verbatim, with no
    deduplication or cross-document structure.
    """

    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._chunk_tokens = chunk_tokens
        self._embedding_model = embedding_model
        self._chunks: list[str] = []
        self._embeddings: np.ndarray | None = None

    def build(self, documents: list[str]) -> None:
        self._chunks = [
            chunk for document in documents for chunk in _chunk_document(document, self._chunk_tokens)
        ]
        model = _load_model(self._embedding_model)
        self._embeddings = model.encode(self._chunks, normalize_embeddings=True)

    def retrieve(self, query: str, max_tokens: int, top_k: int = DEFAULT_TOP_K) -> str:
        """Embed `query`, take the top_k most similar chunks, and
        concatenate them verbatim (in similarity order) until `max_tokens`
        would be exceeded.
        """
        if self._embeddings is None or not self._chunks:
            return ""

        model = _load_model(self._embedding_model)
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ query_vec
        ranked_indices = np.argsort(-scores)[:top_k]

        selected: list[str] = []
        used_tokens = 0
        for idx in ranked_indices:
            chunk = self._chunks[int(idx)]
            chunk_tokens = count_tokens(chunk)
            if selected and used_tokens + chunk_tokens > max_tokens:
                break
            selected.append(chunk)
            used_tokens += chunk_tokens
        return "\n".join(selected)


class FullContextBaseline:
    """Naive full-context stuffing: every query gets the entire corpus as
    context. Retrieval accuracy is the ceiling (nothing is left out) and
    token cost is the worst case, so it bounds the other systems.
    """

    def __init__(self) -> None:
        self._text = ""

    def build(self, documents: list[str]) -> None:
        self._text = "\n".join(documents)

    def retrieve(self, query: str, max_tokens: int | None = None) -> str:
        """Return the whole corpus; `query` and `max_tokens` are ignored."""
        return self._text


class TextOnlyCompressor:
    """A structure-agnostic prompt compressor: starts from the same
    retrieved chunks as NaiveChunkRAG, then compresses by keeping only the
    individual sentences most relevant to the query (ranked by embedding
    similarity) instead of whole chunks -- smaller than naive RAG, but
    still text in, text out, with no notion of entities or facts.
    """

    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        candidate_chunks: int = DEFAULT_TOP_K,
    ) -> None:
        self._chunk_rag = NaiveChunkRAG(chunk_tokens=chunk_tokens, embedding_model=embedding_model)
        self._embedding_model = embedding_model
        self._candidate_chunks = candidate_chunks

    def build(self, documents: list[str]) -> None:
        self._chunk_rag.build(documents)

    def retrieve(self, query: str, max_tokens: int) -> str:
        candidate_text = self._chunk_rag.retrieve(
            query, max_tokens=max_tokens * 4, top_k=self._candidate_chunks
        )
        sentences = _split_sentences(candidate_text)
        if not sentences:
            return ""

        model = _load_model(self._embedding_model)
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        sentence_vecs = model.encode(sentences, normalize_embeddings=True)
        scores = sentence_vecs @ query_vec
        ranked_indices = np.argsort(-scores)

        selected: list[str] = []
        used_tokens = 0
        for idx in ranked_indices:
            sentence = sentences[int(idx)]
            sentence_tokens = count_tokens(sentence)
            if selected and used_tokens + sentence_tokens > max_tokens:
                break
            selected.append(sentence)
            used_tokens += sentence_tokens
        return " ".join(selected)
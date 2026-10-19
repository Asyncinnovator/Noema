"""Tests for benchmarks/baselines.py: NaiveChunkRAG, FullContextBaseline and
TextOnlyCompressor.

Embeddings and the tokenizer are mocked for the same reason as elsewhere in
this suite (huggingface.co/openaipublic.blob.core.windows.net aren't
reachable in this sandbox). The fake embedding model uses a fixed
vocabulary rather than per-call dimensionality, matching
tests/unit/test_vector_search.py -- NaiveChunkRAG/TextOnlyCompressor embed
the corpus and the query in separate encode() calls, so the fake model must
return the same vector dimension regardless of what's in each call (a real
SentenceTransformer always does; a dimension that depends on batch content
would not).
"""

from __future__ import annotations

import importlib
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from baselines import FullContextBaseline, NaiveChunkRAG, TextOnlyCompressor
from baselines import _load_model as baselines_load_model

compress_module = importlib.import_module("noema.compress")


class _KeywordModel:
    """Deterministic keyword-overlap embedding against a fixed vocabulary,
    so query/corpus similarity is predictable and checkable.
    """

    VOCAB: ClassVar[list[str]] = ["ceo", "found", "headquarter", "robot", "other"]

    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            v = np.zeros(len(self.VOCAB))
            low = t.lower()
            for i, kw in enumerate(self.VOCAB):
                if kw in low:
                    v[i] = 1.0
            if not v.any():
                v[-1] = 1.0
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm else v)
        return np.array(vecs, dtype="float32")


class _WhitespaceEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture
def patched_pipeline():
    with (
        patch("baselines._load_model", return_value=_KeywordModel()),
        patch.object(compress_module, "_load_token_encoding", return_value=_WhitespaceEncoding()),
    ):
        yield


_DOC_FOUNDING = (
    "Nova Robotics was founded by Elena Marsh in 2011. "
    "Nova Robotics is headquartered in Pittsburgh."
)
_DOC_LEADERSHIP = "Priya Anand is the CEO of Nova Robotics. Nova Robotics builds warehouse robots."


class TestChunkDocument:
    def test_sentences_stay_whole_never_split_mid_sentence(self, patched_pipeline):
        rag = NaiveChunkRAG(chunk_tokens=3)
        rag.build([_DOC_FOUNDING])
        for chunk in rag._chunks:
            assert chunk.strip().endswith(".")

    def test_small_chunk_budget_produces_multiple_chunks(self, patched_pipeline):
        rag = NaiveChunkRAG(chunk_tokens=3)
        rag.build([_DOC_FOUNDING])
        assert len(rag._chunks) > 1

    def test_large_chunk_budget_keeps_whole_document_as_one_chunk(self, patched_pipeline):
        rag = NaiveChunkRAG(chunk_tokens=1000)
        rag.build([_DOC_FOUNDING])
        assert len(rag._chunks) == 1

    def test_single_sentence_exceeding_budget_is_kept_whole_anyway(self, patched_pipeline):
        # A chunk's budget check only fires when there's already content in
        # the current chunk -- a lone oversized sentence must still become
        # its own chunk rather than being dropped or split.
        rag = NaiveChunkRAG(chunk_tokens=1)
        rag.build([_DOC_FOUNDING])
        assert "".join(rag._chunks).replace(" ", "") != ""


class TestNaiveChunkRAG:
    def test_build_produces_one_chunk_per_short_document(self, patched_pipeline):
        rag = NaiveChunkRAG(chunk_tokens=20)
        rag.build([_DOC_FOUNDING, _DOC_LEADERSHIP])
        assert len(rag._chunks) == 2

    def test_retrieve_ranks_the_relevant_chunk_first(self, patched_pipeline):
        rag = NaiveChunkRAG(chunk_tokens=20)
        rag.build([_DOC_FOUNDING, _DOC_LEADERSHIP])
        result = rag.retrieve("Who is the CEO?", max_tokens=50, top_k=1)
        assert result == _DOC_LEADERSHIP

    def test_retrieve_respects_top_k(self, patched_pipeline):
        rag = NaiveChunkRAG(chunk_tokens=20)
        rag.build([_DOC_FOUNDING, _DOC_LEADERSHIP])
        result = rag.retrieve("Nova Robotics", max_tokens=1000, top_k=1)
        assert result.count("\n") == 0

    def test_retrieve_stops_once_max_tokens_would_be_exceeded(self, patched_pipeline):
        rag = NaiveChunkRAG(chunk_tokens=20)
        rag.build([_DOC_FOUNDING, _DOC_LEADERSHIP])
        result = rag.retrieve("Nova Robotics", max_tokens=1, top_k=2)
        # The first selected chunk is always kept even if it alone exceeds
        # max_tokens (an empty result would be worse than one over-budget
        # chunk); a second chunk is only added if it still fits.
        assert result.count("\n") == 0
        assert result != ""

    def test_retrieve_before_build_returns_empty_string(self, patched_pipeline):
        rag = NaiveChunkRAG()
        assert rag.retrieve("anything", max_tokens=50) == ""

    def test_build_with_no_documents_does_not_crash(self, patched_pipeline):
        rag = NaiveChunkRAG()
        rag.build([])
        assert rag.retrieve("anything", max_tokens=50) == ""


class TestFullContextBaseline:
    def test_returns_every_document_regardless_of_query(self):
        fc = FullContextBaseline()
        fc.build([_DOC_FOUNDING, _DOC_LEADERSHIP])
        for query in ("Who is the CEO?", "unrelated"):
            result = fc.retrieve(query, max_tokens=1)
            assert _DOC_FOUNDING in result
            assert _DOC_LEADERSHIP in result

    def test_before_build_returns_empty_string(self):
        assert FullContextBaseline().retrieve("anything") == ""

    def test_rebuild_replaces_previous_corpus(self):
        fc = FullContextBaseline()
        fc.build([_DOC_FOUNDING])
        fc.build([_DOC_LEADERSHIP])
        assert fc.retrieve("x") == _DOC_LEADERSHIP


class TestTextOnlyCompressor:
    def test_retrieve_surfaces_the_relevant_sentence(self, patched_pipeline):
        tc = TextOnlyCompressor(chunk_tokens=20)
        tc.build([_DOC_FOUNDING, _DOC_LEADERSHIP])
        result = tc.retrieve("Who is the CEO?", max_tokens=50)
        assert "CEO" in result

    def test_retrieve_is_sentence_level_not_whole_chunks(self, patched_pipeline):
        # With a tight budget, the compressor should be able to return a
        # single sentence rather than being forced to include a whole
        # multi-sentence chunk the way NaiveChunkRAG would.
        tc = TextOnlyCompressor(chunk_tokens=20)
        tc.build([_DOC_FOUNDING, _DOC_LEADERSHIP])
        result = tc.retrieve("Who is the CEO?", max_tokens=8)
        assert "headquartered" not in result

    def test_build_with_no_documents_does_not_crash(self, patched_pipeline):
        tc = TextOnlyCompressor()
        tc.build([])
        assert tc.retrieve("anything", max_tokens=50) == ""


class TestLoadModelCaching:
    def test_same_model_name_returns_cached_instance(self):
        baselines_load_model.cache_clear()
        with patch("baselines.SentenceTransformer") as mock_cls:
            baselines_load_model("fake-model")
            baselines_load_model("fake-model")
            assert mock_cls.call_count == 1
        baselines_load_model.cache_clear()
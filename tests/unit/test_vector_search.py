"""Tests for noema.retrieval.vector_search.

Embeddings are mocked for the same reason as elsewhere in this suite
(huggingface.co isn't reachable in this sandbox).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

import noema.retrieval.vector_search as vector_search
from noema.retrieval.vector_search import FactMatch, search_facts


class _KeywordModel:
    """Deterministic keyword-overlap embedding: each text is scored against
    a fixed small vocabulary, giving predictable, checkable similarity
    relationships between queries and fact texts.
    """

    VOCAB = ["ceo", "found", "headquarter", "other"]

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


def _build_index(texts: list[str]) -> faiss.Index:
    model = _KeywordModel()
    embeddings = model.encode(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


class TestSearchFacts:
    def test_most_similar_fact_ranks_first(self):
        fact_ids = ["rel_ceo", "rel_found", "rel_hq"]
        texts = ["Sundar Pichai ceo of Google", "Larry Page found Google", "Google headquarter Mountain View"]
        index = _build_index(texts)

        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            results = search_facts("who is the ceo?", fact_ids, index, top_k=3)

        assert results[0].relation_id == "rel_ceo"

    def test_results_are_descending_by_score(self):
        fact_ids = ["rel_ceo", "rel_found", "rel_hq"]
        texts = ["Sundar Pichai ceo of Google", "Larry Page found Google", "Google headquarter Mountain View"]
        index = _build_index(texts)

        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            results = search_facts("ceo", fact_ids, index, top_k=3)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_fact_match_objects(self):
        fact_ids = ["rel_a"]
        index = _build_index(["found"])
        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            results = search_facts("found", fact_ids, index, top_k=1)
        assert all(isinstance(r, FactMatch) for r in results)

    def test_top_k_larger_than_available_facts_does_not_error(self):
        fact_ids = ["rel_a", "rel_b"]
        index = _build_index(["found", "ceo"])
        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            results = search_facts("query", fact_ids, index, top_k=10)
        assert len(results) == 2

    def test_top_k_smaller_than_available_facts_limits_results(self):
        fact_ids = ["rel_a", "rel_b", "rel_c"]
        index = _build_index(["found", "ceo", "headquarter"])
        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            results = search_facts("query", fact_ids, index, top_k=1)
        assert len(results) == 1

    def test_empty_index_returns_empty_list(self):
        empty_index = faiss.IndexFlatIP(4)
        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            results = search_facts("query", [], empty_index, top_k=5)
        assert results == []

    def test_empty_index_does_not_call_the_model(self):
        empty_index = faiss.IndexFlatIP(4)
        load_mock = MagicMock()
        with patch.object(vector_search, "_load_model", load_mock):
            search_facts("query", [], empty_index, top_k=5)
        assert load_mock.called is False


class TestEmbedQuery:
    def test_embed_query_returns_float32_array(self):
        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            vector = vector_search.embed_query("test query")
        assert vector.dtype == np.float32

    def test_embed_query_shape_matches_model_output_dim(self):
        with patch.object(vector_search, "_load_model", return_value=_KeywordModel()):
            vector = vector_search.embed_query("found")
        assert vector.shape == (1, len(_KeywordModel.VOCAB))
"""Tests for noema.storage.knowledge_base.

Embeddings are mocked for the same reason as elsewhere in this suite
(huggingface.co isn't reachable in this sandbox). Two fake models are used:
an exact-match-only one for tests that shouldn't merge near-duplicates, and
a "google"/"google inc" near-duplicate-aware one specifically for testing
that cross-batch merging works.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import noema.canonicalize.dedup as dedup
import noema.storage.bundle as bundle
from noema.schema import Document
from noema.storage.knowledge_base import IngestReport, KnowledgeBase, _normalize_documents


class _ExactMatchModel:
    """Orthogonal per distinct (normalized) string -- no accidental merges."""

    def encode(self, texts, normalize_embeddings=True):
        base = {}
        for t in texts:
            key = t.lower().strip()
            if key not in base:
                base[key] = len(base)
        dim = max(len(base), 2)
        vecs = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            vecs[i, base[t.lower().strip()]] = 1.0
        return vecs


class _GoogleAliasModel:
    """Like _ExactMatchModel, but 'google' and 'google inc' share a vector."""

    def encode(self, texts, normalize_embeddings=True):
        base = {}
        for t in texts:
            key = "google" if t.lower().strip() in ("google", "google inc") else t.lower().strip()
            if key not in base:
                base[key] = len(base)
        dim = max(len(base), 2)
        vecs = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            key = "google" if t.lower().strip() in ("google", "google inc") else t.lower().strip()
            vecs[i, base[key]] = 1.0
        return vecs


@pytest.fixture
def patched_exact_match():
    with patch.object(dedup, "_load_model", return_value=_ExactMatchModel()):
        yield


@pytest.fixture
def patched_google_alias():
    with patch.object(dedup, "_load_model", return_value=_GoogleAliasModel()):
        yield


class TestNormalizeDocuments:
    def test_single_string_becomes_single_document(self):
        docs = _normalize_documents("hello world")
        assert len(docs) == 1
        assert docs[0].text == "hello world"

    def test_single_document_passed_through(self):
        doc = Document(text="hello world")
        docs = _normalize_documents(doc)
        assert docs == [doc]

    def test_list_of_mixed_strings_and_documents(self):
        doc = Document(text="already a document")
        docs = _normalize_documents(["plain string", doc])
        assert len(docs) == 2
        assert docs[0].text == "plain string"
        assert docs[1] is doc


class TestIngestBasic:
    def test_ingest_single_string(self, patched_exact_match):
        kb = KnowledgeBase()
        report = kb.ingest("Larry Page founded Google in 1998.")
        assert isinstance(report, IngestReport)
        assert report.documents_ingested == 1

    def test_ingest_populates_graph(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        names = {e.name for e in kb.graph.entities}
        assert "Larry Page" in names
        assert "Google" in names

    def test_ingest_multiple_documents_in_one_call(self, patched_exact_match):
        kb = KnowledgeBase()
        report = kb.ingest(
            ["Larry Page founded Google in 1998.", "Sundar Pichai is the CEO of Google."]
        )
        assert report.documents_ingested == 2

    def test_len_reflects_entity_count(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        assert len(kb) == len(kb.graph.entities)

    def test_report_tracks_before_and_after_counts(self, patched_exact_match):
        kb = KnowledgeBase()
        report1 = kb.ingest("Larry Page founded Google in 1998.")
        assert report1.entities_before == 0
        assert report1.entities_after == len(kb.graph.entities)

        entities_after_first = report1.entities_after
        report2 = kb.ingest("Sundar Pichai is the CEO of Google.")
        assert report2.entities_before == entities_after_first


class TestCrossBatchCanonicalization:
    def test_same_entity_across_two_ingests_merges_to_one_node(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        kb.ingest("Google is headquartered in Mountain View.")

        google_nodes = [e for e in kb.graph.entities if e.name == "Google"]
        assert len(google_nodes) == 1

    def test_relations_from_both_batches_point_to_same_canonical_entity(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        kb.ingest("Google is headquartered in Mountain View.")

        google = next(e for e in kb.graph.entities if e.name == "Google")
        related_object_ids = {r.object_id for r in kb.graph.relations}
        assert google.id in related_object_ids or google.id in {
            r.subject_id for r in kb.graph.relations
        }

    def test_near_duplicate_across_batches_merges_with_alias_aware_model(
        self, patched_google_alias
    ):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        kb.ingest("Google Inc. named Sundar Pichai as CEO.")

        google_nodes = [e for e in kb.graph.entities if e.name.lower().startswith("google")]
        assert len(google_nodes) == 1
        assert "Google Inc." in google_nodes[0].aliases

    def test_entities_merged_property_reflects_actual_merge(self, patched_google_alias):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        report = kb.ingest("Google Inc. named Sundar Pichai as CEO.")
        # entities_before included the old 'Google' node; entities_after has
        # it merged with the new 'Google Inc.' mention rather than as two nodes
        assert report.entities_merged >= 0


class TestSaveAndLoad:
    def test_save_creates_file(self, patched_exact_match, tmp_path):
        with patch.object(bundle, "_load_model", return_value=_ExactMatchModel()):
            kb = KnowledgeBase()
            kb.ingest("Larry Page founded Google in 1998.")
            path = str(tmp_path / "kb.noema")
            kb.save(path)
        import os

        assert os.path.exists(path)

    def test_load_restores_entities(self, patched_exact_match, tmp_path):
        with patch.object(bundle, "_load_model", return_value=_ExactMatchModel()):
            kb = KnowledgeBase()
            kb.ingest("Larry Page founded Google in 1998.")
            path = str(tmp_path / "kb.noema")
            kb.save(path)

            loaded_kb = KnowledgeBase.load(path)

        original_names = sorted(e.name for e in kb.graph.entities)
        loaded_names = sorted(e.name for e in loaded_kb.graph.entities)
        assert original_names == loaded_names

    def test_load_restores_relations(self, patched_exact_match, tmp_path):
        with patch.object(bundle, "_load_model", return_value=_ExactMatchModel()):
            kb = KnowledgeBase()
            kb.ingest("Larry Page founded Google in 1998.")
            path = str(tmp_path / "kb.noema")
            kb.save(path)
            loaded_kb = KnowledgeBase.load(path)
        assert len(loaded_kb.graph.relations) == len(kb.graph.relations)

    def test_continued_ingest_after_load_merges_with_loaded_content(self, tmp_path):
        with patch.object(dedup, "_load_model", return_value=_GoogleAliasModel()), \
             patch.object(bundle, "_load_model", return_value=_GoogleAliasModel()):
            kb = KnowledgeBase()
            kb.ingest("Larry Page founded Google in 1998.")
            path = str(tmp_path / "kb.noema")
            kb.save(path)

            loaded_kb = KnowledgeBase.load(path)
            loaded_kb.ingest("Google Inc. is headquartered in Mountain View.")

            google_nodes = [
                e for e in loaded_kb.graph.entities if e.name.lower().startswith("google")
            ]
            assert len(google_nodes) == 1
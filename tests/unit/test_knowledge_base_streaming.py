"""Tests for KnowledgeBase.ingest_stream() and the sync_to_backend() /
load_from_backend() persistent-backend methods.

Embeddings are mocked for the same reason as test_knowledge_base.py
(huggingface.co isn't reachable in this sandbox).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from noema.canonicalize import dedup
from noema.schema import Document
from noema.storage import bundle
from noema.storage.backends import GraphBackend
from noema.storage.knowledge_base import KnowledgeBase


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


@pytest.fixture
def patched_exact_match():
    with patch.object(dedup, "_load_model", return_value=_ExactMatchModel()), patch.object(
        bundle, "_load_model", return_value=_ExactMatchModel()
    ):
        yield


_DOCS = [
    "Larry Page founded Google in 1998.",
    "Sergey Brin co-founded Google with Larry Page.",
    "Sundar Pichai is the CEO of Google.",
    "Google is headquartered in Mountain View.",
]


class TestIngestStreamBasics:
    def test_consumes_a_plain_list(self, patched_exact_match):
        kb = KnowledgeBase()
        report = kb.ingest_stream(_DOCS, batch_size=2)
        assert report.documents_ingested == 4

    def test_consumes_a_generator_without_materializing_it_first(self, patched_exact_match):
        def doc_stream():
            yield from _DOCS

        kb = KnowledgeBase()
        report = kb.ingest_stream(doc_stream(), batch_size=2)
        assert report.documents_ingested == 4

    def test_mixed_strings_and_documents(self, patched_exact_match):
        mixed = [_DOCS[0], Document(text=_DOCS[1])]
        kb = KnowledgeBase()
        report = kb.ingest_stream(mixed, batch_size=5)
        assert report.documents_ingested == 2

    def test_matches_result_of_calling_ingest_once(self, patched_exact_match):
        kb_batched = KnowledgeBase()
        kb_batched.ingest_stream(_DOCS, batch_size=2)

        kb_single = KnowledgeBase()
        kb_single.ingest(_DOCS)

        names_batched = {e.name for e in kb_batched.graph.entities}
        names_single = {e.name for e in kb_single.graph.entities}
        assert names_batched == names_single

    def test_empty_input_produces_zero_documents(self, patched_exact_match):
        kb = KnowledgeBase()
        report = kb.ingest_stream([])
        assert report.documents_ingested == 0
        assert len(kb) == 0

    def test_leftover_partial_batch_is_still_ingested(self, patched_exact_match):
        # 4 docs at batch_size=3 -> one full batch of 3, one partial batch of 1.
        kb = KnowledgeBase()
        report = kb.ingest_stream(_DOCS, batch_size=3)
        assert report.documents_ingested == 4


class TestIngestStreamBatching:
    def test_batch_count_matches_batch_size(self, patched_exact_match):
        seen_batches = []
        kb = KnowledgeBase()
        kb.ingest_stream(_DOCS, batch_size=2, on_batch=lambda i, r: seen_batches.append(i))
        assert seen_batches == [1, 2]

    def test_on_batch_receives_per_batch_report(self, patched_exact_match):
        reports = []
        kb = KnowledgeBase()
        kb.ingest_stream(_DOCS, batch_size=2, on_batch=lambda i, r: reports.append(r))
        assert all(r.documents_ingested == 2 for r in reports)

    def test_canonicalization_runs_once_per_batch_not_once_per_document(self, patched_exact_match):
        call_count = 0

        with patch.object(dedup, "_load_model") as mocked:
            mocked.side_effect = lambda *a, **k: _ExactMatchModel()
            kb = KnowledgeBase()
            kb.ingest_stream(_DOCS, batch_size=2)
            call_count = mocked.call_count

        # 4 docs at batch_size=2 -> 2 batches -> 2 canonicalize() calls,
        # not 4 (one per document).
        assert call_count == 2


class TestIngestStreamCheckpointing:
    def test_checkpoint_file_is_written_after_each_batch(self, patched_exact_match, tmp_path):
        checkpoint = str(tmp_path / "checkpoint.noema")
        kb = KnowledgeBase()
        kb.ingest_stream(_DOCS, batch_size=2, checkpoint_path=checkpoint)
        assert (tmp_path / "checkpoint.noema").exists()

    def test_checkpoint_reloads_to_the_same_entity_set(self, patched_exact_match, tmp_path):
        checkpoint = str(tmp_path / "checkpoint.noema")
        kb = KnowledgeBase()
        kb.ingest_stream(_DOCS, batch_size=2, checkpoint_path=checkpoint)

        reloaded = KnowledgeBase.load(checkpoint)
        assert len(reloaded.graph.entities) == len(kb.graph.entities)

    def test_no_checkpoint_path_means_no_file_written(self, patched_exact_match, tmp_path):
        kb = KnowledgeBase()
        kb.ingest_stream(_DOCS, batch_size=2)
        assert list(tmp_path.iterdir()) == []


class TestIngestStreamReport:
    def test_report_totals_cover_the_whole_stream_not_just_last_batch(self, patched_exact_match):
        kb = KnowledgeBase()
        report = kb.ingest_stream(_DOCS, batch_size=2)
        assert report.entities_parsed > 0
        assert report.entities_after == len(kb.graph.entities)

    def test_entities_before_reflects_state_prior_to_any_batch(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        entities_before_stream = len(kb.graph.entities)

        report = kb.ingest_stream(["Sundar Pichai is the CEO of Google."], batch_size=5)

        assert report.entities_before == entities_before_stream


class _DictBackend(GraphBackend):
    """Minimal in-memory GraphBackend for testing KnowledgeBase's sync
    methods without depending on rdflib/neo4j.
    """

    def __init__(self) -> None:
        self._entities = {}
        self._relations = {}
        self._events = {}

    def upsert_entities(self, entities):
        for e in entities:
            self._entities[e.id] = e

    def upsert_relations(self, relations):
        for r in relations:
            self._relations[r.id] = r

    def upsert_events(self, events):
        for e in events:
            self._events[e.id] = e

    def get_entity(self, entity_id):
        return self._entities.get(entity_id)

    def all_entities(self):
        yield from self._entities.values()

    def all_relations(self):
        yield from self._relations.values()

    def all_events(self):
        yield from self._events.values()

    def entity_count(self):
        return len(self._entities)

    def clear(self):
        self._entities.clear()
        self._relations.clear()
        self._events.clear()


class TestSyncToBackend:
    def test_pushes_full_graph_to_backend(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest(_DOCS)
        backend = _DictBackend()

        kb.sync_to_backend(backend)

        assert backend.entity_count() == len(kb.graph.entities)

    def test_overwrites_whatever_the_backend_held_before(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        backend = _DictBackend()
        backend._entities["stray_id"] = "stray entity from a previous sync"

        kb.sync_to_backend(backend)

        assert "stray_id" not in backend._entities


class TestLoadFromBackend:
    def test_builds_a_knowledge_base_from_backend_contents(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest(_DOCS)
        backend = _DictBackend()
        kb.sync_to_backend(backend)

        loaded = KnowledgeBase.load_from_backend(backend)

        assert len(loaded.graph.entities) == len(kb.graph.entities)
        assert len(loaded.graph.relations) == len(kb.graph.relations)

    def test_loaded_knowledge_base_can_ingest_more_documents(self, patched_exact_match):
        kb = KnowledgeBase()
        kb.ingest("Larry Page founded Google in 1998.")
        backend = _DictBackend()
        kb.sync_to_backend(backend)

        loaded = KnowledgeBase.load_from_backend(backend)
        report = loaded.ingest("Sundar Pichai is the CEO of Google.")

        assert report.documents_ingested == 1
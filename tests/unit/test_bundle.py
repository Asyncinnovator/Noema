"""Tests for noema.storage.bundle.

The real embedding model (huggingface.co) isn't reachable in this sandbox,
so _load_model is mocked. These tests verify bundle pack/unpack correctness
and the fact-embedding wiring, not real embedding quality.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from noema.graph.builder import SemanticGraph
from noema.schema import Entity, EntityType, Event, Relation
from noema.storage import bundle


class _FakeEmbeddingModel:
    def __init__(self, dim: int = 6) -> None:
        self._dim = dim

    def encode(self, texts, normalize_embeddings=True):
        rng = np.random.default_rng(0)
        vecs = rng.normal(size=(len(texts), self._dim)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def _build_graph() -> SemanticGraph:
    graph = SemanticGraph()
    page = Entity(name="Larry Page", type=EntityType.PERSON)
    google = Entity(name="Google", type=EntityType.ORG)
    pichai = Entity(name="Sundar Pichai", type=EntityType.PERSON)
    for e in (page, google, pichai):
        graph.add_entity(e)
    graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))
    graph.add_relation(Relation(subject_id=pichai.id, predicate="ceo_of", object_id=google.id))
    graph.add_event(Event(label="ipo", participant_ids=[google.id], time="2004-08-19"))
    return graph


@pytest.fixture
def tmp_bundle_path(tmp_path):
    return str(tmp_path / "test.noema")


class TestGraphDictRoundTrip:
    def test_entities_survive_round_trip(self):
        graph = _build_graph()
        restored = bundle.graph_from_dict(bundle.graph_to_dict(graph))
        assert sorted(e.name for e in restored.entities) == sorted(e.name for e in graph.entities)

    def test_relations_survive_round_trip(self):
        graph = _build_graph()
        restored = bundle.graph_from_dict(bundle.graph_to_dict(graph))
        assert len(restored.relations) == len(graph.relations)
        predicates = {r.predicate for r in restored.relations}
        assert predicates == {"found", "ceo_of"}

    def test_events_survive_round_trip(self):
        graph = _build_graph()
        restored = bundle.graph_from_dict(bundle.graph_to_dict(graph))
        assert len(restored.events) == 1
        assert restored.events[0].label == "ipo"
        assert restored.events[0].time == "2004-08-19"

    def test_entity_ids_are_preserved(self):
        graph = _build_graph()
        original_ids = sorted(e.id for e in graph.entities)
        restored = bundle.graph_from_dict(bundle.graph_to_dict(graph))
        restored_ids = sorted(e.id for e in restored.entities)
        assert original_ids == restored_ids


class TestBuildFactIndex:
    def test_fact_ids_match_relation_ids_in_order(self):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            fact_ids, _embeddings, _index = bundle.build_fact_index(graph)
        assert fact_ids == [r.id for r in graph.relations]

    def test_embeddings_row_count_matches_relation_count(self):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel(dim=6)):
            _fact_ids, embeddings, _index = bundle.build_fact_index(graph)
        assert embeddings.shape == (2, 6)

    def test_faiss_index_contains_all_facts(self):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            _fact_ids, _embeddings, index = bundle.build_fact_index(graph)
        assert index.ntotal == 2

    def test_zero_relations_does_not_call_the_model(self):
        graph = SemanticGraph()
        graph.add_entity(Entity(name="Lonely", type=EntityType.OTHER))
        load_mock = MagicMock()
        with patch.object(bundle, "_load_model", load_mock):
            fact_ids, embeddings, index = bundle.build_fact_index(graph)
        assert load_mock.called is False
        assert fact_ids == []
        assert embeddings.shape == (0, bundle.DEFAULT_EMBEDDING_DIM)
        assert index.ntotal == 0


class TestSaveAndLoadBundle:
    def test_entities_survive_save_and_load(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert sorted(e.name for e in loaded.graph.entities) == sorted(
            e.name for e in graph.entities
        )

    def test_relations_survive_save_and_load(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert len(loaded.graph.relations) == len(graph.relations)

    def test_fact_ids_match_reloaded_relation_ids(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert set(loaded.fact_ids) == {r.id for r in loaded.graph.relations}

    def test_embeddings_shape_preserved(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel(dim=6)):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert loaded.embeddings.shape == (2, 6)

    def test_faiss_index_ntotal_preserved(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert loaded.index.ntotal == 2

    def test_manifest_counts_are_correct(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert loaded.manifest["entity_count"] == 3
        assert loaded.manifest["relation_count"] == 2
        assert loaded.manifest["event_count"] == 1
        assert loaded.manifest["fact_count"] == 2

    def test_manifest_schema_version_present(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert loaded.manifest["schema_version"] == bundle.SCHEMA_VERSION

    def test_manifest_records_spacy_model_when_provided(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph, spacy_model="en_core_web_md")
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert loaded.manifest["spacy_model"] == "en_core_web_md"

    def test_manifest_spacy_model_is_none_when_not_provided(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert loaded.manifest["spacy_model"] is None

    def test_source_document_ids_collected_from_mentions(self, tmp_bundle_path):
        graph = SemanticGraph()
        entity = Entity(name="Google", type=EntityType.ORG)
        from noema.schema import SourceSpan

        entity.add_mention(SourceSpan(doc_id="doc_abc", start_char=0, end_char=6, text="Google"))
        graph.add_entity(entity)
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert loaded.manifest["source_document_ids"] == ["doc_abc"]

    def test_bundle_file_is_created_on_disk(self, tmp_bundle_path):
        graph = _build_graph()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
            bundle.save_bundle(tmp_bundle_path, graph)
        assert os.path.exists(tmp_bundle_path)

    def test_empty_graph_round_trips_without_calling_model(self, tmp_bundle_path):
        graph = SemanticGraph()
        graph.add_entity(Entity(name="Lonely", type=EntityType.OTHER))
        load_mock = MagicMock()
        with patch.object(bundle, "_load_model", load_mock):
            bundle.save_bundle(tmp_bundle_path, graph)
            loaded = bundle.load_bundle(tmp_bundle_path)
        assert load_mock.called is False
        assert len(loaded.graph.entities) == 1
        assert loaded.index.ntotal == 0
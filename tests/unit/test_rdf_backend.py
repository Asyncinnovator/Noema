"""Tests for noema.storage.rdf_backend.RDFBackend.

rdflib runs entirely locally (no network/model download involved), so
these tests exercise the real triple store rather than mocking it.
"""

from __future__ import annotations

import pytest

rdflib = pytest.importorskip("rdflib")

from noema.schema import Entity, EntityType, Event, Relation
from noema.storage.backends import read_graph, write_graph
from noema.storage.rdf_backend import RDFBackend


@pytest.fixture
def backend():
    return RDFBackend()


class TestUpsertEntities:
    def test_entity_round_trips(self, backend):
        entity = Entity(name="Google", type=EntityType.ORG, aliases=["Google Inc."])
        backend.upsert_entities([entity])

        fetched = backend.get_entity(entity.id)

        assert fetched is not None
        assert fetched.name == "Google"
        assert fetched.type == EntityType.ORG
        assert fetched.aliases == ["Google Inc."]

    def test_entity_count(self, backend):
        backend.upsert_entities(
            [
                Entity(name="Larry Page", type=EntityType.PERSON),
                Entity(name="Google", type=EntityType.ORG),
            ]
        )
        assert backend.entity_count() == 2

    def test_reupserting_same_id_replaces_rather_than_duplicates(self, backend):
        entity = Entity(id="ent_fixed", name="Google", type=EntityType.ORG)
        backend.upsert_entities([entity])
        updated = Entity(id="ent_fixed", name="Google LLC", type=EntityType.ORG)
        backend.upsert_entities([updated])

        assert backend.entity_count() == 1
        assert backend.get_entity("ent_fixed").name == "Google LLC"

    def test_unknown_entity_returns_none(self, backend):
        assert backend.get_entity("ent_missing") is None


class TestUpsertRelations:
    def test_relation_round_trips(self, backend):
        page = Entity(name="Larry Page", type=EntityType.PERSON)
        google = Entity(name="Google", type=EntityType.ORG)
        backend.upsert_entities([page, google])
        relation = Relation(subject_id=page.id, predicate="found", object_id=google.id)
        backend.upsert_relations([relation])

        relations = list(backend.all_relations())

        assert len(relations) == 1
        assert relations[0].predicate == "found"
        assert relations[0].subject_id == page.id
        assert relations[0].object_id == google.id


class TestUpsertEvents:
    def test_event_round_trips(self, backend):
        google = Entity(name="Google", type=EntityType.ORG)
        backend.upsert_entities([google])
        event = Event(label="ipo", participant_ids=[google.id], time="2004-08-19")
        backend.upsert_events([event])

        events = list(backend.all_events())

        assert len(events) == 1
        assert events[0].label == "ipo"
        assert events[0].time == "2004-08-19"
        assert events[0].participant_ids == [google.id]


class TestClear:
    def test_clear_removes_everything(self, backend):
        backend.upsert_entities([Entity(name="Google", type=EntityType.ORG)])
        backend.clear()
        assert backend.entity_count() == 0


class TestWriteReadGraphIntegration:
    def test_full_graph_round_trips_through_rdf_backend(self):
        from noema.graph.builder import SemanticGraph

        graph = SemanticGraph()
        page = Entity(name="Larry Page", type=EntityType.PERSON)
        google = Entity(name="Google", type=EntityType.ORG)
        graph.add_entity(page)
        graph.add_entity(google)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))

        backend = RDFBackend()
        write_graph(backend, graph)
        restored = read_graph(backend)

        assert len(restored.entities) == 2
        assert len(restored.relations) == 1
        assert restored.relations[0].predicate == "found"


class TestFilePersistence:
    def test_save_and_reload_from_path(self, tmp_path):
        path = str(tmp_path / "kb.ttl")
        backend = RDFBackend(path=path)
        backend.upsert_entities([Entity(name="Google", type=EntityType.ORG)])
        backend.save()

        reloaded = RDFBackend(path=path)
        assert reloaded.entity_count() == 1

    def test_flush_persists_when_constructed_with_a_path(self, tmp_path):
        path = str(tmp_path / "kb.ttl")
        backend = RDFBackend(path=path)
        backend.upsert_entities([Entity(name="Google", type=EntityType.ORG)])
        backend.flush()

        reloaded = RDFBackend(path=path)
        assert reloaded.entity_count() == 1

    def test_missing_file_on_construction_starts_empty_instead_of_raising(self, tmp_path):
        path = str(tmp_path / "does_not_exist.ttl")
        backend = RDFBackend(path=path)
        assert backend.entity_count() == 0

    def test_save_without_any_path_raises(self):
        backend = RDFBackend()
        with pytest.raises(ValueError):
            backend.save()
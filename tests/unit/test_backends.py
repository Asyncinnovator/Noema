"""Tests for noema.storage.backends: the GraphBackend interface and the
write_graph/read_graph helpers, using an in-memory fake backend so these
tests don't depend on any real external store.
"""

from __future__ import annotations

from collections.abc import Iterator

from noema.graph.builder import SemanticGraph
from noema.schema import Entity, EntityType, Event, Relation
from noema.storage.backends import GraphBackend, read_graph, write_graph


class InMemoryBackend(GraphBackend):
    """Minimal GraphBackend implementation backed by plain dicts, used only
    to test the shared write_graph/read_graph logic in isolation from any
    real database.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}
        self.events: dict[str, Event] = {}
        self.flush_count = 0
        self.close_count = 0

    def upsert_entities(self, entities: list[Entity]) -> None:
        for e in entities:
            self.entities[e.id] = e

    def upsert_relations(self, relations: list[Relation]) -> None:
        for r in relations:
            self.relations[r.id] = r

    def upsert_events(self, events: list[Event]) -> None:
        for e in events:
            self.events[e.id] = e

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def all_entities(self) -> Iterator[Entity]:
        yield from self.entities.values()

    def all_relations(self) -> Iterator[Relation]:
        yield from self.relations.values()

    def all_events(self) -> Iterator[Event]:
        yield from self.events.values()

    def entity_count(self) -> int:
        return len(self.entities)

    def clear(self) -> None:
        self.entities.clear()
        self.relations.clear()
        self.events.clear()

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


def _build_graph() -> SemanticGraph:
    graph = SemanticGraph()
    page = Entity(name="Larry Page", type=EntityType.PERSON)
    google = Entity(name="Google", type=EntityType.ORG)
    for e in (page, google):
        graph.add_entity(e)
    graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))
    graph.add_event(Event(label="ipo", participant_ids=[google.id], time="2004-08-19"))
    return graph


class TestWriteGraph:
    def test_writes_all_entities_relations_events(self):
        graph = _build_graph()
        backend = InMemoryBackend()
        write_graph(backend, graph)

        assert len(backend.entities) == 2
        assert len(backend.relations) == 1
        assert len(backend.events) == 1

    def test_flush_is_called_after_writing(self):
        graph = _build_graph()
        backend = InMemoryBackend()
        write_graph(backend, graph)
        assert backend.flush_count == 1

    def test_clear_first_wipes_existing_contents_by_default(self):
        graph = _build_graph()
        backend = InMemoryBackend()
        stray = Entity(name="Stray Entity", type=EntityType.OTHER)
        backend.upsert_entities([stray])

        write_graph(backend, graph)

        assert stray.id not in backend.entities

    def test_clear_first_false_merges_instead_of_wiping(self):
        graph = _build_graph()
        backend = InMemoryBackend()
        stray = Entity(name="Stray Entity", type=EntityType.OTHER)
        backend.upsert_entities([stray])

        write_graph(backend, graph, clear_first=False)

        assert stray.id in backend.entities
        assert len(backend.entities) == 3


class TestReadGraph:
    def test_round_trip_preserves_entities_and_relations(self):
        graph = _build_graph()
        backend = InMemoryBackend()
        write_graph(backend, graph)

        restored = read_graph(backend)

        assert sorted(e.name for e in restored.entities) == sorted(e.name for e in graph.entities)
        assert len(restored.relations) == len(graph.relations)
        assert restored.relations[0].predicate == "found"

    def test_round_trip_preserves_events(self):
        graph = _build_graph()
        backend = InMemoryBackend()
        write_graph(backend, graph)

        restored = read_graph(backend)

        assert len(restored.events) == 1
        assert restored.events[0].label == "ipo"

    def test_empty_backend_produces_empty_graph(self):
        backend = InMemoryBackend()
        restored = read_graph(backend)
        assert len(restored) == 0


class TestGraphBackendContextManager:
    def test_enter_returns_self(self):
        backend = InMemoryBackend()
        with backend as ctx:
            assert ctx is backend

    def test_exit_calls_close(self):
        backend = InMemoryBackend()
        with backend:
            pass
        assert backend.close_count == 1

    def test_default_flush_and_close_are_no_ops(self):
        # Subclasses that don't override flush()/close() shouldn't need to --
        # the base class provides harmless defaults.
        class BareBackend(InMemoryBackend):
            def flush(self) -> None:
                return GraphBackend.flush(self)

            def close(self) -> None:
                return GraphBackend.close(self)

        backend = BareBackend()
        backend.flush()
        backend.close()
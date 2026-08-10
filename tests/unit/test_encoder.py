"""Tests for noema.notation.encoder: SemanticGraph -> notation text."""

from __future__ import annotations

from noema.graph.builder import SemanticGraph
from noema.notation.encoder import encode_graph
from noema.schema import Entity, EntityType, Event, Relation


def _entity(name: str, entity_type: EntityType) -> Entity:
    return Entity(name=name, type=entity_type)


class TestEntityAndRelationLines:
    def test_entity_line_format(self):
        graph = SemanticGraph()
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(page)
        graph.add_entity(google)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))

        notation = encode_graph(graph)
        lines = notation.splitlines()

        assert "#1 Larry Page :PERSON" in lines
        assert "#2 Google :ORG" in lines

    def test_relation_line_format(self):
        graph = SemanticGraph()
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(page)
        graph.add_entity(google)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))

        notation = encode_graph(graph)
        assert "#1 -found-> #2" in notation.splitlines()

    def test_local_ids_are_sequential_starting_at_one(self):
        graph = SemanticGraph()
        entities = [_entity(f"Entity{i}", EntityType.OTHER) for i in range(3)]
        for e in entities:
            graph.add_entity(e)
        graph.add_relation(
            Relation(subject_id=entities[0].id, predicate="rel_a", object_id=entities[1].id)
        )
        graph.add_relation(
            Relation(subject_id=entities[1].id, predicate="rel_b", object_id=entities[2].id)
        )

        notation = encode_graph(graph)
        entity_lines = [l for l in notation.splitlines() if l.startswith("#") and ":" in l]
        local_ids = [l.split()[0] for l in entity_lines]

        assert local_ids == ["#1", "#2", "#3"]

    def test_entity_lines_come_before_relation_lines(self):
        graph = SemanticGraph()
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(page)
        graph.add_entity(google)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))

        notation = encode_graph(graph)
        lines = notation.splitlines()
        entity_line_indices = [i for i, l in enumerate(lines) if "->" not in l and l.startswith("#")]
        relation_line_indices = [i for i, l in enumerate(lines) if "->" in l]

        assert max(entity_line_indices) < min(relation_line_indices)


class TestIsolatedEntityFiltering:
    def test_entity_with_no_relation_is_excluded_by_default(self):
        graph = SemanticGraph()
        graph.add_entity(_entity("Lonely Entity", EntityType.OTHER))

        notation = encode_graph(graph)
        assert notation == ""

    def test_entity_participating_in_relation_is_included(self):
        graph = SemanticGraph()
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        isolated = _entity("Isolated", EntityType.OTHER)
        graph.add_entity(page)
        graph.add_entity(google)
        graph.add_entity(isolated)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))

        notation = encode_graph(graph)
        assert "Isolated" not in notation
        assert "Larry Page" in notation
        assert "Google" in notation

    def test_explicit_entity_ids_force_includes_isolated_entity(self):
        graph = SemanticGraph()
        isolated = _entity("Isolated", EntityType.OTHER)
        graph.add_entity(isolated)

        notation = encode_graph(graph, entity_ids=[isolated.id])
        assert "#1 Isolated :OTHER" in notation.splitlines()

    def test_explicit_entity_ids_excludes_relations_with_endpoint_outside_set(self):
        graph = SemanticGraph()
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(page)
        graph.add_entity(google)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))

        notation = encode_graph(graph, entity_ids=[page.id])
        assert "->" not in notation


class TestEventEncoding:
    def test_event_line_format_with_time(self):
        graph = SemanticGraph()
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(google)
        graph.add_event(Event(label="ipo", participant_ids=[google.id], time="2004-08-19"))

        notation = encode_graph(graph)
        assert "*ipo (#1) @2004-08-19" in notation.splitlines()

    def test_event_line_format_without_time(self):
        graph = SemanticGraph()
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(google)
        graph.add_event(Event(label="ipo", participant_ids=[google.id]))

        notation = encode_graph(graph)
        assert "*ipo (#1)" in notation.splitlines()

    def test_event_with_multiple_participants(self):
        graph = SemanticGraph()
        a = _entity("Company A", EntityType.ORG)
        b = _entity("Company B", EntityType.ORG)
        graph.add_entity(a)
        graph.add_entity(b)
        graph.add_event(Event(label="merger", participant_ids=[a.id, b.id]))

        notation = encode_graph(graph)
        assert "*merger (#1,#2)" in notation.splitlines()


class TestEmptyGraph:
    def test_empty_graph_produces_empty_notation(self):
        graph = SemanticGraph()
        assert encode_graph(graph) == ""
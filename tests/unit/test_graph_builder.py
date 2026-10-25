"""Tests for noema.graph.builder: SemanticGraph construction and traversal."""

from __future__ import annotations

import pytest

from noema.graph.builder import SemanticGraph
from noema.schema import Entity, EntityType, Event, ParsedDocument, Relation


def _entity(name: str, entity_type: EntityType = EntityType.OTHER) -> Entity:
    return Entity(name=name, type=entity_type)


class TestAddEntity:
    def test_adding_entity_increases_graph_size(self):
        graph = SemanticGraph()
        graph.add_entity(_entity("Google", EntityType.ORG))
        assert len(graph) == 1

    def test_get_entity_returns_the_entity(self):
        graph = SemanticGraph()
        entity = _entity("Google", EntityType.ORG)
        graph.add_entity(entity)
        assert graph.get_entity(entity.id) is entity

    def test_get_entity_returns_none_for_unknown_id(self):
        graph = SemanticGraph()
        assert graph.get_entity("nonexistent") is None

    def test_readding_same_id_merges_mentions_instead_of_duplicating_node(self):
        graph = SemanticGraph()
        entity = _entity("Google", EntityType.ORG)
        graph.add_entity(entity)
        graph.add_entity(entity)
        assert len(graph) == 1


class TestAddRelation:
    def test_relation_creates_edge_between_existing_entities(self):
        graph = SemanticGraph()
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(page)
        graph.add_entity(google)
        relation = Relation(subject_id=page.id, predicate="found", object_id=google.id)
        graph.add_relation(relation)

        assert relation in graph.relations

    def test_relation_referencing_missing_entity_raises(self):
        graph = SemanticGraph()
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(google)
        relation = Relation(subject_id="ent_missing", predicate="found", object_id=google.id)

        with pytest.raises(ValueError):
            graph.add_relation(relation)

    def test_relations_of_returns_only_outgoing_relations(self):
        graph = SemanticGraph()
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(page)
        graph.add_entity(google)
        relation = Relation(subject_id=page.id, predicate="found", object_id=google.id)
        graph.add_relation(relation)

        assert graph.relations_of(page.id) == [relation]
        assert graph.relations_of(google.id) == []


class TestAddEvent:
    def test_event_creates_participation_edges(self):
        graph = SemanticGraph()
        google = _entity("Google", EntityType.ORG)
        graph.add_entity(google)
        event = Event(label="ipo", participant_ids=[google.id])
        graph.add_event(event)

        assert event in graph.events

    def test_event_referencing_missing_participant_raises(self):
        graph = SemanticGraph()
        event = Event(label="ipo", participant_ids=["ent_missing"])
        with pytest.raises(ValueError):
            graph.add_event(event)


class TestAddDocument:
    def test_add_document_adds_all_entities_and_relations(self):
        page = _entity("Larry Page", EntityType.PERSON)
        google = _entity("Google", EntityType.ORG)
        relation = Relation(subject_id=page.id, predicate="found", object_id=google.id)
        from noema.schema import Document

        parsed = ParsedDocument(
            document=Document(text="Larry Page founded Google."),
            entities=[page, google],
            relations=[relation],
        )

        graph = SemanticGraph()
        graph.add_document(parsed)

        assert len(graph) == 2
        assert relation in graph.relations


class TestNeighbors:
    def _build_star_graph(self) -> tuple[SemanticGraph, Entity, Entity, Entity]:
        """Google <- founded - Larry Page, Google -headquarter_in-> Mountain View."""
        graph = SemanticGraph()
        google = _entity("Google", EntityType.ORG)
        page = _entity("Larry Page", EntityType.PERSON)
        mtv = _entity("Mountain View", EntityType.GPE)
        for e in (google, page, mtv):
            graph.add_entity(e)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))
        graph.add_relation(
            Relation(subject_id=google.id, predicate="headquarter_in", object_id=mtv.id)
        )
        return graph, google, page, mtv

    def test_one_hop_neighbors(self):
        graph, google, _page, _mtv = self._build_star_graph()
        neighbor_names = {e.name for e in graph.neighbors(google.id, hops=1)}
        assert neighbor_names == {"Larry Page", "Mountain View"}

    def test_traversal_is_direction_agnostic(self):
        # Larry Page only has an OUTGOING edge to Google, but neighbors()
        # must still find Google when starting from Larry Page.
        graph, _google, page, _mtv = self._build_star_graph()
        neighbor_names = {e.name for e in graph.neighbors(page.id, hops=1)}
        assert "Google" in neighbor_names

    def test_zero_hops_returns_empty(self):
        graph, google, _page, _mtv = self._build_star_graph()
        assert graph.neighbors(google.id, hops=0) == []

    def test_two_hops_reaches_indirect_entity(self):
        graph, _google, page, _mtv = self._build_star_graph()
        neighbor_names = {e.name for e in graph.neighbors(page.id, hops=2)}
        assert "Mountain View" in neighbor_names

    def test_neighbors_of_unknown_entity_returns_empty(self):
        graph, *_ = self._build_star_graph()
        assert graph.neighbors("ent_missing", hops=1) == []
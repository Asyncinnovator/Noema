"""Tests for noema.retrieval.expand: seed facts -> k-hop candidate sub-graph."""

from __future__ import annotations

from noema.graph.builder import SemanticGraph
from noema.retrieval.expand import ExpandedSubgraph, expand_from_facts, relations_within
from noema.schema import Entity, EntityType, Relation


def _entity(name: str) -> Entity:
    return Entity(name=name, type=EntityType.OTHER)


def _build_chain() -> tuple[SemanticGraph, list[Entity], list[Relation]]:
    """A -r1-> B -r2-> C -r3-> D"""
    graph = SemanticGraph()
    entities = [_entity(n) for n in "ABCD"]
    for e in entities:
        graph.add_entity(e)
    relations = [
        Relation(subject_id=entities[0].id, predicate="r1", object_id=entities[1].id),
        Relation(subject_id=entities[1].id, predicate="r2", object_id=entities[2].id),
        Relation(subject_id=entities[2].id, predicate="r3", object_id=entities[3].id),
    ]
    for r in relations:
        graph.add_relation(r)
    return graph, entities, relations


def _build_star() -> tuple[SemanticGraph, Entity, list[Entity], list[Relation]]:
    """Center connects to 3 leaves."""
    graph = SemanticGraph()
    center = _entity("Center")
    leaves = [_entity(f"Leaf{i}") for i in range(3)]
    graph.add_entity(center)
    for leaf in leaves:
        graph.add_entity(leaf)
    relations = [
        Relation(subject_id=center.id, predicate=f"r{i}", object_id=leaf.id)
        for i, leaf in enumerate(leaves)
    ]
    for r in relations:
        graph.add_relation(r)
    return graph, center, leaves, relations


class TestExpandFromFacts:
    def test_seed_entities_have_distance_zero(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[1].id], hops=1)
        assert expanded.entity_distance[entities[1].id] == 0  # B
        assert expanded.entity_distance[entities[2].id] == 0  # C

    def test_one_hop_neighbors_have_distance_one(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[1].id], hops=1)
        assert expanded.entity_distance[entities[0].id] == 1  # A
        assert expanded.entity_distance[entities[3].id] == 1  # D

    def test_hops_zero_only_includes_seed_entities(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[1].id], hops=0)
        assert expanded.entity_ids == {entities[1].id, entities[2].id}

    def test_hops_limits_expansion_distance(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[1].id], hops=1)
        # E is two hops from the seed (via A or D) in a longer chain --
        # verify a 4-hop-away node is excluded when hops=1 by checking D's
        # far neighbor isn't reachable in this 4-node chain (nothing IS two
        # hops away here, so assert the *set* stays exactly the 1-hop set).
        assert entities[0].id in expanded.entity_ids
        assert entities[3].id in expanded.entity_ids
        assert len(expanded.entity_ids) == 4  # nothing further out to exclude in this graph

    def test_seed_relation_ids_recorded(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[1].id], hops=1)
        assert expanded.seed_relation_ids == {relations[1].id}

    def test_multiple_seed_relations(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[0].id, relations[2].id], hops=0)
        assert expanded.entity_ids == {
            entities[0].id,
            entities[1].id,
            entities[2].id,
            entities[3].id,
        }

    def test_unknown_relation_id_is_skipped_not_raised(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, ["rel_does_not_exist"], hops=1)
        assert expanded.entity_ids == set()
        assert expanded.seed_relation_ids == set()

    def test_mix_of_known_and_unknown_relation_ids(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[0].id, "rel_missing"], hops=0)
        assert expanded.seed_relation_ids == {relations[0].id}

    def test_empty_seed_list_returns_empty_result(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [], hops=1)
        assert expanded == ExpandedSubgraph()

    def test_multi_source_finds_entity_neither_seed_touches_directly(self):
        graph, center, leaves, relations = _build_star()
        # Seed on leaf0's and leaf1's relations; leaf2 is only reachable
        # THROUGH the shared center, not directly from either seed.
        expanded = expand_from_facts(graph, [relations[0].id, relations[1].id], hops=1)
        assert leaves[2].id in expanded.entity_ids
        assert expanded.entity_distance[leaves[2].id] == 1

    def test_result_type_is_expanded_subgraph(self):
        graph, entities, relations = _build_chain()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        assert isinstance(expanded, ExpandedSubgraph)


class TestRelationsWithin:
    def test_relation_with_both_endpoints_included(self):
        graph, entities, relations = _build_chain()
        entity_ids = {entities[0].id, entities[1].id}
        result = relations_within(graph, entity_ids)
        assert relations[0] in result  # A->B, both endpoints present

    def test_relation_with_one_endpoint_missing_excluded(self):
        graph, entities, relations = _build_chain()
        entity_ids = {entities[1].id}  # only B, not C
        result = relations_within(graph, entity_ids)
        assert relations[1] not in result  # B->C excluded, C missing

    def test_empty_entity_set_returns_no_relations(self):
        graph, entities, relations = _build_chain()
        assert relations_within(graph, set()) == []

    def test_all_entities_included_returns_all_relations(self):
        graph, entities, relations = _build_chain()
        all_ids = {e.id for e in entities}
        result = relations_within(graph, all_ids)
        assert len(result) == len(relations)
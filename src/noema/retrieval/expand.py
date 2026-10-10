"""Graph expansion: turn vector-search hits into a candidate sub-graph.

Takes the seed facts (relations) vector_search.py found, pulls out the
entities directly involved, then walks outward up to `hops`
relation/participation edges via a multi-source BFS from all seed entities
together, so each entity in the result also carries its minimum
hop-distance from the nearest seed. assemble.py uses that distance to
decide what to keep or trim under a token budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from noema.graph.builder import SemanticGraph
from noema.schema import Relation


@dataclass
class ExpandedSubgraph:
    """Result of expanding outward from a set of seed facts.

    `entity_distance` maps entity id -> minimum hops from the nearest seed
    entity (0 for entities that were themselves part of a seed fact).
    `seed_relation_ids` are the original vector-search hits, kept separate
    from the wider set of relations the expansion discovers, so assembly can
    prioritize "the facts that directly matched the query" over "facts that
    just happen to touch the same entities."
    """

    entity_ids: set[str] = field(default_factory=set)
    entity_distance: dict[str, int] = field(default_factory=dict)
    seed_relation_ids: set[str] = field(default_factory=set)


def expand_from_facts(
    graph: SemanticGraph, seed_relation_ids: list[str], hops: int = 1
) -> ExpandedSubgraph:
    """Expand from a list of seed Relation ids (e.g. from vector_search.search_facts)
    into a candidate sub-graph: the seed entities plus everything within
    `hops` relation/participation edges of any of them.

    Unknown relation ids (already deleted, or from a stale index) are
    skipped rather than raising -- a stale vector-search hit shouldn't crash
    retrieval, it should just contribute nothing.
    """
    relation_by_id = {r.id: r for r in graph.relations}

    seed_entities: set[str] = set()
    valid_seed_relation_ids: set[str] = set()
    for relation_id in seed_relation_ids:
        relation = relation_by_id.get(relation_id)
        if relation is None:
            continue
        valid_seed_relation_ids.add(relation_id)
        seed_entities.add(relation.subject_id)
        seed_entities.add(relation.object_id)

    if not seed_entities:
        return ExpandedSubgraph()

    # Multi-source BFS over the undirected view, computed once for all
    # seeds together so it gives the true minimum distance when two seed
    # entities are near each other.
    undirected = graph.graph.to_undirected(as_view=True)
    entity_distance: dict[str, int] = {}
    for seed in seed_entities:
        if seed not in undirected:
            continue
        lengths = nx.single_source_shortest_path_length(undirected, seed, cutoff=hops)
        for node_id, distance in lengths.items():
            if node_id not in entity_distance or distance < entity_distance[node_id]:
                entity_distance[node_id] = distance

    # Traversal can walk through/onto event nodes, but they aren't entities
    # and don't belong in entity_distance/entity_ids.
    entity_distance = {
        node_id: dist for node_id, dist in entity_distance.items() if graph.get_entity(node_id)
    }

    return ExpandedSubgraph(
        entity_ids=set(entity_distance.keys()),
        entity_distance=entity_distance,
        seed_relation_ids=valid_seed_relation_ids,
    )


def relations_within(graph: SemanticGraph, entity_ids: set[str]) -> list[Relation]:
    """Relations whose subject AND object both fall within `entity_ids`.

    A relation touching only one included entity (the other endpoint fell
    outside the hop budget) is excluded, since a half-visible fact isn't
    meaningful notation.
    """
    return [
        relation
        for relation in graph.relations
        if relation.subject_id in entity_ids and relation.object_id in entity_ids
    ]
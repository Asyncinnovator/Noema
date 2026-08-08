"""Assembles entities, relations, and events into an in-memory semantic graph.

This is the raw, pre-canonicalization graph: each ParsedDocument contributes
its own entities with their own ids, so the same real-world thing mentioned
in two documents ("Google" in doc A, "Google" in doc B) becomes two separate
nodes here. Merging those into one canonical node is the canonicalize
subpackage's job (Phase 1 basic version, Phase 5 stronger version) -- this
module only builds the structure the canonicalizer will operate on.

Uses networkx.MultiDiGraph so that multiple distinct relations can exist
between the same pair of entities (e.g. "worked_at" and "board_member_of"
both holding between the same two nodes) without overwriting each other.
"""

from __future__ import annotations

import networkx as nx

from noema.schema import Entity, Event, ParsedDocument, Relation

# Node "kind" values, stored as a node attribute so entity nodes and event
# nodes can share one graph without a separate lookup structure.
ENTITY_KIND = "entity"
EVENT_KIND = "event"


class SemanticGraph:
    """Thin, typed wrapper around a networkx.MultiDiGraph.

    Nodes are entities and events, keyed by their schema id. Edges are:
      - relation edges: subject_id -> object_id, keyed by relation.id
      - participation edges: participant_id -> event.id, keyed by event.id,
        representing "this entity took part in this event"

    Pydantic objects are stored whole as node/edge data (under the "data"
    attribute) so nothing is lost converting into graph form -- the graph is
    a view/index over the same Entity/Relation/Event objects, not a lossy
    projection of them.
    """

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # -- construction ----------------------------------------------------

    def add_document(self, parsed: ParsedDocument) -> None:
        """Add all entities, relations, and events from one parsed document."""
        for entity in parsed.entities:
            self.add_entity(entity)
        for relation in parsed.relations:
            self.add_relation(relation)
        for event in parsed.events:
            self.add_event(event)
  
    def add_entity(self, entity: Entity) -> None:
        if self.graph.has_node(entity.id):
            # Same id re-added (e.g. document processed twice) -- merge
            # mentions rather than duplicate the node.
            existing: Entity = self.graph.nodes[entity.id]["data"]
            existing.mentions.extend(entity.mentions)
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            return
        self.graph.add_node(entity.id, kind=ENTITY_KIND, data=entity)

    def add_relation(self, relation: Relation) -> None:
        if relation.subject_id not in self.graph or relation.object_id not in self.graph:
            raise ValueError(
                f"Relation {relation.id} references entity id(s) not yet in the graph "
                f"({relation.subject_id!r} -> {relation.object_id!r}). "
                "Add entities before relations that reference them."
            )
        self.graph.add_edge(
            relation.subject_id,
            relation.object_id,
            key=relation.id,
            kind="relation",
            data=relation,
        )

    def add_event(self, event: Event) -> None:
        self.graph.add_node(event.id, kind=EVENT_KIND, data=event)
        for participant_id in event.participant_ids:
            if participant_id not in self.graph:
                raise ValueError(
                    f"Event {event.id} references participant {participant_id!r} "
                    "not yet in the graph. Add entities before events that reference them."
                )
            self.graph.add_edge(
                participant_id,
                event.id,
                key=event.id,
                kind="participation",
                data=event,
            )

    # -- accessors ---------------------------------------------------------

    def get_entity(self, entity_id: str) -> Entity | None:
        node = self.graph.nodes.get(entity_id)
        if node is None or node["kind"] != ENTITY_KIND:
            return None
        return node["data"]

    def get_event(self, event_id: str) -> Event | None:
        node = self.graph.nodes.get(event_id)
        if node is None or node["kind"] != EVENT_KIND:
            return None
        return node["data"]

    def relations_of(self, entity_id: str) -> list[Relation]:
        """All relations where this entity is the subject (outgoing edges)."""
        return [
            data["data"]
            for _, _, data in self.graph.out_edges(entity_id, data=True)
            if data["kind"] == "relation"
        ]

    @property
    def entities(self) -> list[Entity]:
        return [
            data["data"] for _, data in self.graph.nodes(data=True) if data["kind"] == ENTITY_KIND
        ]

    @property
    def relations(self) -> list[Relation]:
        return [
            data["data"] for _, _, data in self.graph.edges(data=True) if data["kind"] == "relation"
        ]

    @property
    def events(self) -> list[Event]:
        return [
            data["data"] for _, data in self.graph.nodes(data=True) if data["kind"] == EVENT_KIND
        ]

    # -- traversal ---------------------------------------------------------

    def neighbors(self, entity_id: str, hops: int = 1) -> list[Entity]:
        """Entities reachable within `hops` relation/participation edges,
        treating the graph as undirected for traversal purposes (a fact is
        relevant regardless of which side of the edge you start from).

        This is the primitive the retrieval layer (Phase 4) builds on for
        "vector search -> graph expansion" sub-graph assembly.
        """
        if entity_id not in self.graph:
            return []
        undirected = self.graph.to_undirected(as_view=True)
        reachable = nx.single_source_shortest_path_length(undirected, entity_id, cutoff=hops)
        return [
            self.get_entity(node_id)
            for node_id in reachable
            if node_id != entity_id and self.get_entity(node_id) is not None
        ]

    def __len__(self) -> int:
        return self.graph.number_of_nodes()
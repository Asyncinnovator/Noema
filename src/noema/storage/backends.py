"""Pluggable persistent graph backend interface.

A .noema bundle keeps the whole graph in one file, loaded entirely into
memory. GraphBackend is the interface an external store (Neo4j, an RDF
triple store) implements so a SemanticGraph can instead live there, for
knowledge bases too large to comfortably hold as a single in-memory graph.

Concrete backends are optional -- see neo4j_backend.py and rdf_backend.py.
Both import their third-party client lazily so this module and KnowledgeBase
have no hard dependency on either.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Self

from noema.graph.builder import SemanticGraph
from noema.schema import Entity, Event, Relation


class GraphBackend(ABC):
    """Storage interface for a SemanticGraph's entities, relations, and events.

    Implementations may batch writes internally; callers should treat
    upsert_* calls as not visible to reads until flush() returns.
    """

    @abstractmethod
    def upsert_entities(self, entities: list[Entity]) -> None:
        """Insert or update entities, keyed by Entity.id."""

    @abstractmethod
    def upsert_relations(self, relations: list[Relation]) -> None:
        """Insert or update relations, keyed by Relation.id.

        Referenced subject/object entities must already exist in the
        backend (upsert entities first).
        """

    @abstractmethod
    def upsert_events(self, events: list[Event]) -> None:
        """Insert or update events, keyed by Event.id.

        Referenced participant entities must already exist in the backend.
        """

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None:
        """Fetch a single entity by id, or None if not present."""

    @abstractmethod
    def all_entities(self) -> Iterator[Entity]:
        """Iterate every entity currently in the backend."""

    @abstractmethod
    def all_relations(self) -> Iterator[Relation]:
        """Iterate every relation currently in the backend."""

    @abstractmethod
    def all_events(self) -> Iterator[Event]:
        """Iterate every event currently in the backend."""

    @abstractmethod
    def entity_count(self) -> int:
        """Total number of entities in the backend."""

    @abstractmethod
    def clear(self) -> None:
        """Remove everything this backend holds, leaving it empty."""

    def flush(self) -> None:
        """Ensure prior upserts are durable/visible. No-op unless overridden."""

    def close(self) -> None:
        """Release any underlying connection. No-op unless overridden."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def write_graph(backend: GraphBackend, graph: SemanticGraph, *, clear_first: bool = True) -> None:
    """Push an entire in-memory SemanticGraph into a backend.

    Entities are written before relations/events since both reference
    entity ids. `clear_first` wipes the backend's existing contents first,
    so the backend ends up matching `graph` exactly rather than merging.
    """
    if clear_first:
        backend.clear()
    backend.upsert_entities(graph.entities)
    backend.upsert_relations(graph.relations)
    backend.upsert_events(graph.events)
    backend.flush()


def read_graph(backend: GraphBackend) -> SemanticGraph:
    """Pull a backend's full contents into a new in-memory SemanticGraph."""
    graph = SemanticGraph()
    for entity in backend.all_entities():
        graph.add_entity(entity)
    for relation in backend.all_relations():
        graph.add_relation(relation)
    for event in backend.all_events():
        graph.add_event(event)
    return graph
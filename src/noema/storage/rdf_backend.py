"""RDF triple-store GraphBackend, built on rdflib.

Each entity becomes a URIRef subject under a fixed `noema:` namespace, with
its name/type/aliases/confidence as literal triples. Each relation becomes
a triple from the subject entity's URI to the object entity's URI, using
the relation's predicate (namespaced) as the RDF predicate itself, plus a
reified node for the relation's own id/confidence/mentions, which plain
subject-predicate-object triples can't carry on their own.

rdflib is imported lazily, same pattern as neo4j_backend -- only
instantiating RDFBackend requires it installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from noema.schema import Entity, EntityType, Event, Relation, SourceSpan
from noema.storage.backends import GraphBackend

try:
    import rdflib
    from rdflib import RDF, Literal, Namespace, URIRef
except ImportError:
    rdflib = None

NS = "http://noema.dev/kg#"


def _require_rdflib() -> None:
    if rdflib is None:
        raise ImportError(
            "RDFBackend requires the 'rdflib' package. Install it with: "
            "pip install noema[rdf]"
        )


def _entity_uri(ns: Namespace, entity_id: str) -> URIRef:
    return ns[f"entity/{entity_id}"]


def _relation_uri(ns: Namespace, relation_id: str) -> URIRef:
    return ns[f"relation/{relation_id}"]


def _event_uri(ns: Namespace, event_id: str) -> URIRef:
    return ns[f"event/{event_id}"]


class RDFBackend(GraphBackend):
    """GraphBackend implementation over an in-memory or file-persisted
    rdflib.Graph.

        backend = RDFBackend(path="kb.ttl")   # loads if the file exists
        write_graph(backend, kb.graph)
        backend.save()                        # persists to kb.ttl

    With no `path`, the store lives purely in memory for the process
    lifetime (same lifecycle as the default .noema bundle, but as triples).
    """

    def __init__(self, path: str | None = None, format: str = "turtle") -> None:
        _require_rdflib()
        self._ns = Namespace(NS)
        self._path = path
        self._format = format
        self._graph = rdflib.Graph()
        self._graph.bind("noema", self._ns)
        # Check existence ourselves rather than catching FileNotFoundError:
        # for a missing path, rdflib's parser tries to interpret the string
        # as a URI before opening it as a file, and on Windows a drive
        # letter like "C:\" is misread as a URI scheme, raising a URLError
        # instead of FileNotFoundError.
        if path is not None and os.path.exists(path):
            self._graph.parse(path, format=format)

    def save(self, path: str | None = None) -> None:
        """Serialize the triple store to disk (defaults to the path given
        at construction time).
        """
        target = path or self._path
        if target is None:
            raise ValueError("No path given to RDFBackend() or save()")
        self._graph.serialize(destination=target, format=self._format)

    def flush(self) -> None:
        if self._path is not None:
            self.save()

    def upsert_entities(self, entities: list[Entity]) -> None:
        ns = self._ns
        for entity in entities:
            uri = _entity_uri(ns, entity.id)
            self._graph.remove((uri, None, None))
            self._graph.add((uri, RDF.type, ns.Entity))
            self._graph.add((uri, ns.id, Literal(entity.id)))
            self._graph.add((uri, ns.name, Literal(entity.name)))
            self._graph.add((uri, ns.entityType, Literal(entity.type.value)))
            self._graph.add((uri, ns.confidence, Literal(entity.confidence)))
            for alias in entity.aliases:
                self._graph.add((uri, ns.alias, Literal(alias)))
            for mention in entity.mentions:
                self._graph.add((uri, ns.mention, Literal(mention.model_dump_json())))

    def upsert_relations(self, relations: list[Relation]) -> None:
        ns = self._ns
        for relation in relations:
            subj_uri = _entity_uri(ns, relation.subject_id)
            obj_uri = _entity_uri(ns, relation.object_id)
            rel_uri = _relation_uri(ns, relation.id)
            self._graph.remove((rel_uri, None, None))
            # Reified node carries the relation's own id/predicate/metadata.
            self._graph.add((rel_uri, RDF.type, ns.Relation))
            self._graph.add((rel_uri, ns.id, Literal(relation.id)))
            self._graph.add((rel_uri, ns.subject, subj_uri))
            self._graph.add((rel_uri, ns.object, obj_uri))
            self._graph.add((rel_uri, ns.predicate, Literal(relation.predicate)))
            self._graph.add((rel_uri, ns.confidence, Literal(relation.confidence)))
            for mention in relation.mentions:
                self._graph.add((rel_uri, ns.mention, Literal(mention.model_dump_json())))
            # ...plus a direct subject-predicate-object triple, so the store
            # is also queryable as plain RDF without knowing about reification.
            self._graph.add((subj_uri, ns[relation.predicate], obj_uri))

    def upsert_events(self, events: list[Event]) -> None:
        ns = self._ns
        for event in events:
            uri = _event_uri(ns, event.id)
            self._graph.remove((uri, None, None))
            self._graph.add((uri, RDF.type, ns.Event))
            self._graph.add((uri, ns.id, Literal(event.id)))
            self._graph.add((uri, ns.label, Literal(event.label)))
            if event.time is not None:
                self._graph.add((uri, ns.time, Literal(event.time)))
            self._graph.add((uri, ns.confidence, Literal(event.confidence)))
            for mention in event.mentions:
                self._graph.add((uri, ns.mention, Literal(mention.model_dump_json())))
            for participant_id in event.participant_ids:
                self._graph.add((uri, ns.participant, _entity_uri(ns, participant_id)))

    def get_entity(self, entity_id: str) -> Entity | None:
        for entity in self.all_entities():
            if entity.id == entity_id:
                return entity
        return None

    def all_entities(self) -> Iterator[Entity]:
        ns = self._ns
        for uri in self._graph.subjects(RDF.type, ns.Entity):
            yield Entity(
                id=str(self._graph.value(uri, ns.id)),
                name=str(self._graph.value(uri, ns.name)),
                type=EntityType(str(self._graph.value(uri, ns.entityType))),
                aliases=[str(a) for a in self._graph.objects(uri, ns.alias)],
                mentions=[
                    SourceSpan.model_validate_json(str(m))
                    for m in self._graph.objects(uri, ns.mention)
                ],
                confidence=float(self._graph.value(uri, ns.confidence)),
            )

    def all_relations(self) -> Iterator[Relation]:
        ns = self._ns
        for uri in self._graph.subjects(RDF.type, ns.Relation):
            subject_uri = self._graph.value(uri, ns.subject)
            object_uri = self._graph.value(uri, ns.object)
            yield Relation(
                id=str(self._graph.value(uri, ns.id)),
                subject_id=_uri_to_id(str(subject_uri)),
                predicate=str(self._graph.value(uri, ns.predicate)),
                object_id=_uri_to_id(str(object_uri)),
                mentions=[
                    SourceSpan.model_validate_json(str(m))
                    for m in self._graph.objects(uri, ns.mention)
                ],
                confidence=float(self._graph.value(uri, ns.confidence)),
            )

    def all_events(self) -> Iterator[Event]:
        ns = self._ns
        for uri in self._graph.subjects(RDF.type, ns.Event):
            time_value = self._graph.value(uri, ns.time)
            yield Event(
                id=str(self._graph.value(uri, ns.id)),
                label=str(self._graph.value(uri, ns.label)),
                time=str(time_value) if time_value is not None else None,
                participant_ids=[
                    _uri_to_id(str(p)) for p in self._graph.objects(uri, ns.participant)
                ],
                mentions=[
                    SourceSpan.model_validate_json(str(m))
                    for m in self._graph.objects(uri, ns.mention)
                ],
                confidence=float(self._graph.value(uri, ns.confidence)),
            )

    def entity_count(self) -> int:
        return sum(1 for _ in self._graph.subjects(RDF.type, self._ns.Entity))

    def clear(self) -> None:
        self._graph.remove((None, None, None))


def _uri_to_id(uri: str) -> str:
    """Recover the bare schema id from an entity/.../<id> URI."""
    return uri.rsplit("/", 1)[-1]
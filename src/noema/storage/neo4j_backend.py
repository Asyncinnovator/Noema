"""Neo4j-backed GraphBackend, for knowledge bases too large to keep as a
single in-memory .noema bundle.

Entities become `:Entity` nodes (plus a type label, e.g. `:PERSON`), keyed
by `id`. Relations and event-participation links become edges. The `neo4j`
driver is imported lazily so importing this module never requires it --
only instantiating Neo4jBackend does.
"""

from __future__ import annotations

from collections.abc import Iterator

from noema.schema import Entity, EntityType, Event, Relation, SourceSpan
from noema.storage.backends import GraphBackend

try:
    import neo4j as _neo4j
except ImportError:
    _neo4j = None

# Relation predicates aren't fixed ahead of time, so every relation edge
# uses one generic type and keeps the real predicate as a property --
# Cypher relationship types can't be parameterized per-row.
_RELATION_EDGE = "RELATES_TO"
_PARTICIPATION_EDGE = "PARTICIPATES_IN"


def _require_driver() -> None:
    if _neo4j is None:
        raise ImportError(
            "Neo4jBackend requires the 'neo4j' package. Install it with: "
            "pip install noema-rag[neo4j]"
        )


def _entity_to_props(entity: Entity) -> dict:
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": entity.aliases,
        "confidence": entity.confidence,
        "mentions": [m.model_dump_json() for m in entity.mentions],
    }


def _props_to_entity(props: dict) -> Entity:
    return Entity(
        id=props["id"],
        name=props["name"],
        type=EntityType(props["type"]),
        aliases=list(props.get("aliases") or []),
        mentions=[SourceSpan.model_validate_json(m) for m in props.get("mentions") or []],
        confidence=props.get("confidence", 1.0),
    )


def _relation_to_props(relation: Relation) -> dict:
    return {
        "id": relation.id,
        "predicate": relation.predicate,
        "confidence": relation.confidence,
        "mentions": [m.model_dump_json() for m in relation.mentions],
    }


def _props_to_relation(subject_id: str, object_id: str, props: dict) -> Relation:
    return Relation(
        id=props["id"],
        subject_id=subject_id,
        predicate=props["predicate"],
        object_id=object_id,
        mentions=[SourceSpan.model_validate_json(m) for m in props.get("mentions") or []],
        confidence=props.get("confidence", 1.0),
    )


class Neo4jBackend(GraphBackend):
    """GraphBackend implementation over a Neo4j database.

        backend = Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "password"))
        write_graph(backend, kb.graph)
        ...
        kb2 = KnowledgeBase()
        kb2.load_from_backend(backend)
    """

    def __init__(self, uri: str, auth: tuple[str, str] | None = None, database: str = "neo4j") -> None:
        _require_driver()
        self._driver = _neo4j.GraphDatabase.driver(uri, auth=auth)
        self._database = database

    def _session(self):
        return self._driver.session(database=self._database)

    def upsert_entities(self, entities: list[Entity]) -> None:
        if not entities:
            return
        rows = [_entity_to_props(e) for e in entities]
        with self._session() as session:
            session.run(
                "UNWIND $rows AS row "
                "MERGE (e:Entity {id: row.id}) "
                "SET e += row",
                rows=rows,
            )

    def upsert_relations(self, relations: list[Relation]) -> None:
        if not relations:
            return
        rows = [
            {"subject_id": r.subject_id, "object_id": r.object_id, **_relation_to_props(r)}
            for r in relations
        ]
        with self._session() as session:
            session.run(
                "UNWIND $rows AS row "
                "MATCH (s:Entity {id: row.subject_id}), (o:Entity {id: row.object_id}) "
                f"MERGE (s)-[rel:{_RELATION_EDGE} {{id: row.id}}]->(o) "
                "SET rel += row",
                rows=rows,
            )

    def upsert_events(self, events: list[Event]) -> None:
        if not events:
            return
        with self._session() as session:
            for event in events:
                session.run(
                    "MERGE (ev:Event {id: $id}) "
                    "SET ev.label = $label, ev.time = $time, ev.confidence = $confidence "
                    "WITH ev "
                    "UNWIND $participant_ids AS pid "
                    "MATCH (p:Entity {id: pid}) "
                    f"MERGE (p)-[:{_PARTICIPATION_EDGE}]->(ev)",
                    id=event.id,
                    label=event.label,
                    time=event.time,
                    confidence=event.confidence,
                    participant_ids=event.participant_ids,
                )

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._session() as session:
            record = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e", id=entity_id
            ).single()
        return _props_to_entity(dict(record["e"])) if record else None

    def all_entities(self) -> Iterator[Entity]:
        with self._session() as session:
            result = session.run("MATCH (e:Entity) RETURN e")
            for record in result:
                yield _props_to_entity(dict(record["e"]))

    def all_relations(self) -> Iterator[Relation]:
        with self._session() as session:
            result = session.run(
                f"MATCH (s:Entity)-[rel:{_RELATION_EDGE}]->(o:Entity) "
                "RETURN s.id AS subject_id, o.id AS object_id, rel AS rel"
            )
            for record in result:
                yield _props_to_relation(
                    record["subject_id"], record["object_id"], dict(record["rel"])
                )

    def all_events(self) -> Iterator[Event]:
        with self._session() as session:
            result = session.run(
                f"MATCH (p:Entity)-[:{_PARTICIPATION_EDGE}]->(ev:Event) "
                "RETURN ev.id AS id, ev.label AS label, ev.time AS time, "
                "ev.confidence AS confidence, collect(p.id) AS participant_ids"
            )
            for record in result:
                yield Event(
                    id=record["id"],
                    label=record["label"],
                    time=record["time"],
                    confidence=record["confidence"],
                    participant_ids=list(record["participant_ids"]),
                )

    def entity_count(self) -> int:
        with self._session() as session:
            record = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()
        return record["n"] if record else 0

    def clear(self) -> None:
        with self._session() as session:
            session.run("MATCH (n) WHERE n:Entity OR n:Event DETACH DELETE n")

    def close(self) -> None:
        self._driver.close()
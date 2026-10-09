"""Compact notation: the token-efficient text form a sub-graph compresses to.

Grammar (formal, line-oriented -- one fact per line, no filler words):

    notation      := entity_line* relation_line* event_line*
    entity_line   := "#" id " " name " :" TYPE NEWLINE
    relation_line := "#" subject_id " -" predicate "-> #" object_id NEWLINE
    event_line    := "#" id " *" label " (" participant_id ("," participant_id)* ")" ["@" time] NEWLINE
    id            := DIGIT+
    TYPE          := "PERSON" | "ORG" | "GPE" | "LOC" | "DATE" | "EVENT"
                    | "PRODUCT" | "MONEY" | "OTHER"

Example, for "Larry Page founded Google in 1998. Sundar Pichai is the CEO of Google.":

    #1 Larry Page :PERSON
    #2 Google :ORG
    #3 Sundar Pichai :PERSON
    #1 -found-> #2
    #3 -ceo_of-> #2

Ids are short, local, sequential integers scoped to this notation block, not
the long ent_/rel_ ids schema.py generates. Every referenced entity gets
exactly one entity_line, emitted before any line that references it, so the
notation parses in one top-to-bottom pass (see notation.decoder).
"""

from __future__ import annotations

from noema.graph.builder import SemanticGraph
from noema.schema import Entity, Event, Relation


def encode_graph(graph: SemanticGraph, entity_ids: list[str] | None = None) -> str:
    """Render a SemanticGraph (or a chosen sub-graph of it) as notation text.

    `entity_ids` restricts encoding to a specific sub-graph, e.g. the ids
    returned by SemanticGraph.neighbors() for query-time retrieval. Leaving
    it None encodes the whole graph, except entities with no relation or
    event referencing them are dropped (an isolated entity has no fact to
    compress). Pass entity_ids explicitly to force-include a specific
    entity regardless.
    """
    included_entity_ids = (
        set(entity_ids) if entity_ids is not None else {e.id for e in graph.entities}
    )

    relations = [
        r
        for r in graph.relations
        if r.subject_id in included_entity_ids and r.object_id in included_entity_ids
    ]
    events = [
        ev
        for ev in graph.events
        if all(pid in included_entity_ids for pid in ev.participant_ids)
    ]

    if entity_ids is None:
        referenced_ids = {r.subject_id for r in relations} | {r.object_id for r in relations}
        referenced_ids |= {pid for ev in events for pid in ev.participant_ids}
        included_entity_ids = referenced_ids

    entities = [e for e in graph.entities if e.id in included_entity_ids]

    local_id = {entity.id: str(i + 1) for i, entity in enumerate(entities)}

    lines: list[str] = []
    lines.extend(_encode_entity(e, local_id) for e in entities)
    lines.extend(_encode_relation(r, local_id) for r in relations)
    lines.extend(_encode_event(ev, local_id) for ev in events)

    return "\n".join(lines)


def _encode_entity(entity: Entity, local_id: dict[str, str]) -> str:
    return f"#{local_id[entity.id]} {entity.name} :{entity.type.value}"


def _encode_relation(relation: Relation, local_id: dict[str, str]) -> str:
    subject = local_id[relation.subject_id]
    obj = local_id[relation.object_id]
    return f"#{subject} -{relation.predicate}-> #{obj}"


def _encode_event(event: Event, local_id: dict[str, str]) -> str:
    participants = ",".join(f"#{local_id[pid]}" for pid in event.participant_ids)
    line = f"*{event.label} ({participants})"
    if event.time:
        line += f" @{event.time}"
    return line
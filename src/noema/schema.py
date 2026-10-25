"""Core data types shared across Noema's pipeline stages: parser, graph,
canonicalize, notation, storage. Built on pydantic for validation and
(de)serialization to graph.json.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    """Short, prefixed unique id, e.g. 'ent_9f3a2b1c'."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class EntityType(str, Enum):
    """Coarse entity types, aligned with spaCy's default NER labels.
    Unrecognized labels fall back to OTHER.
    """

    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"  # countries, cities, states
    LOC = "LOC"  # non-GPE locations
    DATE = "DATE"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    MONEY = "MONEY"
    OTHER = "OTHER"

    @classmethod
    def from_spacy_label(cls, label: str) -> EntityType:
        try:
            return cls(label)
        except ValueError:
            return cls.OTHER


class SourceSpan(BaseModel):
    """A single mention of an entity/relation/event in a source document.

    Tracked per-mention so dedup and the round-trip validator can trace a
    canonical fact back to the original text.
    """

    doc_id: str
    start_char: int
    end_char: int
    text: str


class Entity(BaseModel):
    """A canonical entity node.

    `id` identifies this entity for Relation/Event references. `aliases`
    accumulates alternate surface forms found during canonicalization.
    """

    id: str = Field(default_factory=lambda: _new_id("ent"))
    name: str
    type: EntityType = EntityType.OTHER
    aliases: list[str] = Field(default_factory=list)
    mentions: list[SourceSpan] = Field(default_factory=list)
    confidence: float = 1.0

    def add_mention(self, span: SourceSpan) -> None:
        self.mentions.append(span)
        if span.text != self.name and span.text not in self.aliases:
            self.aliases.append(span.text)


class Relation(BaseModel):
    """A subject-predicate-object fact linking two entities.

    `predicate` is a normalized, lowercase, snake_case label (e.g.
    "founded", "ceo_of"), produced by the parser.
    """

    id: str = Field(default_factory=lambda: _new_id("rel"))
    subject_id: str
    predicate: str
    object_id: str
    mentions: list[SourceSpan] = Field(default_factory=list)
    confidence: float = 1.0
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class Event(BaseModel):
    """A fact involving multiple entities and/or a time anchor, unlike a
    Relation which links exactly two entities.
    """

    id: str = Field(default_factory=lambda: _new_id("evt"))
    label: str
    participant_ids: list[str] = Field(default_factory=list)
    time: str | None = None  # free text for now; normalized later
    mentions: list[SourceSpan] = Field(default_factory=list)
    confidence: float = 1.0


class Document(BaseModel):
    """A single ingested source document, referenced by SourceSpan.doc_id."""

    id: str = Field(default_factory=lambda: _new_id("doc"))
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Raw per-document extractions, before canonicalization/dedup against
    the rest of the knowledge base.
    """

    document: Document
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
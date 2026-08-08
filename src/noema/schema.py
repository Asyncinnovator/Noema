"""Core data schema shared across Noema's pipeline stages.

Every stage (parser -> graph -> canonicalize -> notation -> storage) operates
on these types, so they live at the package root rather than inside any one
subpackage. Built on pydantic v2 for validation and easy (de)serialization
to graph.json.
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

    Kept intentionally small for Phase 1. Domain-specific types can be
    added later without breaking existing graphs (unknown labels fall
    back to OTHER).
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
    def from_spacy_label(cls, label: str) -> "EntityType":
        try:
            return cls(label)
        except ValueError:
            return cls.OTHER


class SourceSpan(BaseModel):
    """A single mention of an entity/relation/event in a source document.

    Provenance is tracked per-mention (not just per-fact) so that dedup and
    the round-trip validator can trace any canonical fact back to the
    original text it was derived from.
    """

    doc_id: str
    start_char: int
    end_char: int
    text: str


class Entity(BaseModel):
    """A canonical entity node.

    `id` is stable identity used by Relation/Event to reference this entity.
    `aliases` accumulates alternate surface forms discovered during
    canonicalization (e.g. "Google" / "Alphabet's Google" / "the company").
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

    `predicate` is a normalized, lowercase, snake_case verb/relation label
    (e.g. "founded", "ceo_of", "acquired"), not raw verb text. Normalization
    happens in the parser; this class only stores the result.
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
    """A higher-order fact involving multiple entities and/or a time anchor.

    Events are distinguished from Relations in that they may have more than
    two participants and typically anchor to a point or span in time
    (e.g. "Google IPO'd on 2004-08-19" with participants=[Google]).
    """

    id: str = Field(default_factory=lambda: _new_id("evt"))
    label: str
    participant_ids: list[str] = Field(default_factory=list)
    time: str | None = None  # kept as free text in Phase 1; normalized later
    mentions: list[SourceSpan] = Field(default_factory=list)
    confidence: float = 1.0


class Document(BaseModel):
    """A single ingested source document, referenced by SourceSpan.doc_id."""

    id: str = Field(default_factory=lambda: _new_id("doc"))
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Output of the parser stage for one document: raw extractions before
    canonicalization/dedup against the rest of the knowledge base.
    """

    document: Document
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
"""Notation -> natural language.

Two responsibilities, kept in one module since they share the same parsing
step:

  1. parse_notation() -- reads notation text (see notation.encoder for the
     grammar) back into structured facts. This is a real parser, not just a
     stepping stone to text: storage and retrieval (Phases 3-4) will also
     need "notation string -> structured facts" independent of NL rendering.
  2. decode() -- renders those parsed facts as natural language sentences,
     for the round-trip validator (validator.roundtrip) to compare against
     the original source text.

decode() is template-based, not a generative model -- Phase 1/2 has no NLG
component. Predicates are humanized by simple pattern rules (see
_humanize_predicate), which produce grammatically rough but semantically
recognizable sentences ("Google is headquarter in Mountain View" rather than
the fully-conjugated "Google is headquartered in Mountain View"). This is
an accepted limitation for now: the round-trip validator's first-pass
confidence signal is embedding cosine similarity (Phase 2), which is fairly
tolerant of minor grammar roughness since it scores meaning, not fluency.
Phase 5's NLI-based validator will need decode() to be tightened, or swapped
for something generative, if that roughness turns out to matter in practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # unused here, kept for symmetry with other modules

_ENTITY_LINE = re.compile(r"^#(\d+) (.+) :(\w+)$")
_RELATION_LINE = re.compile(r"^#(\d+) -(.+)-> #(\d+)$")
_EVENT_LINE = re.compile(r"^\*(\S+) \(([^)]*)\)(?: @(.+))?$")

_PREPOSITIONS = {
    "in", "of", "at", "by", "from", "to", "with", "for", "on", "into", "under", "over",
}


@dataclass
class NotationEntity:
    local_id: str
    name: str
    type: str


@dataclass
class NotationRelation:
    subject_id: str
    predicate: str
    object_id: str


@dataclass
class NotationEvent:
    label: str
    participant_ids: list[str]
    time: str | None = None


@dataclass
class ParsedNotation:
    entities: dict[str, NotationEntity] = field(default_factory=dict)
    relations: list[NotationRelation] = field(default_factory=list)
    events: list[NotationEvent] = field(default_factory=list)


def parse_notation(notation: str) -> ParsedNotation:
    """Parse notation text into structured facts.

    Raises ValueError with a line number on malformed input, or on a
    relation/event referencing an entity id no entity_line has defined --
    the grammar requires entities to be declared before anything references
    them (see notation.encoder docstring), so a forward reference is always
    either a hand-editing mistake or truncated notation.
    """
    parsed = ParsedNotation()

    for line_number, raw_line in enumerate(notation.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("*"):
            match = _EVENT_LINE.match(line)
            if not match:
                raise ValueError(f"Line {line_number}: malformed event line: {raw_line!r}")
            label, participants_str, time = match.groups()
            participant_ids = [p.strip().lstrip("#") for p in participants_str.split(",") if p.strip()]
            for pid in participant_ids:
                _require_known_entity(parsed, pid, line_number, raw_line)
            parsed.events.append(NotationEvent(label=label, participant_ids=participant_ids, time=time))
            continue

        if "->" in line:
            match = _RELATION_LINE.match(line)
            if not match:
                raise ValueError(f"Line {line_number}: malformed relation line: {raw_line!r}")
            subject_id, predicate, object_id = match.groups()
            _require_known_entity(parsed, subject_id, line_number, raw_line)
            _require_known_entity(parsed, object_id, line_number, raw_line)
            parsed.relations.append(
                NotationRelation(subject_id=subject_id, predicate=predicate, object_id=object_id)
            )
            continue

        match = _ENTITY_LINE.match(line)
        if not match:
            raise ValueError(f"Line {line_number}: malformed entity line: {raw_line!r}")
        local_id, name, entity_type = match.groups()
        parsed.entities[local_id] = NotationEntity(local_id=local_id, name=name, type=entity_type)

    return parsed


def _require_known_entity(parsed: ParsedNotation, entity_id: str, line_number: int, raw_line: str) -> None:
    if entity_id not in parsed.entities:
        raise ValueError(
            f"Line {line_number}: references entity #{entity_id}, which was not "
            f"declared before this line: {raw_line!r}"
        )


def _humanize_predicate(predicate: str) -> str:
    """'found' -> 'found', 'ceo_of' -> 'is the ceo of', 'headquarter_in' -> 'is headquarter in'."""
    words = predicate.split("_")
    if len(words) == 1:
        return words[0]
    if words[-1] == "of":
        return f"is the {' '.join(words[:-1])} of"
    if words[-1] in _PREPOSITIONS:
        return f"is {' '.join(words[:-1])} {words[-1]}"
    return " ".join(words)


def decode(notation: str) -> str:
    """Render notation text as a natural-language paragraph.

    One sentence per relation, in the order they appear in the notation,
    followed by one sentence per event. Multiple relations sharing a
    subject are NOT combined into a single sentence in Phase 2 -- that's a
    fluency nicety, not a meaning-preservation requirement, and combining
    them correctly needs real NLG rather than templates.
    """
    parsed = parse_notation(notation)
    sentences: list[str] = []

    for relation in parsed.relations:
        subject = parsed.entities[relation.subject_id].name
        obj = parsed.entities[relation.object_id].name
        phrase = _humanize_predicate(relation.predicate)
        sentences.append(f"{subject} {phrase} {obj}.")

    for event in parsed.events:
        participant_names = [parsed.entities[pid].name for pid in event.participant_ids]
        who = " and ".join(participant_names) if participant_names else "It"
        sentence = f"{who} {event.label.replace('_', ' ')}"
        if event.time:
            sentence += f" on {event.time}"
        sentences.append(sentence + ".")

    return " ".join(sentences)
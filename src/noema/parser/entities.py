"""Text -> Entity extraction using spaCy NER.

Runs spaCy's NER, maps its labels onto EntityType, and collapses exact-text
repeats within a single document into one Entity with multiple mentions.
Cross-document merging (fuzzy/embedding-based) is handled separately by the
canonicalize subpackage.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

from noema.schema import Document, Entity, EntityType, SourceSpan

DEFAULT_MODEL = "en_core_web_sm"

# Labels spaCy emits that aren't useful as knowledge-graph entities.
_IGNORED_LABELS = {"PERCENT", "QUANTITY", "ORDINAL", "CARDINAL", "TIME", "LAW", "LANGUAGE"}


@lru_cache(maxsize=4)
def _load_model(model_name: str = DEFAULT_MODEL) -> Language:
    """Load and cache a spaCy pipeline by name (loading is the slow part)."""
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


def extract_entities(document: Document, model_name: str = DEFAULT_MODEL) -> list[Entity]:
    """Extract entities from a single document's text.

    Exact-text duplicate mentions (e.g. "Google" appearing five times) are
    merged into one Entity with multiple mentions. Returns entities in
    first-mention order.
    """
    nlp = _load_model(model_name)
    doc = nlp(document.text)

    entities_by_name: dict[tuple[str, EntityType], Entity] = {}
    ordered: list[Entity] = []

    for ent in doc.ents:
        if ent.label_ in _IGNORED_LABELS:
            continue

        entity_type = EntityType.from_spacy_label(ent.label_)
        key = (ent.text, entity_type)

        span = SourceSpan(
            doc_id=document.id,
            start_char=ent.start_char,
            end_char=ent.end_char,
            text=ent.text,
        )

        existing = entities_by_name.get(key)
        if existing is not None:
            existing.add_mention(span)
            continue

        entity = Entity(name=ent.text, type=entity_type)
        entity.add_mention(span)
        entities_by_name[key] = entity
        ordered.append(entity)

    return ordered
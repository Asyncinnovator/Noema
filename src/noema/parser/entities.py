"""Text -> Entity extraction using spaCy NER.

Phase 1 scope: run spaCy's NER, map its labels onto EntityType, and collapse
exact-text repeats *within a single document* into one Entity with multiple
mentions. Cross-document canonicalization (fuzzy/embedding-based merging) is
the canonicalize subpackage's job, not this module's -- this module only
produces the raw per-document extraction.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

from noema.schema import Document, Entity, EntityType, SourceSpan

DEFAULT_MODEL = "en_core_web_sm"

# Labels spaCy emits that aren't useful as knowledge-graph entities
# (percentages, ordinals, quantities, etc). Filtered out at extraction time
# rather than mapped to OTHER, since they're rarely graph-worthy nodes.
_IGNORED_LABELS = {"PERCENT", "QUANTITY", "ORDINAL", "CARDINAL", "TIME", "LAW", "LANGUAGE"}


@lru_cache(maxsize=4)
def _load_model(model_name: str = DEFAULT_MODEL) -> Language:
    """Load and cache a spaCy pipeline by name.

    Cached because loading is the expensive part (~100ms+) and callers
    (parser.entities.extract, tests, CLI) may construct many short-lived
    calls in a single process.
    """
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


def extract_entities(document: Document, model_name: str = DEFAULT_MODEL) -> list[Entity]:
    """Extract entities from a single document's text.

    Exact-text duplicate mentions within the document (e.g. "Google"
    appearing five times) are merged into a single Entity with five
    mentions, rather than five separate Entity objects -- this keeps
    within-document dedup trivial and pushes only genuine cross-document /
    fuzzy merging to the canonicalizer.

    Returns entities in first-mention order.
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
"""Coreference resolution: rewrite same-entity mentions to a canonical
surface form before parser.entities/parser.relations see the text.

Uses fastcoref to cluster mentions referring to the same entity within a
document, then rewrites every mention to that cluster's most specific
surface form (e.g. "It" -> "Google"), since pronouns aren't NER-recognizable.
"""

from __future__ import annotations

from functools import lru_cache

from noema.schema import Document

DEFAULT_COREF_MODEL = "biu-nlp/f-coref"

# Leading words marking a mention as generic ("the company", "its") rather
# than a proper name.
_GENERIC_LEAD_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "it", "its", "he", "his", "she", "her", "they", "their", "them",
}


@lru_cache(maxsize=1)
def _load_model(model_name: str = DEFAULT_COREF_MODEL):
    # Imported lazily: fastcoref pulls in transformers/torch, a much
    # heavier install than the rest of the parser needs.
    from fastcoref import FCoref

    return FCoref(model_name_or_path=model_name)


def _looks_like_proper_name(text: str) -> bool:
    """True if `text` looks like a proper name rather than a generic
    reference: no leading pronoun/determiner, starts with a capital letter.
    """
    words = text.split()
    if not words:
        return False
    return words[0].lower() not in _GENERIC_LEAD_WORDS and text[0].isupper()


def _select_canonical_mention(cluster_texts: list[str]) -> str:
    """Pick which surface form in a coreference cluster everything else
    gets rewritten to: prefer proper-name-looking mentions (so "The company"
    doesn't win over "Google" just for being longer), then the longest
    among those as a proxy for "most complete" (e.g. "Google LLC" > "Google").
    """
    proper_candidates = [t for t in cluster_texts if _looks_like_proper_name(t)]
    candidates = proper_candidates if proper_candidates else cluster_texts
    return max(candidates, key=len)


def resolve_coreferences(document: Document, model_name: str = DEFAULT_COREF_MODEL) -> Document:
    """Return a NEW Document with coreferring mentions rewritten to each
    cluster's canonical (most specific) surface form. The input Document is
    never mutated. Replacements happen right-to-left so earlier offsets
    stay valid.
    """
    model = _load_model(model_name)
    prediction = model.predict(texts=[document.text])[0]
    clusters = prediction.get_clusters(as_strings=False)

    if not clusters:
        return document

    text = document.text
    replacements: list[tuple[int, int, str]] = []
    for cluster in clusters:
        cluster_texts = [text[start:end] for start, end in cluster]
        canonical = _select_canonical_mention(cluster_texts)
        for start, end in cluster:
            if text[start:end] != canonical:
                replacements.append((start, end, canonical))

    replacements.sort(key=lambda r: r[0], reverse=True)
    for start, end, replacement in replacements:
        text = text[:start] + replacement + text[end:]

    return Document(id=document.id, text=text, metadata=document.metadata)
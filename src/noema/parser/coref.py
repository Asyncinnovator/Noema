"""Coreference resolution: rewrite same-entity mentions to a canonical
surface form before parser.entities/parser.relations ever see the text.

Phase 5 scope. Uses fastcoref (coreferee was ruled out earlier in this
project for spaCy version incompatibility -- see project notes) to find
clusters of mentions referring to the same real-world entity within a
document, then rewrites every mention in a cluster to that cluster's most
specific surface form.

Why this runs BEFORE parsing rather than after: the rule-based relation
extractor (parser.relations) only looks at each sentence's own dependency
parse in isolation. A sentence like "It is headquartered in Mountain View."
has no NER-recognizable subject at all -- "It" isn't a PERSON/ORG/GPE spaCy
will tag, so parser.entities never creates an entity for it, and the
relation is invisible no matter how good the relation-extraction rules are.
Rewriting "It" to "Google" BEFORE parsing turns this into an ordinary
sentence the existing Phase 1 pipeline already knows how to handle --
coreference resolution buys recall for every downstream stage for free,
without any of them needing to know coreference happened at all.

    "Google was founded in 1998. It is headquartered in Mountain View."
 -> "Google was founded in 1998. Google is headquartered in Mountain View."
"""

from __future__ import annotations

from functools import lru_cache

from noema.schema import Document

DEFAULT_COREF_MODEL = "biu-nlp/f-coref"

# Leading words that mark a mention as a generic reference rather than a
# proper name -- pronouns and determiner-led phrases like "the company",
# "its", "this organization". Used to keep _select_canonical_mention from
# picking a merely-longer generic phrase over the actual proper name.
_GENERIC_LEAD_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "it", "its", "he", "his", "she", "her", "they", "their", "them",
}


@lru_cache(maxsize=1)
def _load_model(model_name: str = DEFAULT_COREF_MODEL):
    # Imported lazily: fastcoref pulls in transformers/torch, which is a
    # meaningfully heavier install than the rest of the parser needs, and
    # this module is the only thing in the codebase that uses it.
    from fastcoref import FCoref

    return FCoref(model_name_or_path=model_name)


def _looks_like_proper_name(text: str) -> bool:
    """True if `text` looks like an actual name rather than a generic
    reference -- doesn't start with a pronoun/determiner, and starts with
    a capital letter.
    """
    words = text.split()
    if not words:
        return False
    return words[0].lower() not in _GENERIC_LEAD_WORDS and text[0].isupper()


def _select_canonical_mention(cluster_texts: list[str]) -> str:
    """Pick which surface form in a coreference cluster everything else
    gets rewritten to.

    Proper-name-looking mentions are preferred over generic ones first --
    this matters because "longest text" alone is a bad proxy for "most
    specific": in a cluster like {"Google", "its", "The company"}, "The
    company" is textually longer than "Google" but far LESS specific, and
    picking it would replace the actual entity name with a vaguer generic
    phrase throughout the document. Among proper-name-looking candidates
    (or, if none exist, among all candidates), the longest is used as a
    cheap proxy for "most complete form" (e.g. "Google LLC" over "Google").
    """
    proper_candidates = [t for t in cluster_texts if _looks_like_proper_name(t)]
    candidates = proper_candidates if proper_candidates else cluster_texts
    return max(candidates, key=len)


def resolve_coreferences(document: Document, model_name: str = DEFAULT_COREF_MODEL) -> Document:
    """Return a NEW Document with coreferring mentions rewritten to each
    cluster's canonical (most specific) surface form. The input Document is
    never mutated -- callers that want the original text for provenance
    (SourceSpan.text should still reflect what was actually written) keep it.

    Replacement happens right-to-left (highest character offset first) so
    that replacing one span doesn't shift the character offsets of spans
    still waiting to be replaced.
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
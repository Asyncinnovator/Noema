"""One-off text compression entry point: noema.compress(text).

Pipeline: parse (entities, relations) -> build graph -> canonicalize
(dedup) -> encode to notation -> decode back to text -> round-trip
validate -> count tokens and compute saved_percent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from noema.canonicalize.dedup import canonicalize
from noema.graph.builder import SemanticGraph
from noema.notation.decoder import decode
from noema.notation.encoder import encode_graph
from noema.parser.entities import DEFAULT_MODEL as SPACY_DEFAULT_MODEL
from noema.parser.entities import extract_entities
from noema.parser.relations import extract_relations
from noema.schema import Document, ParsedDocument
from noema.validator.roundtrip import DEFAULT_PASS_THRESHOLD, validate

DEFAULT_TOKEN_MODEL = "cl100k_base"


@dataclass
class CompressResult:
    """Result of noema.compress(text). `decoded_text` is what `.confidence`
    was scored against, useful for debugging low-confidence results.
    """

    notation: str
    decoded_text: str
    confidence: float
    saved_percent: float
    original_tokens: int
    notation_tokens: int


@lru_cache(maxsize=4)
def _load_token_encoding(model: str = DEFAULT_TOKEN_MODEL) -> tiktoken.Encoding:
    return tiktoken.get_encoding(model)


def count_tokens(text: str, model: str = DEFAULT_TOKEN_MODEL) -> int:
    """Token count for `text` under the given tiktoken encoding."""
    if not text.strip():
        return 0
    encoding = _load_token_encoding(model)
    return len(encoding.encode(text))


def _saved_percent(original_tokens: int, notation_tokens: int) -> float:
    """Percentage of tokens saved by notation vs. the original text.

    Returns 0 if there's nothing to save a percentage of, or if notation
    is empty (an extraction failure, not 100% compression). Otherwise
    clamped to >= 0, since notation can be longer than the original for
    short or fact-sparse text.
    """
    if original_tokens == 0:
        return 0.0
    if notation_tokens == 0:
        return 0.0
    saved = 1.0 - (notation_tokens / original_tokens)
    return max(0.0, saved * 100)


def compress(
    text: str,
    *,
    similarity_threshold: float | None = None,
    validation_threshold: float = DEFAULT_PASS_THRESHOLD,
    token_model: str = DEFAULT_TOKEN_MODEL,
    spacy_model: str = SPACY_DEFAULT_MODEL,
) -> CompressResult:
    """Compress `text` into compact notation, with a round-trip confidence score.

    One-off, single-text compression with no persistent knowledge base
    (see KnowledgeBase.ingest for the multi-document path).
    `similarity_threshold`, if given, is passed through to
    canonicalize.dedup. `spacy_model` selects the spaCy pipeline for NER
    and dependency parsing; relation-extraction patterns are
    English-specific, so non-English models get weaker relation
    extraction than entity extraction.
    """
    document = Document(text=text)
    entities = extract_entities(document, model_name=spacy_model)
    relations = extract_relations(document, entities, model_name=spacy_model)

    graph = SemanticGraph()
    graph.add_document(ParsedDocument(document=document, entities=entities, relations=relations))

    canonicalize_kwargs = {}
    if similarity_threshold is not None:
        canonicalize_kwargs["similarity_threshold"] = similarity_threshold
    canonical_graph = canonicalize(graph, **canonicalize_kwargs)

    notation = encode_graph(canonical_graph)
    decoded_text = decode(notation)

    validation = validate(text, decoded_text, threshold=validation_threshold)

    original_tokens = count_tokens(text, model=token_model)
    notation_tokens = count_tokens(notation, model=token_model)

    return CompressResult(
        notation=notation,
        decoded_text=decoded_text,
        confidence=validation.confidence,
        saved_percent=_saved_percent(original_tokens, notation_tokens),
        original_tokens=original_tokens,
        notation_tokens=notation_tokens,
    )
"""noema.compress(text): the one-off text compression entry point.

Wires together every piece built so far into the pipeline described in the
project plan:

    text -> parse (entities, relations)
         -> build graph
         -> canonicalize (dedup, even within a single text -- "Google" and
            "Google Inc." mentioned in the same paragraph should still
            collapse to one node)
         -> encode to notation
         -> decode back to natural language
         -> round-trip validate (confidence)
         -> count tokens, compute saved_percent

Returns a CompressResult rather than a bare tuple/dict so fields are named
and typed at the call site: `result.notation`, `result.confidence`, etc.,
matching the API shown in the project plan's usage examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from noema.canonicalize.dedup import canonicalize
from noema.graph.builder import SemanticGraph
from noema.notation.decoder import decode
from noema.notation.encoder import encode_graph
from noema.parser.entities import extract_entities
from noema.parser.relations import extract_relations
from noema.schema import Document, ParsedDocument
from noema.validator.roundtrip import DEFAULT_PASS_THRESHOLD, validate

DEFAULT_TOKEN_MODEL = "cl100k_base"


@dataclass
class CompressResult:
    """Result of noema.compress(text).

    `decoded_text` is included even though the project plan's usage example
    only shows `.notation`, `.saved_percent`, and `.confidence` -- it's what
    `.confidence` was actually scored against, and surfacing it costs
    nothing while making a low-confidence result debuggable (see what the
    notation decoded back into, not just the number).
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
    """Token count for `text` under the given tiktoken encoding.

    Empty/whitespace-only text short-circuits to 0 without invoking the
    tokenizer -- consistent with validator.roundtrip's empty-text
    short-circuit, and avoids a tokenizer call for a case with an obvious
    answer.
    """
    if not text.strip():
        return 0
    encoding = _load_token_encoding(model)
    return len(encoding.encode(text))


def _saved_percent(original_tokens: int, notation_tokens: int) -> float:
    """Percentage of tokens saved by notation vs. the original text.

    Two cases are special-cased rather than left to the raw formula:

      - original_tokens == 0: nothing to save a percentage of.
      - notation_tokens == 0 (with nonzero original_tokens): the raw formula
        would report 100% "savings", which is mathematically correct but
        misleading -- an empty notation here means no facts were extractable
        from the text at all, i.e. an extraction failure, not a successful
        full compression. Nothing was actually preserved (confidence for
        this case is separately 0.0, via validator.roundtrip's own
        empty-decoded-text short-circuit), so claiming 100% savings would
        contradict that signal rather than agree with it.

    Otherwise clamped to >= 0: for short or fact-sparse text, notation can
    legitimately end up *longer* than the original in token terms (see
    notation.encoder's docstring on the same tradeoff). Reporting a negative
    "savings" number would be technically accurate but confusing for a field
    called `saved_percent` to return, so 0 (no savings) is reported instead.
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
) -> CompressResult:
    """Compress `text` into compact notation, with a round-trip confidence score.

    This is one-off, single-text compression with no persistent knowledge
    base involved (see KnowledgeBase.ingest, Phase 3, for the multi-document
    incremental-update path). `similarity_threshold` is passed straight
    through to canonicalize.dedup if provided; leaving it as None uses that
    module's own default.
    """
    document = Document(text=text)
    entities = extract_entities(document)
    relations = extract_relations(document, entities)

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
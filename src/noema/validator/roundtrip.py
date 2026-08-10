"""Round-trip validation: does decode(encode(text)) still mean what text meant?

Phase 2 scope: embedding-based cosine similarity between the original source
text and the notation's decoded natural-language rendering, used as a
first-pass confidence signal. This is deliberately a cheap, fast check, not
a strong guarantee -- two sentences can be embedding-similar while one
contradicts or omits something the other asserts (e.g. "Google acquired
YouTube" vs "YouTube acquired Google" embed close together despite being
opposite claims). Phase 5 swaps/augments this for an NLI/entailment model,
which checks actual logical entailment rather than surface similarity.

Per the project plan, facts that fall below the confidence threshold should
be kept in fuller form rather than compacted further -- this module produces
the number that decision is based on; it doesn't make the decision itself
(that's compress()'s job, once it exists).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PASS_THRESHOLD = 0.75


@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@dataclass
class ValidationResult:
    """Round-trip validation outcome for one piece of text.

    `passed` is just `confidence >= threshold`, kept as its own field so
    callers don't have to re-derive the comparison (and so the threshold
    that was actually used travels alongside the result).
    """

    confidence: float
    passed: bool
    threshold: float


def validate(
    original_text: str,
    decoded_text: str,
    threshold: float = DEFAULT_PASS_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> ValidationResult:
    """Compare original source text against its decoded notation rendering.

    Returns confidence in [0, 1] -- cosine similarity is clipped at 0 since
    sentence-transformers similarity can dip slightly negative for wholly
    unrelated text, and a negative "confidence" isn't a meaningful concept
    for callers to reason about.

    Empty decoded text (nothing was extracted/encoded at all) short-circuits
    to 0.0 confidence rather than going through the model: there's no basis
    to claim any meaning was preserved when nothing was produced, no matter
    what a similarity score against an empty string would technically compute
    to.
    """
    if not decoded_text.strip():
        return ValidationResult(confidence=0.0, passed=False, threshold=threshold)

    model = _load_model(model_name)
    embeddings = model.encode([original_text, decoded_text], normalize_embeddings=True)
    similarity = float(embeddings[0] @ embeddings[1])
    confidence = max(0.0, min(1.0, similarity))

    return ValidationResult(confidence=confidence, passed=confidence >= threshold, threshold=threshold)
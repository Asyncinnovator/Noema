"""NLI-based round-trip validation (Phase 5 upgrade).

Swaps validator.roundtrip's embedding-cosine-similarity confidence signal
for an NLI/entailment model, which checks actual logical entailment between
the original text and its decoded notation rendering rather than surface-
level similarity. This closes a real, named gap from Phase 2: two sentences
can be embedding-similar while one contradicts or omits what the other
asserts -- validator.roundtrip's own docstring gives the example "Google
acquired YouTube" vs. "YouTube acquired Google" embedding close together
despite being opposite claims. An NLI model, trained specifically to judge
entailment/contradiction/neutrality, catches exactly this class of error,
which cosine similarity structurally cannot.

This module is additive, not a replacement of validator.roundtrip --
"swap" in the project plan is read here as "have a stronger option
available", not "delete the cheaper one". The embedding-based check is
still the right default for high-volume/low-latency use (no separate model
load, much smaller model), and NLI is the right choice when the stronger
guarantee matters more than speed. Which one compress()/retrieve() use by
default is a call for whoever wires this in, not something this module
decides unilaterally.

Uses cross-encoder/nli-deberta-v3-base (sentence-transformers' CrossEncoder
class), which outputs three raw scores per pair in a FIXED order --
[contradiction, entailment, neutral] -- confirmed from the model's own
card, not assumed. Getting this order wrong would silently invert the
entire confidence signal (reporting high confidence for contradictions),
so it's asserted via the _LABELS constant rather than left as a comment
that could drift from the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_PASS_THRESHOLD = 0.75

# Fixed by the model itself -- see module docstring. Index 0=contradiction,
# 1=entailment, 2=neutral.
_LABELS = ("contradiction", "entailment", "neutral")


@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_NLI_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Converts CrossEncoder's raw per-label scores into probabilities.

    Applied explicitly rather than relying on any activation function the
    library may or may not apply by default -- that behavior has differed
    across sentence-transformers versions, and computing it ourselves makes
    the output deterministic and correct regardless of which version is
    installed.
    """
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


@dataclass
class NLIValidationResult:
    """Round-trip validation outcome via entailment, not similarity.

    `confidence` is the entailment probability specifically (not, say,
    1 - contradiction) -- entailment and "not-contradiction" are different
    things: a neutral (unrelated-but-not-contradicting) pair should score
    low confidence too, since "doesn't contradict" is a much weaker claim
    than "is actually supported by the original text". `contradiction` and
    `neutral` are exposed separately so a caller can distinguish "this is
    wrong" from "this just wasn't confidently verified" when confidence
    comes back low -- those are different failure modes worth different
    handling upstream.
    """

    confidence: float
    passed: bool
    threshold: float
    entailment: float
    neutral: float
    contradiction: float


def validate_nli(
    original_text: str,
    decoded_text: str,
    threshold: float = DEFAULT_PASS_THRESHOLD,
    model_name: str = DEFAULT_NLI_MODEL,
) -> NLIValidationResult:
    """Check whether `original_text` entails `decoded_text`.

    Premise = original_text, hypothesis = decoded_text: does the original
    source actually support what the decoded notation rendering claims?
    This direction matters -- it's checking that decode(encode(text))
    hasn't introduced or distorted a claim the original never made. The
    reverse direction (does decoded fully restate every original detail)
    is a completeness question, not an entailment one, and isn't what this
    function measures.

    Empty decoded text short-circuits to 0.0 confidence without invoking
    the model, matching validator.roundtrip.validate's behavior for the
    same case -- there's no basis to claim entailment when nothing was
    produced to check entailment of.
    """
    if not decoded_text.strip():
        return NLIValidationResult(
            confidence=0.0,
            passed=False,
            threshold=threshold,
            entailment=0.0,
            neutral=0.0,
            contradiction=0.0,
        )

    model = _load_model(model_name)
    raw_scores = model.predict([(original_text, decoded_text)])[0]
    probabilities = _softmax(np.asarray(raw_scores, dtype=float))
    scores = dict(zip(_LABELS, probabilities))
    contradiction, entailment, neutral = (
        scores["contradiction"],
        scores["entailment"],
        scores["neutral"],
    )

    confidence = float(entailment)
    return NLIValidationResult(
        confidence=confidence,
        passed=confidence >= threshold,
        threshold=threshold,
        entailment=float(entailment),
        neutral=float(neutral),
        contradiction=float(contradiction),
    )
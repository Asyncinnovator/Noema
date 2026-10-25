"""Tests for noema.validator.nli.

The CrossEncoder model is mocked (huggingface.co isn't reachable in this
sandbox) with a fake returning controlled raw logits, so softmax/label
behavior can be tested precisely.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from noema.validator import nli


class _FakeCrossEncoder:
    """Returns fixed raw logits for every call, in [contradiction,
    entailment, neutral] order (matching the real model's documented order).
    """

    def __init__(self, raw_scores: list[float]) -> None:
        self._raw_scores = raw_scores
        self.predict_call_count = 0

    def predict(self, pairs):
        self.predict_call_count += 1
        return np.array([self._raw_scores])


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        result = nli._softmax(np.array([1.0, 2.0, 3.0]))
        assert result.sum() == pytest.approx(1.0)

    def test_softmax_preserves_ordering(self):
        result = nli._softmax(np.array([1.0, 5.0, 2.0]))
        assert result[1] > result[2] > result[0]

    def test_softmax_of_equal_logits_is_uniform(self):
        result = nli._softmax(np.array([2.0, 2.0, 2.0]))
        assert result == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_softmax_is_numerically_stable_for_large_logits(self):
        # Would overflow without the max-subtraction trick.
        result = nli._softmax(np.array([1000.0, 1001.0, 999.0]))
        assert not np.isnan(result).any()
        assert result.sum() == pytest.approx(1.0)


class TestValidateNliLabelCases:
    def test_strong_entailment_yields_high_confidence(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 5.0, 0.0])):
            result = nli.validate_nli(
                "Google was founded by Larry Page.", "Larry Page founded Google."
            )
        assert result.confidence > 0.9
        assert result.passed is True

    def test_contradiction_yields_low_confidence(self):
        # The exact scenario this module exists for: a reversed relation.
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([5.0, 0.0, 0.0])):
            result = nli.validate_nli("Google acquired YouTube.", "YouTube acquired Google.")
        assert result.confidence < 0.1
        assert result.passed is False
        assert result.contradiction > result.entailment

    def test_neutral_yields_low_confidence_but_distinguishable_from_contradiction(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 0.0, 5.0])):
            result = nli.validate_nli("Google was founded in 1998.", "Bananas are yellow.")
        assert result.passed is False
        assert result.neutral > result.contradiction
        assert result.neutral > result.entailment

    def test_confidence_is_entailment_probability_not_inverse_contradiction(self):
        # confidence must equal `entailment`, NOT (1 - contradiction) --
        # those differ whenever `neutral` carries meaningful probability mass.
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([1.0, 2.0, 3.0])):
            result = nli.validate_nli("some premise", "some hypothesis")
        assert result.confidence == result.entailment
        assert result.confidence != pytest.approx(1 - result.contradiction)


class TestThreshold:
    def test_passes_when_confidence_meets_threshold(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 5.0, 0.0])):
            result = nli.validate_nli("a", "b", threshold=0.5)
        assert result.passed is True

    def test_fails_when_confidence_below_threshold(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 1.0, 0.0])):
            result = nli.validate_nli("a", "b", threshold=0.99)
        assert result.passed is False

    def test_default_threshold_is_applied(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 5.0, 0.0])):
            result = nli.validate_nli("a", "b")
        assert result.threshold == nli.DEFAULT_PASS_THRESHOLD


class TestEmptyDecodedText:
    def test_empty_decoded_text_short_circuits_to_zero_confidence(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 5.0, 0.0])):
            result = nli.validate_nli("some original text", "")
        assert result.confidence == 0.0
        assert result.passed is False

    def test_empty_decoded_text_does_not_call_the_model(self):
        fake_model = _FakeCrossEncoder([0.0, 5.0, 0.0])
        with patch.object(nli, "_load_model", return_value=fake_model):
            nli.validate_nli("some original text", "")
        assert fake_model.predict_call_count == 0

    def test_whitespace_only_decoded_text_also_short_circuits(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 5.0, 0.0])):
            result = nli.validate_nli("some text", "   \n  ")
        assert result.confidence == 0.0


class TestResultType:
    def test_result_is_nli_validation_result(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([0.0, 5.0, 0.0])):
            result = nli.validate_nli("a", "b")
        assert isinstance(result, nli.NLIValidationResult)

    def test_all_three_scores_sum_to_approximately_one(self):
        with patch.object(nli, "_load_model", return_value=_FakeCrossEncoder([1.0, 2.0, 0.5])):
            result = nli.validate_nli("a", "b")
        assert (result.entailment + result.neutral + result.contradiction) == pytest.approx(1.0)
"""Tests for noema.validator.roundtrip.

The real embedding model (sentence-transformers/all-MiniLM-L6-v2, from
huggingface.co) isn't reachable in this sandbox, so _load_model is mocked
with a deterministic fake. These tests check validate()'s logic --
clipping, thresholding, the empty-text short-circuit -- not real embedding
quality.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import noema.validator.roundtrip as roundtrip


class _FakeEmbeddingModel:
    """Returns unit vectors with a controllable dot product (= cosine
    similarity, since inputs are pre-normalized) between the two texts
    passed to encode(). Only supports the two-text call validate() makes.
    """

    def __init__(self, similarity: float) -> None:
        self._similarity = similarity
        self.encode_call_count = 0

    def encode(self, texts, normalize_embeddings=True):
        self.encode_call_count += 1
        s = self._similarity
        # two unit vectors in 2D with dot product exactly `s`
        return np.array([[1.0, 0.0], [s, (1 - s**2) ** 0.5 if abs(s) <= 1 else 0.0]])


class TestConfidenceScoring:
    def test_identical_similarity_yields_confidence_one(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(1.0)):
            result = roundtrip.validate("Larry Page founded Google.", "Larry Page founded Google.")
        assert result.confidence == pytest.approx(1.0)

    def test_zero_similarity_yields_confidence_zero(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(0.0)):
            result = roundtrip.validate("Larry Page founded Google.", "The weather is nice.")
        assert result.confidence == pytest.approx(0.0)

    def test_mid_similarity_passes_through(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(0.6)):
            result = roundtrip.validate("original", "decoded")
        assert result.confidence == pytest.approx(0.6)

    def test_negative_similarity_is_clipped_to_zero(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(-0.3)):
            result = roundtrip.validate("original", "decoded")
        assert result.confidence == 0.0


class TestPassThreshold:
    def test_confidence_above_threshold_passes(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(0.9)):
            result = roundtrip.validate("a", "b", threshold=0.75)
        assert result.passed is True

    def test_confidence_below_threshold_fails(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(0.5)):
            result = roundtrip.validate("a", "b", threshold=0.75)
        assert result.passed is False

    def test_confidence_exactly_at_threshold_passes(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(0.75)):
            result = roundtrip.validate("a", "b", threshold=0.75)
        assert result.passed is True

    def test_threshold_used_is_returned_on_result(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(0.9)):
            result = roundtrip.validate("a", "b", threshold=0.6)
        assert result.threshold == 0.6

    def test_default_threshold_is_applied_when_not_specified(self):
        with patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel(0.8)):
            result = roundtrip.validate("a", "b")
        assert result.threshold == roundtrip.DEFAULT_PASS_THRESHOLD


class TestEmptyDecodedText:
    def test_empty_decoded_text_short_circuits_to_zero_confidence(self):
        fake_model = _FakeEmbeddingModel(1.0)
        with patch.object(roundtrip, "_load_model", return_value=fake_model):
            result = roundtrip.validate("Larry Page founded Google.", "")
        assert result.confidence == 0.0
        assert result.passed is False

    def test_empty_decoded_text_does_not_call_the_model(self):
        fake_model = _FakeEmbeddingModel(1.0)
        with patch.object(roundtrip, "_load_model", return_value=fake_model):
            roundtrip.validate("Larry Page founded Google.", "")
        assert fake_model.encode_call_count == 0

    def test_whitespace_only_decoded_text_also_short_circuits(self):
        fake_model = _FakeEmbeddingModel(1.0)
        with patch.object(roundtrip, "_load_model", return_value=fake_model):
            result = roundtrip.validate("Larry Page founded Google.", "   \n  ")
        assert result.confidence == 0.0
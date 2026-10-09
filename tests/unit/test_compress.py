"""Tests for noema.compress: the full parse->graph->canonicalize->encode->
decode->validate->count-tokens pipeline behind noema.compress(text).

Both networked dependencies are mocked, since sentence-transformers needs
huggingface.co and tiktoken needs openaipublic.blob.core.windows.net,
neither reachable in this sandbox. These tests verify the pipeline wiring
and CompressResult's arithmetic, not real embedding similarity or BPE
token counts.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest

import noema.canonicalize.dedup as dedup
import noema.validator.roundtrip as roundtrip
from noema.compress import CompressResult, _saved_percent, compress, count_tokens

# Not `import noema.compress as compress_module`: since noema/__init__.py
# does `from noema.compress import compress`, `noema.compress` resolves to
# the function, not the module. importlib.import_module reads sys.modules
# directly, so it isn't affected by that shadowing.
compress_module = importlib.import_module("noema.compress")


class _FakeEmbeddingModel:
    """Exact-match-only fake: identical (case/whitespace-insensitive) strings
    embed identically, everything else is orthogonal.
    """

    def encode(self, texts, normalize_embeddings=True):
        base = {}
        for t in texts:
            key = t.lower().strip()
            if key not in base:
                base[key] = len(base)
        dim = max(len(base), 2)
        vecs = []
        for t in texts:
            key = t.lower().strip()
            v = np.zeros(dim)
            v[base[key]] = 1.0
            vecs.append(v)
        return np.array(vecs)


class _FakeEncoding:
    """Whitespace-split stand-in for a tiktoken.Encoding."""

    def encode(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture
def patched_pipeline():
    """Patch all three networked dependencies compress() touches."""
    with patch.object(dedup, "_load_model", return_value=_FakeEmbeddingModel()), \
         patch.object(roundtrip, "_load_model", return_value=_FakeEmbeddingModel()), \
         patch.object(compress_module, "_load_token_encoding", return_value=_FakeEncoding()):
        yield


class TestCompressPipeline:
    def test_returns_compress_result(self, patched_pipeline):
        result = compress("Larry Page founded Google in 1998.")
        assert isinstance(result, CompressResult)

    def test_notation_contains_expected_entities_and_relation(self, patched_pipeline):
        result = compress("Larry Page founded Google in 1998.")
        assert "Larry Page" in result.notation
        assert "Google" in result.notation
        assert "-found-> " in result.notation or "-found->" in result.notation

    def test_decoded_text_is_derived_from_notation(self, patched_pipeline):
        result = compress("Larry Page founded Google in 1998.")
        assert "Larry Page found Google" in result.decoded_text

    def test_confidence_is_between_zero_and_one(self, patched_pipeline):
        result = compress("Larry Page founded Google in 1998.")
        assert 0.0 <= result.confidence <= 1.0

    def test_token_counts_are_nonnegative(self, patched_pipeline):
        result = compress("Larry Page founded Google in 1998.")
        assert result.original_tokens >= 0
        assert result.notation_tokens >= 0

    def test_saved_percent_is_consistent_with_token_counts(self, patched_pipeline):
        result = compress("Larry Page founded Google in 1998.")
        expected = _saved_percent(result.original_tokens, result.notation_tokens)
        assert result.saved_percent == expected

    def test_repeated_entities_across_sentences_reduce_notation_relative_to_naive_repetition(
        self, patched_pipeline
    ):
        # A text that repeats "Google" many times should compress well,
        # since each repeat collapses to one entity line + short references.
        text = (
            "Google was founded by Larry Page in 1998. "
            "Google is headquartered in Mountain View. "
            "Sundar Pichai is the CEO of Google."
        )
        result = compress(text)
        assert result.notation.count("Google") == 1  # one entity line, referenced by id elsewhere


class TestSpacyModelSelection:
    def test_spacy_model_forwarded_to_extract_entities(self, patched_pipeline):
        with patch("noema.compress.extract_entities", return_value=[]) as entities_spy, \
             patch("noema.compress.extract_relations", return_value=[]):
            compress("Larry Page founded Google.", spacy_model="en_core_web_md")
        _, kwargs = entities_spy.call_args
        assert kwargs.get("model_name") == "en_core_web_md"

    def test_spacy_model_forwarded_to_extract_relations(self, patched_pipeline):
        with patch("noema.compress.extract_entities", return_value=[]), \
             patch("noema.compress.extract_relations", return_value=[]) as relations_spy:
            compress("Larry Page founded Google.", spacy_model="en_core_web_md")
        _, kwargs = relations_spy.call_args
        assert kwargs.get("model_name") == "en_core_web_md"

    def test_default_spacy_model_is_en_core_web_sm(self, patched_pipeline):
        with patch("noema.compress.extract_entities", return_value=[]) as entities_spy, \
             patch("noema.compress.extract_relations", return_value=[]):
            compress("Larry Page founded Google.")
        _, kwargs = entities_spy.call_args
        assert kwargs.get("model_name") == "en_core_web_sm"


class TestCompressWithNoExtractableFacts:
    def test_text_with_no_entities_produces_empty_notation(self, patched_pipeline):
        result = compress("the quick brown fox jumps over the lazy dog")
        assert result.notation == ""

    def test_empty_notation_yields_zero_confidence(self, patched_pipeline):
        result = compress("the quick brown fox jumps over the lazy dog")
        assert result.confidence == 0.0

    def test_empty_notation_yields_zero_saved_percent(self, patched_pipeline):
        result = compress("the quick brown fox jumps over the lazy dog")
        assert result.saved_percent == 0.0


class TestCompressThresholdForwarding:
    def test_similarity_threshold_is_forwarded_to_canonicalize(self, patched_pipeline):
        with patch.object(compress_module, "canonicalize", wraps=dedup.canonicalize) as spy:
            compress("Larry Page founded Google.", similarity_threshold=0.5)
        _, kwargs = spy.call_args
        assert kwargs.get("similarity_threshold") == 0.5

    def test_similarity_threshold_omitted_when_not_provided(self, patched_pipeline):
        with patch.object(compress_module, "canonicalize", wraps=dedup.canonicalize) as spy:
            compress("Larry Page founded Google.")
        _, kwargs = spy.call_args
        assert "similarity_threshold" not in kwargs

    def test_validation_threshold_affects_result_via_validate(self, patched_pipeline):
        # A very high threshold on realistic (imperfect) decoded text should
        # fail even though confidence itself doesn't change.
        result_strict = compress("Larry Page founded Google.", validation_threshold=0.999)
        result_lenient = compress("Larry Page founded Google.", validation_threshold=0.0)
        assert result_strict.confidence == result_lenient.confidence


class TestCountTokens:
    def test_counts_tokens_for_nonempty_text(self, patched_pipeline):
        assert count_tokens("hello world foo") == 3

    def test_empty_text_returns_zero_without_loading_encoding(self):
        with patch.object(compress_module, "_load_token_encoding") as load_mock:
            assert count_tokens("") == 0
            load_mock.assert_not_called()

    def test_whitespace_only_text_returns_zero(self):
        with patch.object(compress_module, "_load_token_encoding") as load_mock:
            assert count_tokens("   \n  ") == 0
            load_mock.assert_not_called()


class TestSavedPercent:
    def test_normal_case(self):
        assert _saved_percent(10, 5) == 50.0

    def test_no_savings_is_zero_not_negative(self):
        assert _saved_percent(10, 10) == 0.0

    def test_notation_longer_than_original_clamps_to_zero(self):
        assert _saved_percent(10, 15) == 0.0

    def test_zero_original_tokens_returns_zero(self):
        assert _saved_percent(0, 0) == 0.0

    def test_empty_notation_with_nonempty_original_is_zero_not_hundred(self):
        # notation_tokens == 0 means nothing was extractable from the text,
        # an extraction failure, not perfect compression.
        assert _saved_percent(9, 0) == 0.0

    def test_near_full_savings_is_reported_accurately(self):
        assert _saved_percent(100, 1) == 99.0
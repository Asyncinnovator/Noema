"""Tests for noema.parser.coref: rewrite coreferring mentions before parsing.

fastcoref's model is mocked (huggingface.co isn't reachable in this
sandbox) with a fake that returns pre-computed character-span clusters,
matching FCoref's real get_clusters(as_strings=False) return shape.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import noema.parser.coref as coref
from noema.schema import Document


def _span(text: str, substring: str) -> tuple[int, int]:
    start = text.index(substring)
    return (start, start + len(substring))


class _FakePrediction:
    def __init__(self, clusters: list[list[tuple[int, int]]]) -> None:
        self._clusters = clusters

    def get_clusters(self, as_strings: bool = False):
        return self._clusters


class _FakeCorefModel:
    def __init__(self, clusters: list[list[tuple[int, int]]]) -> None:
        self._clusters = clusters

    def predict(self, texts):
        return [_FakePrediction(self._clusters)]


class TestResolveCoreferences:
    def test_pronoun_rewritten_to_proper_name(self):
        text = "Google was founded in 1998. It is headquartered in Mountain View."
        clusters = [[_span(text, "Google"), _span(text, "It is")[0], ]]
        # build proper span for "It" not "It is"
        it_start = text.index("It is")
        clusters = [[_span(text, "Google"), (it_start, it_start + 2)]]

        doc = Document(text=text)
        with patch.object(coref, "_load_model", return_value=_FakeCorefModel(clusters)):
            resolved = coref.resolve_coreferences(doc)

        assert "Google is headquartered in Mountain View" in resolved.text
        assert resolved.text.count("It ") == 0

    def test_no_clusters_returns_same_document_unchanged(self):
        doc = Document(text="A simple sentence with nothing to resolve.")
        with patch.object(coref, "_load_model", return_value=_FakeCorefModel([])):
            resolved = coref.resolve_coreferences(doc)
        assert resolved is doc

    def test_document_id_preserved(self):
        text = "Google was founded in 1998. It grew fast."
        it_start = text.index("It grew")
        clusters = [[_span(text, "Google"), (it_start, it_start + 2)]]
        doc = Document(text=text)
        with patch.object(coref, "_load_model", return_value=_FakeCorefModel(clusters)):
            resolved = coref.resolve_coreferences(doc)
        assert resolved.id == doc.id

    def test_metadata_preserved(self):
        text = "Google was founded in 1998. It grew fast."
        it_start = text.index("It grew")
        clusters = [[_span(text, "Google"), (it_start, it_start + 2)]]
        doc = Document(text=text, metadata={"source": "test"})
        with patch.object(coref, "_load_model", return_value=_FakeCorefModel(clusters)):
            resolved = coref.resolve_coreferences(doc)
        assert resolved.metadata == {"source": "test"}

    def test_multiple_clusters_both_resolved_correctly(self):
        text = "Larry Page founded Google. He later became its CEO."
        c1 = [_span(text, "Larry Page"), _span(text, "He later")[0]]
        he_start = text.index("He later")
        c1 = [_span(text, "Larry Page"), (he_start, he_start + 2)]
        its_start = text.index("its CEO")
        c2 = [_span(text, "Google"), (its_start, its_start + 3)]

        doc = Document(text=text)
        with patch.object(coref, "_load_model", return_value=_FakeCorefModel([c1, c2])):
            resolved = coref.resolve_coreferences(doc)

        assert "Larry Page later became Google CEO" in resolved.text

    def test_already_canonical_mention_not_rewritten(self):
        # A mention that already equals the cluster's chosen canonical form
        # should be left untouched (no-op replacement).
        text = "Google was founded in 1998. Google grew fast."
        clusters = [[_span(text, "Google"), (text.rindex("Google"), text.rindex("Google") + 6)]]
        doc = Document(text=text)
        with patch.object(coref, "_load_model", return_value=_FakeCorefModel(clusters)):
            resolved = coref.resolve_coreferences(doc)
        assert resolved.text == text


class TestSelectCanonicalMention:
    def test_prefers_proper_name_over_longer_generic_phrase(self):
        # "The company" is longer than "Google" but far less specific --
        # must not be picked.
        result = coref._select_canonical_mention(["Google", "its", "The company"])
        assert result == "Google"

    def test_prefers_longest_among_proper_names(self):
        result = coref._select_canonical_mention(["Google", "Google LLC"])
        assert result == "Google LLC"

    def test_falls_back_to_longest_when_no_proper_name_present(self):
        # All pronoun/generic -- no proper-name candidate exists at all.
        result = coref._select_canonical_mention(["it", "the company"])
        assert result == "the company"

    def test_single_candidate_returned_as_is(self):
        assert coref._select_canonical_mention(["Google"]) == "Google"


class TestLooksLikeProperName:
    def test_capitalized_non_determiner_is_proper(self):
        assert coref._looks_like_proper_name("Google") is True

    def test_capitalized_multiword_is_proper(self):
        assert coref._looks_like_proper_name("Larry Page") is True

    def test_pronoun_is_not_proper(self):
        assert coref._looks_like_proper_name("it") is False
        assert coref._looks_like_proper_name("its") is False
        assert coref._looks_like_proper_name("he") is False

    def test_determiner_led_phrase_is_not_proper(self):
        assert coref._looks_like_proper_name("The company") is False
        assert coref._looks_like_proper_name("this organization") is False

    def test_empty_string_is_not_proper(self):
        assert coref._looks_like_proper_name("") is False

    def test_lowercase_word_is_not_proper(self):
        assert coref._looks_like_proper_name("google") is False


class TestCorefImprovesRelationRecall:
    def test_pronoun_subject_relation_recovered_after_resolution(self):
        """A relation invisible to the rule-based extractor before
        resolution (subject 'It' isn't an NER entity) becomes extractable
        after resolution.
        """
        from noema.parser.entities import extract_entities
        from noema.parser.relations import extract_relations

        text = "Google was founded in 1998. It is headquartered in Mountain View."
        it_start = text.index("It is")
        clusters = [[_span(text, "Google"), (it_start, it_start + 2)]]

        doc = Document(text=text)
        entities_before = extract_entities(doc)
        relations_before = extract_relations(doc, entities_before)

        with patch.object(coref, "_load_model", return_value=_FakeCorefModel(clusters)):
            resolved = coref.resolve_coreferences(doc)

        entities_after = extract_entities(resolved)
        relations_after = extract_relations(resolved, entities_after)

        assert len(relations_after) > len(relations_before)
"""Tests for noema.parser.transformer_relations.

The REBEL model's generation step is mocked (huggingface.co isn't reachable
in this sandbox), but _extract_triplets is tested against REBEL's real,
documented output format directly -- no mocking needed for pure string
parsing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import noema.parser.transformer_relations as tr
from noema.schema import Document, Entity, EntityType


class TestExtractTriplets:
    def test_single_triplet(self):
        raw = "<s><triplet> Google <subj> Larry Page <obj> founded by</s>"
        triplets = tr._extract_triplets(raw)
        assert triplets == [{"head": "Google", "type": "founded by", "tail": "Larry Page"}]

    def test_multiple_triplets(self):
        raw = (
            "<s><triplet> Google <subj> Larry Page <obj> founded by "
            "<triplet> Google <subj> Sergey Brin <obj> founded by</s>"
        )
        triplets = tr._extract_triplets(raw)
        assert len(triplets) == 2
        assert triplets[0]["tail"] == "Larry Page"
        assert triplets[1]["tail"] == "Sergey Brin"

    def test_empty_string_returns_no_triplets(self):
        assert tr._extract_triplets("") == []

    def test_text_with_no_triplet_markers_returns_no_triplets(self):
        assert tr._extract_triplets("<s>just some plain text</s>") == []

    def test_multiword_head_and_tail(self):
        raw = "<s><triplet> Sundar Pichai <subj> Google <obj> chief executive officer</s>"
        triplets = tr._extract_triplets(raw)
        assert triplets[0]["head"] == "Sundar Pichai"
        assert triplets[0]["type"] == "chief executive officer"

    def test_incomplete_trailing_triplet_still_captured(self):
        # No closing </s> -- the final triplet should still be flushed if
        # subject/relation/object are all non-empty by end of string.
        raw = "<s><triplet> Google <subj> Larry Page <obj> founded by"
        triplets = tr._extract_triplets(raw)
        assert len(triplets) == 1


class TestNormalizePredicate:
    def test_multiword_relation_becomes_snake_case(self):
        assert tr._normalize_predicate("chief executive officer") == "chief_executive_officer"

    def test_single_word_relation(self):
        assert tr._normalize_predicate("founder") == "founder"

    def test_uppercase_input_is_lowercased(self):
        assert tr._normalize_predicate("Headquarters Location") == "headquarters_location"


class TestMatchEntity:
    def test_exact_name_match(self):
        entities = [Entity(name="Google", type=EntityType.ORG)]
        result = tr._match_entity("Google", entities)
        assert result is entities[0]

    def test_case_insensitive_match(self):
        entities = [Entity(name="Google", type=EntityType.ORG)]
        result = tr._match_entity("google", entities)
        assert result is entities[0]

    def test_alias_match(self):
        entity = Entity(name="Google", type=EntityType.ORG, aliases=["Google Inc."])
        result = tr._match_entity("Google Inc.", [entity])
        assert result is entity

    def test_no_match_returns_none(self):
        entities = [Entity(name="Google", type=EntityType.ORG)]
        assert tr._match_entity("Microsoft", entities) is None

    def test_substring_is_not_a_match(self):
        # Deliberately conservative -- a partial/substring match must NOT
        # count, only exact (case-insensitive) equality.
        entities = [Entity(name="Google", type=EntityType.ORG)]
        assert tr._match_entity("Google Search", entities) is None


class TestExtractRelationsTransformer:
    def test_grounded_triplet_becomes_relation(self):
        doc = Document(text="Sundar Pichai leads Google as its chief executive.")
        entities = [
            Entity(name="Sundar Pichai", type=EntityType.PERSON),
            Entity(name="Google", type=EntityType.ORG),
        ]
        fake_raw = "<s><triplet> Sundar Pichai <subj> Google <obj> chief executive officer</s>"
        with patch.object(tr, "_generate_raw_triplet_text", return_value=fake_raw):
            relations = tr.extract_relations_transformer(doc, entities)
        assert len(relations) == 1
        assert relations[0].predicate == "chief_executive_officer"

    def test_transformer_relation_confidence_is_lower_than_default(self):
        doc = Document(text="Sundar Pichai leads Google.")
        entities = [
            Entity(name="Sundar Pichai", type=EntityType.PERSON),
            Entity(name="Google", type=EntityType.ORG),
        ]
        fake_raw = "<s><triplet> Sundar Pichai <subj> Google <obj> leads</s>"
        with patch.object(tr, "_generate_raw_triplet_text", return_value=fake_raw):
            relations = tr.extract_relations_transformer(doc, entities)
        assert relations[0].confidence == tr.TRANSFORMER_RELATION_CONFIDENCE
        assert relations[0].confidence < 1.0

    def test_ungrounded_entity_triplet_is_dropped(self):
        doc = Document(text="Sundar Pichai leads Google.")
        entities = [Entity(name="Sundar Pichai", type=EntityType.PERSON)]  # Google unknown
        fake_raw = "<s><triplet> Sundar Pichai <subj> Google <obj> leads</s>"
        with patch.object(tr, "_generate_raw_triplet_text", return_value=fake_raw):
            relations = tr.extract_relations_transformer(doc, entities)
        assert relations == []

    def test_self_referential_triplet_is_dropped(self):
        doc = Document(text="Google is Google.")
        entities = [Entity(name="Google", type=EntityType.ORG)]
        fake_raw = "<s><triplet> Google <subj> Google <obj> alias of</s>"
        with patch.object(tr, "_generate_raw_triplet_text", return_value=fake_raw):
            relations = tr.extract_relations_transformer(doc, entities)
        assert relations == []

    def test_mention_records_correct_document_id(self):
        doc = Document(text="Sundar Pichai leads Google.")
        entities = [
            Entity(name="Sundar Pichai", type=EntityType.PERSON),
            Entity(name="Google", type=EntityType.ORG),
        ]
        fake_raw = "<s><triplet> Sundar Pichai <subj> Google <obj> leads</s>"
        with patch.object(tr, "_generate_raw_triplet_text", return_value=fake_raw):
            relations = tr.extract_relations_transformer(doc, entities)
        assert relations[0].mentions[0].doc_id == doc.id


class TestExtractRelationsHybrid:
    def test_transformer_not_called_when_rule_based_covers_everything(self):
        doc = Document(text="Larry Page founded Google in 1998. Sundar Pichai is the CEO of Google.")
        from noema.parser.entities import extract_entities

        entities = extract_entities(doc)

        gen_mock = MagicMock()
        with patch.object(tr, "_generate_raw_triplet_text", gen_mock):
            relations = tr.extract_relations_hybrid(doc, entities)

        assert gen_mock.called is False
        assert len(relations) == 2

    def test_transformer_fills_gap_for_missed_sentence(self):
        doc = Document(
            text=(
                "Larry Page founded Google in 1998. "
                "Sundar Pichai, previously a Google executive, is now its chief executive officer."
            )
        )
        from noema.parser.entities import extract_entities

        entities = extract_entities(doc)
        fake_raw = "<s><triplet> Sundar Pichai <subj> Google <obj> chief executive officer</s>"

        with patch.object(tr, "_generate_raw_triplet_text", return_value=fake_raw) as gen_mock:
            relations = tr.extract_relations_hybrid(doc, entities)

        assert gen_mock.call_count == 1
        predicates = {r.predicate for r in relations}
        assert "chief_executive_officer" in predicates

    def test_rule_based_relations_keep_full_confidence_in_hybrid_result(self):
        doc = Document(text="Larry Page founded Google in 1998.")
        from noema.parser.entities import extract_entities

        entities = extract_entities(doc)
        with patch.object(tr, "_generate_raw_triplet_text") as gen_mock:
            relations = tr.extract_relations_hybrid(doc, entities)
        assert gen_mock.called is False
        assert all(r.confidence == 1.0 for r in relations)
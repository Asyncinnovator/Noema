"""Tests for noema.parser.entities: spaCy NER -> Entity objects."""

from __future__ import annotations

from noema.parser.entities import extract_entities
from noema.schema import EntityType


class TestExtractEntities:
    def test_extracts_expected_entity_types(self, make_document):
        doc = make_document("Larry Page founded Google in 1998.")
        entities = extract_entities(doc)
        by_name = {e.name: e for e in entities}

        assert by_name["Larry Page"].type == EntityType.PERSON
        assert by_name["Google"].type == EntityType.ORG
        assert by_name["1998"].type == EntityType.DATE

    def test_repeated_mention_in_same_document_merges_into_one_entity(self, make_document):
        doc = make_document(
            "Google was founded in 1998. Google is headquartered in California."
        )
        entities = extract_entities(doc)
        google_entities = [e for e in entities if e.name == "Google"]

        assert len(google_entities) == 1
        assert len(google_entities[0].mentions) == 2

    def test_mentions_carry_correct_document_id(self, make_document):
        doc = make_document("Google was founded in 1998.")
        entities = extract_entities(doc)
        for entity in entities:
            for mention in entity.mentions:
                assert mention.doc_id == doc.id

    def test_mention_char_offsets_match_source_text(self, make_document):
        doc = make_document("Google was founded in 1998.")
        entities = extract_entities(doc)
        google = next(e for e in entities if e.name == "Google")
        mention = google.mentions[0]

        assert doc.text[mention.start_char : mention.end_char] == "Google"

    def test_ignored_labels_are_filtered_out(self, make_document):
        # "fifty percent" should not surface as a PERCENT entity node.
        doc = make_document("Revenue grew by fifty percent last year.")
        entities = extract_entities(doc)
        assert all(e.type != EntityType.OTHER or "percent" not in e.name.lower() for e in entities)

    def test_no_entities_in_text_returns_empty_list(self, make_document):
        doc = make_document("the quick brown fox jumps over the lazy dog")
        entities = extract_entities(doc)
        assert entities == []

    def test_entities_returned_in_first_mention_order(self, make_document):
        doc = make_document(
            "Larry Page founded Google in 1998, and Sundar Pichai later became its CEO."
        )
        entities = extract_entities(doc)
        names_in_order = [e.name for e in entities]

        assert names_in_order.index("Larry Page") < names_in_order.index("Google")
        assert names_in_order.index("Google") < names_in_order.index("Sundar Pichai")
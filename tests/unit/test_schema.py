"""Tests for noema.schema: the core Entity/Relation/Event/Document models."""

from __future__ import annotations

from noema.schema import (
    Document,
    Entity,
    EntityType,
    Event,
    ParsedDocument,
    Relation,
    SourceSpan,
)


class TestEntityType:
    def test_known_spacy_label_maps_directly(self):
        assert EntityType.from_spacy_label("ORG") == EntityType.ORG
        assert EntityType.from_spacy_label("PERSON") == EntityType.PERSON

    def test_unknown_label_falls_back_to_other(self):
        assert EntityType.from_spacy_label("NORP") == EntityType.OTHER
        assert EntityType.from_spacy_label("totally_made_up") == EntityType.OTHER


class TestEntity:
    def test_default_id_is_generated_and_prefixed(self):
        entity = Entity(name="Google", type=EntityType.ORG)
        assert entity.id.startswith("ent_")

    def test_two_entities_get_distinct_ids(self):
        a = Entity(name="Google", type=EntityType.ORG)
        b = Entity(name="Microsoft", type=EntityType.ORG)
        assert a.id != b.id

    def test_add_mention_appends_mention(self):
        entity = Entity(name="Google", type=EntityType.ORG)
        span = SourceSpan(doc_id="doc_1", start_char=0, end_char=6, text="Google")
        entity.add_mention(span)
        assert entity.mentions == [span]

    def test_add_mention_with_matching_text_does_not_create_alias(self):
        entity = Entity(name="Google", type=EntityType.ORG)
        entity.add_mention(SourceSpan(doc_id="doc_1", start_char=0, end_char=6, text="Google"))
        assert entity.aliases == []

    def test_add_mention_with_different_text_creates_alias(self):
        entity = Entity(name="Google", type=EntityType.ORG)
        entity.add_mention(
            SourceSpan(doc_id="doc_1", start_char=0, end_char=10, text="Google Inc.")
        )
        assert entity.aliases == ["Google Inc."]

    def test_add_mention_does_not_duplicate_existing_alias(self):
        entity = Entity(name="Google", type=EntityType.ORG)
        entity.add_mention(
            SourceSpan(doc_id="doc_1", start_char=0, end_char=10, text="Google Inc.")
        )
        entity.add_mention(
            SourceSpan(doc_id="doc_2", start_char=5, end_char=15, text="Google Inc.")
        )
        assert entity.aliases == ["Google Inc."]
        assert len(entity.mentions) == 2


class TestRelation:
    def test_default_id_is_generated_and_prefixed(self):
        relation = Relation(subject_id="ent_1", predicate="founded", object_id="ent_2")
        assert relation.id.startswith("rel_")

    def test_optional_temporal_fields_default_to_none(self):
        relation = Relation(subject_id="ent_1", predicate="ceo_of", object_id="ent_2")
        assert relation.valid_from is None
        assert relation.valid_to is None


class TestEvent:
    def test_default_id_is_generated_and_prefixed(self):
        event = Event(label="ipo", participant_ids=["ent_1"])
        assert event.id.startswith("evt_")

    def test_participants_default_to_empty_list(self):
        event = Event(label="ipo")
        assert event.participant_ids == []


class TestDocumentAndParsedDocument:
    def test_document_default_id_is_generated_and_prefixed(self):
        doc = Document(text="hello world")
        assert doc.id.startswith("doc_")

    def test_parsed_document_round_trips_through_json(self):
        doc = Document(text="Larry Page founded Google.")
        entity = Entity(name="Google", type=EntityType.ORG)
        parsed = ParsedDocument(document=doc, entities=[entity])

        restored = ParsedDocument.model_validate_json(parsed.model_dump_json())

        assert restored.document.text == doc.text
        assert restored.entities[0].name == "Google"
        assert restored.entities[0].id == entity.id
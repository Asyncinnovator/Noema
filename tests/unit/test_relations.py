"""Tests for noema.parser.relations: dependency-parse -> Relation objects."""

from __future__ import annotations

from noema.parser.entities import extract_entities
from noema.parser.relations import extract_relations


def _relation_triples(document, entities, relations):
    """Helper: (subject_name, predicate, object_name) tuples for easy asserts."""
    name_by_id = {e.id: e.name for e in entities}
    return {(name_by_id[r.subject_id], r.predicate, name_by_id[r.object_id]) for r in relations}


class TestActiveVoiceSVO:
    def test_simple_active_svo(self, make_document):
        doc = make_document("Larry Page founded Google in 1998.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)

        assert ("Larry Page", "found", "Google") in _relation_triples(doc, entities, relations)

    def test_no_relation_extracted_without_dobj_or_attr_or_prep(self, make_document):
        doc = make_document("Larry Page smiled.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)
        assert relations == []


class TestObjectComplementAsRole:
    """'X named/appointed Y as Z' -- an object-complement attached to the
    verb via "as" that carries role information a plain SVO reading would
    drop entirely.
    """

    def test_named_as_role_produces_role_of_relation(self, make_document):
        doc = make_document("Google Inc. named Sundar Pichai as CEO.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)
        triples = _relation_triples(doc, entities, relations)
        assert ("Sundar Pichai", "ceo_of", "Google Inc.") in triples

    def test_plain_svo_relation_still_also_produced(self, make_document):
        # The role-labeled relation is ADDITIONAL, not a replacement --
        # both facts are true and both should be present.
        doc = make_document("Google Inc. named Sundar Pichai as CEO.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)
        triples = _relation_triples(doc, entities, relations)
        assert ("Google Inc.", "name", "Sundar Pichai") in triples

    def test_appointed_as_role_pattern(self, make_document):
        doc = make_document("Microsoft appointed Larry Page as chairman.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)
        triples = _relation_triples(doc, entities, relations)
        assert ("Larry Page", "chairman_of", "Microsoft") in triples

    def test_plain_dobj_without_as_clause_unaffected(self, make_document):
        # No "as X" complement present -- must not spuriously add a role_of
        # relation just because the sentence has a dobj.
        doc = make_document("Larry Page founded Google.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)
        assert all("_of" not in r.predicate for r in relations)


class TestPassiveWithAgent:
    def test_passive_agent_is_reoriented_to_logical_direction(self, make_document):
        doc = make_document("Google was founded by Larry Page in 1998.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)

        triples = _relation_triples(doc, entities, relations)
        assert ("Larry Page", "found", "Google") in triples
        # Google must NOT end up as the subject -- that would invert meaning.
        assert not any(subj == "Google" and pred == "found" for subj, pred, _ in triples)

    def test_conjunct_subjects_in_passive_agent_each_get_a_relation(self, make_document):
        doc = make_document("Google was founded by Larry Page and Sergey Brin in 1998.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)

        triples = _relation_triples(doc, entities, relations)
        assert ("Larry Page", "found", "Google") in triples
        assert ("Sergey Brin", "found", "Google") in triples


class TestPassiveWithPreposition:
    def test_headquartered_in_pattern(self, make_document):
        doc = make_document("Google is headquartered in Mountain View.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)

        triples = _relation_triples(doc, entities, relations)
        assert ("Google", "headquarter_in", "Mountain View") in triples


class TestCopularRoleOf:
    def test_ceo_of_pattern(self, make_document):
        doc = make_document("Sundar Pichai is the CEO of Google.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)

        triples = _relation_triples(doc, entities, relations)
        assert ("Sundar Pichai", "ceo_of", "Google") in triples


class TestRelationMentionProvenance:
    def test_relation_mention_references_correct_document(self, make_document):
        doc = make_document("Larry Page founded Google in 1998.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)

        assert len(relations) == 1
        assert relations[0].mentions[0].doc_id == doc.id

    def test_repeated_relation_across_sentences_merges_mentions(self, make_document):
        doc = make_document(
            "Larry Page founded Google in 1998. Larry Page founded Google, a search company."
        )
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)

        founded = [r for r in relations if r.predicate == "found"]
        assert len(founded) == 1
        assert len(founded[0].mentions) == 2


class TestNoSelfRelations:
    def test_subject_and_object_resolving_to_same_entity_are_skipped(self, make_document):
        # "Google" is both surface positions here after coref-free resolution;
        # rule should never emit a relation from an entity to itself.
        doc = make_document("Google founded Google.")
        entities = extract_entities(doc)
        relations = extract_relations(doc, entities)
        assert all(r.subject_id != r.object_id for r in relations)
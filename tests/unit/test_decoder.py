"""Tests for noema.notation.decoder: notation text -> structured facts -> NL."""

from __future__ import annotations

import pytest

from noema.notation.decoder import decode, parse_notation


class TestParseEntityLines:
    def test_parses_entity_fields(self):
        parsed = parse_notation("#1 Google :ORG")
        assert parsed.entities["1"].name == "Google"
        assert parsed.entities["1"].type == "ORG"

    def test_entity_name_with_spaces(self):
        parsed = parse_notation("#1 Larry Page :PERSON")
        assert parsed.entities["1"].name == "Larry Page"

    def test_malformed_entity_line_raises_with_line_number(self):
        with pytest.raises(ValueError, match="Line 1"):
            parse_notation("#1 Google ORG")  # missing colon


class TestParseRelationLines:
    def test_parses_relation_fields(self):
        notation = "#1 Larry Page :PERSON\n#2 Google :ORG\n#1 -found-> #2"
        parsed = parse_notation(notation)
        assert len(parsed.relations) == 1
        relation = parsed.relations[0]
        assert relation.subject_id == "1"
        assert relation.predicate == "found"
        assert relation.object_id == "2"

    def test_relation_referencing_undeclared_entity_raises(self):
        with pytest.raises(ValueError, match="not declared before this line"):
            parse_notation("#1 Google :ORG\n#1 -found_by-> #2")

    def test_malformed_relation_line_raises(self):
        with pytest.raises(ValueError):
            parse_notation("#1 Google :ORG\n#1 found #2")  # missing -> arrow syntax


class TestParseEventLines:
    def test_parses_event_with_time(self):
        notation = "#1 Google :ORG\n*ipo (#1) @2004-08-19"
        parsed = parse_notation(notation)
        assert len(parsed.events) == 1
        event = parsed.events[0]
        assert event.label == "ipo"
        assert event.participant_ids == ["1"]
        assert event.time == "2004-08-19"

    def test_parses_event_without_time(self):
        notation = "#1 Google :ORG\n*ipo (#1)"
        parsed = parse_notation(notation)
        assert parsed.events[0].time is None

    def test_parses_event_with_multiple_participants(self):
        notation = "#1 A :ORG\n#2 B :ORG\n*merger (#1,#2)"
        parsed = parse_notation(notation)
        assert parsed.events[0].participant_ids == ["1", "2"]

    def test_event_referencing_undeclared_participant_raises(self):
        with pytest.raises(ValueError):
            parse_notation("*ipo (#1)")


class TestParseMiscellaneous:
    def test_blank_lines_are_ignored(self):
        notation = "#1 Google :ORG\n\n#2 Larry Page :PERSON\n\n#2 -found-> #1"
        parsed = parse_notation(notation)
        assert len(parsed.entities) == 2
        assert len(parsed.relations) == 1

    def test_empty_notation_parses_to_empty_result(self):
        parsed = parse_notation("")
        assert parsed.entities == {}
        assert parsed.relations == []
        assert parsed.events == []


class TestDecodeRelations:
    def test_single_word_predicate_renders_as_verb(self):
        notation = "#1 Larry Page :PERSON\n#2 Google :ORG\n#1 -found-> #2"
        assert decode(notation) == "Larry Page found Google."

    def test_role_of_predicate_renders_with_is_the_of(self):
        notation = "#1 Sundar Pichai :PERSON\n#2 Google :ORG\n#1 -ceo_of-> #2"
        assert decode(notation) == "Sundar Pichai is the ceo of Google."

    def test_verb_preposition_predicate(self):
        notation = "#1 Google :ORG\n#2 Mountain View :GPE\n#1 -headquarter_in-> #2"
        assert decode(notation) == "Google is headquarter in Mountain View."

    def test_multiple_relations_produce_multiple_sentences_in_order(self):
        notation = (
            "#1 Larry Page :PERSON\n#2 Google :ORG\n#3 Sundar Pichai :PERSON\n"
            "#1 -found-> #2\n#3 -ceo_of-> #2"
        )
        decoded = decode(notation)
        assert decoded == "Larry Page found Google. Sundar Pichai is the ceo of Google."


class TestDecodeEvents:
    def test_event_with_time(self):
        notation = "#1 Google :ORG\n*ipo (#1) @2004-08-19"
        assert decode(notation) == "Google ipo on 2004-08-19."

    def test_event_without_time(self):
        notation = "#1 Google :ORG\n*ipo (#1)"
        assert decode(notation) == "Google ipo."

    def test_event_with_multiple_participants_joined_with_and(self):
        notation = "#1 Company A :ORG\n#2 Company B :ORG\n*merger (#1,#2)"
        assert decode(notation) == "Company A and Company B merger."


class TestDecodeEmpty:
    def test_empty_notation_decodes_to_empty_string(self):
        assert decode("") == ""

    def test_entities_only_no_relations_decodes_to_empty_string(self):
        assert decode("#1 Google :ORG") == ""
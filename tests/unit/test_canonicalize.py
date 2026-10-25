"""Tests for noema.canonicalize.dedup.

The real embedding model (sentence-transformers/all-MiniLM-L6-v2) needs
huggingface.co, which most sandboxed CI environments can't reach.
`patched_embedding_model` replaces it with a deterministic fake so these
tests check the merge logic (clustering, alias/mention union, relation
remapping) without depending on model downloads or real embedding quality.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from noema.canonicalize import dedup
from noema.graph.builder import SemanticGraph
from noema.schema import Entity, EntityType, Event, Relation


class _FakeEmbeddingModel:
    """Deterministic stand-in for SentenceTransformer.

    Every distinct input string gets its own orthogonal one-hot vector
    (cosine similarity 0), except names supplied in `same_as` groups, which
    share a vector (cosine similarity 1).
    """

    def __init__(self, same_as: list[set[str]] | None = None) -> None:
        self._groups = same_as or []

    def _key(self, text: str) -> str:
        for group in self._groups:
            if text in group:
                return "|".join(sorted(group))
        return text

    def encode(self, texts, normalize_embeddings=True):
        keys = [self._key(t) for t in texts]
        unique_keys = list(dict.fromkeys(keys))
        dim = len(unique_keys)
        vecs = []
        for key in keys:
            v = np.zeros(dim)
            v[unique_keys.index(key)] = 1.0
            vecs.append(v)
        return np.array(vecs)


@pytest.fixture
def patched_embedding_model():
    """Patch dedup._load_model with a fake that has no cross-name similarity.
    Tests needing a near-duplicate pair construct their own
    _FakeEmbeddingModel(same_as=[...]) instead.
    """
    with patch.object(dedup, "_load_model", return_value=_FakeEmbeddingModel()):
        yield


def _entity(name: str, entity_type: EntityType, mentions: int = 1) -> Entity:
    entity = Entity(name=name, type=entity_type)
    for i in range(mentions):
        entity.mentions.append(
            _mention(f"doc_{i}", 0, len(name), name)
        )
    return entity


def _mention(doc_id: str, start: int, end: int, text: str):
    from noema.schema import SourceSpan

    return SourceSpan(doc_id=doc_id, start_char=start, end_char=end, text=text)


class TestExactMatchMerging:
    def test_identical_normalized_names_merge_within_same_type(self, patched_embedding_model):
        graph = SemanticGraph()
        a = _entity("Google", EntityType.ORG)
        b = _entity("Google", EntityType.ORG)
        graph.add_entity(a)
        graph.add_entity(b)

        canon = dedup.canonicalize(graph)

        assert len(canon.entities) == 1
        assert canon.entities[0].mentions.__len__() == 2

    def test_case_and_punctuation_insensitive_exact_match(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("google", EntityType.ORG))
        graph.add_entity(_entity("Google!", EntityType.ORG))

        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 1

    def test_same_name_different_type_does_not_merge(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Washington", EntityType.PERSON))
        graph.add_entity(_entity("Washington", EntityType.GPE))

        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 2

    def test_distinct_names_do_not_merge(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Google", EntityType.ORG))
        graph.add_entity(_entity("Microsoft", EntityType.ORG))

        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 2


class TestEmbeddingSimilarityMerging:
    def test_near_duplicate_names_merge_above_threshold(self):
        graph = SemanticGraph()
        graph.add_entity(_entity("Google", EntityType.ORG))
        graph.add_entity(_entity("Google Inc.", EntityType.ORG))

        fake_model = _FakeEmbeddingModel(same_as=[{"google", "google inc"}])
        with patch.object(dedup, "_load_model", return_value=fake_model):
            canon = dedup.canonicalize(graph, similarity_threshold=0.85)

        assert len(canon.entities) == 1

    def test_merged_entity_keeps_other_name_as_alias(self):
        graph = SemanticGraph()
        graph.add_entity(_entity("Google", EntityType.ORG, mentions=2))
        graph.add_entity(_entity("Google Inc.", EntityType.ORG, mentions=1))

        fake_model = _FakeEmbeddingModel(same_as=[{"google", "google inc"}])
        with patch.object(dedup, "_load_model", return_value=fake_model):
            canon = dedup.canonicalize(graph, similarity_threshold=0.85)

        merged = canon.entities[0]
        assert merged.name == "Google"  # more mentions -> chosen as canonical
        assert "Google Inc." in merged.aliases

    def test_dissimilar_names_do_not_merge(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Google", EntityType.ORG))
        graph.add_entity(_entity("Amazon", EntityType.ORG))

        canon = dedup.canonicalize(graph, similarity_threshold=0.85)
        assert len(canon.entities) == 2


class TestRelationRemapping:
    def test_relations_remapped_to_canonical_ids(self, patched_embedding_model):
        graph = SemanticGraph()
        google_a = _entity("Google", EntityType.ORG)
        google_b = _entity("Google", EntityType.ORG)
        page = _entity("Larry Page", EntityType.PERSON)
        graph.add_entity(google_a)
        graph.add_entity(google_b)
        graph.add_entity(page)
        graph.add_relation(
            Relation(subject_id=page.id, predicate="found", object_id=google_a.id)
        )

        canon = dedup.canonicalize(graph)

        assert len(canon.relations) == 1
        canonical_google_id = next(e.id for e in canon.entities if e.name == "Google")
        assert canon.relations[0].object_id == canonical_google_id

    def test_relations_that_become_self_loops_are_dropped(self, patched_embedding_model):
        graph = SemanticGraph()
        google_a = _entity("Google", EntityType.ORG)
        google_b = _entity("Google", EntityType.ORG)
        graph.add_entity(google_a)
        graph.add_entity(google_b)
        # A spurious relation from one "Google" node to the other -- after
        # merge these are the same entity, so this must not survive as a
        # self-loop.
        graph.add_relation(
            Relation(subject_id=google_a.id, predicate="subsidiary_of", object_id=google_b.id)
        )

        canon = dedup.canonicalize(graph)
        assert canon.relations == []

    def test_duplicate_relations_after_remap_merge_mentions(self, patched_embedding_model):
        graph = SemanticGraph()
        google_a = _entity("Google", EntityType.ORG)
        google_b = _entity("Google", EntityType.ORG)
        page = _entity("Larry Page", EntityType.PERSON)
        graph.add_entity(google_a)
        graph.add_entity(google_b)
        graph.add_entity(page)
        graph.add_relation(
            Relation(subject_id=page.id, predicate="found", object_id=google_a.id)
        )
        graph.add_relation(
            Relation(subject_id=page.id, predicate="found", object_id=google_b.id)
        )

        canon = dedup.canonicalize(graph)
        assert len(canon.relations) == 1


class TestEventRemapping:
    def test_event_participants_remapped_to_canonical_ids(self, patched_embedding_model):
        graph = SemanticGraph()
        google_a = _entity("Google", EntityType.ORG)
        google_b = _entity("Google", EntityType.ORG)
        graph.add_entity(google_a)
        graph.add_entity(google_b)
        graph.add_event(Event(label="ipo", participant_ids=[google_a.id]))

        canon = dedup.canonicalize(graph)

        canonical_google_id = next(e.id for e in canon.entities if e.name == "Google")
        assert canon.events[0].participant_ids == [canonical_google_id]


class TestBusinessSuffixStripping:
    def test_inc_suffix_merges_via_exact_match_alone(self):
        """Cosine similarity between 'google' and 'google inc' can fall
        below the default threshold, so this uses a zero-similarity model
        to prove the merge happens via suffix-stripped exact match.
        """
        graph = SemanticGraph()
        graph.add_entity(_entity("Google", EntityType.ORG))
        graph.add_entity(_entity("Google Inc.", EntityType.ORG))

        class _ZeroSimilarityModel:
            def encode(self, texts, normalize_embeddings=True):
                # Orthogonal vectors for every distinct string.
                base = {t: i for i, t in enumerate(dict.fromkeys(texts))}
                dim = max(len(base), 2)
                vecs = np.zeros((len(texts), dim), dtype="float32")
                for i, t in enumerate(texts):
                    vecs[i, base[t]] = 1.0
                return vecs

        with patch.object(dedup, "_load_model", return_value=_ZeroSimilarityModel()):
            canon = dedup.canonicalize(graph)

        assert len(canon.entities) == 1

    def test_corp_suffix_merges(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Acme", EntityType.ORG))
        graph.add_entity(_entity("Acme Corp", EntityType.ORG))
        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 1

    def test_llc_suffix_merges(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Acme", EntityType.ORG))
        graph.add_entity(_entity("Acme LLC", EntityType.ORG))
        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 1

    def test_suffix_stripping_does_not_cause_false_merge(self, patched_embedding_model):
        # "Acme Inc." and "Beta Inc." must NOT merge just because they
        # share a suffix -- only the base name needs to match.
        graph = SemanticGraph()
        graph.add_entity(_entity("Acme Inc.", EntityType.ORG))
        graph.add_entity(_entity("Beta Inc.", EntityType.ORG))
        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 2

    def test_suffix_stripping_scoped_to_business_words_only(self, patched_embedding_model):
        # A name that merely CONTAINS a word resembling a suffix elsewhere
        # in the name (not as the trailing word) must not be altered.
        assert dedup._normalize_name("Company House") == "company house"


class TestSurnameMatching:
    def test_bare_surname_merges_with_full_name(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Chen", EntityType.PERSON))
        graph.add_entity(_entity("David Chen", EntityType.PERSON))
        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 1

    def test_full_name_stays_canonical_even_with_fewer_mentions(self):
        # A bare surname can accumulate more mentions than the full name
        # that introduced it; canonical selection must still prefer the
        # full name for PERSON entities.
        graph = SemanticGraph()
        chen = _entity("Chen", EntityType.PERSON, mentions=8)
        david_chen = _entity("David Chen", EntityType.PERSON, mentions=3)
        graph.add_entity(chen)
        graph.add_entity(david_chen)

        class _ZeroSimilarityModel:
            def encode(self, texts, normalize_embeddings=True):
                base = {t: i for i, t in enumerate(dict.fromkeys(texts))}
                dim = max(len(base), 2)
                vecs = np.zeros((len(texts), dim), dtype="float32")
                for i, t in enumerate(texts):
                    vecs[i, base[t]] = 1.0
                return vecs

        with patch.object(dedup, "_load_model", return_value=_ZeroSimilarityModel()):
            canon = dedup.canonicalize(graph)

        assert canon.entities[0].name == "David Chen"
        assert "Chen" in canon.entities[0].aliases

    def test_two_different_surnames_do_not_cross_merge(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Chen", EntityType.PERSON))
        graph.add_entity(_entity("Elena Marsh", EntityType.PERSON))
        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 2

    def test_unrelated_single_name_without_a_match_is_untouched(self, patched_embedding_model):
        graph = SemanticGraph()
        graph.add_entity(_entity("Sarah Kim", EntityType.PERSON))
        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 1
        assert canon.entities[0].name == "Sarah Kim"

    def test_surname_matching_not_applied_to_org_type(self, patched_embedding_model):
        # The same word-suffix pattern on an ORG name must NOT trigger a
        # merge -- "Labs" alone should not match "Bright Sensor Labs".
        graph = SemanticGraph()
        graph.add_entity(_entity("Labs", EntityType.ORG))
        graph.add_entity(_entity("Bright Sensor Labs", EntityType.ORG))
        canon = dedup.canonicalize(graph)
        assert len(canon.entities) == 2

    def test_org_canonical_selection_still_prefers_most_mentions(self, patched_embedding_model):
        # The more-attested short brand name ("Google") should stay
        # canonical over the longer legal name ("Google Inc."), unlike PERSON.
        graph = SemanticGraph()
        graph.add_entity(_entity("Google", EntityType.ORG, mentions=5))
        graph.add_entity(_entity("Google Inc.", EntityType.ORG, mentions=1))
        canon = dedup.canonicalize(graph)
        assert canon.entities[0].name == "Google"


class TestIsSurnameMatch:
    def test_matching_surname(self):
        assert dedup._is_surname_match("chen", "david chen") is True

    def test_non_matching_single_word(self):
        assert dedup._is_surname_match("smith", "david chen") is False

    def test_both_multiword_is_not_a_surname_match(self):
        assert dedup._is_surname_match("david chen", "david chen jr") is False

    def test_both_single_word_is_not_a_surname_match(self):
        assert dedup._is_surname_match("chen", "page") is False

    def test_surname_matching_first_word_does_not_match(self):
        # "david" is the FIRST word of "david chen", not the last --
        # must not match (avoids matching on given names too liberally).
        assert dedup._is_surname_match("david", "david chen") is False


class TestFullPipelineIntegration:
    def test_realistic_multi_document_scenario(self, patched_embedding_model):
        """Three documents mention 'Google'; after canonicalization it
        must collapse to one node carrying all cross-document relations.
        """
        google_1 = _entity("Google", EntityType.ORG)
        google_2 = _entity("Google", EntityType.ORG)
        google_3 = _entity("Google", EntityType.ORG)
        page = _entity("Larry Page", EntityType.PERSON)
        pichai = _entity("Sundar Pichai", EntityType.PERSON)
        mtv = _entity("Mountain View", EntityType.GPE)

        graph = SemanticGraph()
        for e in (google_1, page):
            graph.add_entity(e)
        graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google_1.id))

        for e in (google_2, pichai):
            graph.add_entity(e)
        graph.add_relation(
            Relation(subject_id=pichai.id, predicate="ceo_of", object_id=google_2.id)
        )

        for e in (google_3, mtv):
            graph.add_entity(e)
        graph.add_relation(
            Relation(subject_id=google_3.id, predicate="headquarter_in", object_id=mtv.id)
        )

        canon = dedup.canonicalize(graph)

        google_nodes = [e for e in canon.entities if e.name == "Google"]
        assert len(google_nodes) == 1
        google_id = google_nodes[0].id

        neighbor_names = {e.name for e in canon.neighbors(google_id, hops=1)}
        assert neighbor_names == {"Larry Page", "Sundar Pichai", "Mountain View"}
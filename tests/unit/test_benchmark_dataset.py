"""Tests for benchmarks/dataset.py: the synthetic corpus and QA generator."""

from __future__ import annotations

from dataset import generate_corpus


class TestCorpusSize:
    def test_two_documents_per_company(self):
        documents, _ = generate_corpus(10, seed=1)
        assert len(documents) == 20

    def test_four_qa_pairs_per_company(self):
        _, qa_pairs = generate_corpus(10, seed=1)
        assert len(qa_pairs) == 40

    def test_scales_to_hundreds_of_companies(self):
        documents, qa_pairs = generate_corpus(300, seed=1)
        assert len(documents) == 600
        assert len(qa_pairs) == 1200


class TestDeterminism:
    def test_same_seed_produces_identical_corpus(self):
        documents_a, qa_a = generate_corpus(20, seed=7)
        documents_b, qa_b = generate_corpus(20, seed=7)
        assert documents_a == documents_b
        assert qa_a == qa_b

    def test_different_seed_produces_different_corpus(self):
        documents_a, _ = generate_corpus(20, seed=1)
        documents_b, _ = generate_corpus(20, seed=2)
        assert documents_a != documents_b


class TestUniqueness:
    def test_company_names_are_unique_at_moderate_scale(self):
        documents, _ = generate_corpus(200, seed=3)
        profile_docs = documents[0::2]
        company_names = {doc.split(" was founded")[0] for doc in profile_docs}
        assert len(company_names) == 200

    def test_still_produces_requested_count_at_large_scale(self):
        # Exercises the numeric-suffix fallback once the combinatorial name
        # pool would otherwise run out.
        documents, qa_pairs = generate_corpus(5000, seed=4)
        assert len(documents) == 10000
        assert len(qa_pairs) == 20000


class TestQAContent:
    def test_hop_types_are_only_single_or_multi(self):
        _, qa_pairs = generate_corpus(10, seed=1)
        assert {qa.hop_type for qa in qa_pairs} == {"single_hop", "multi_hop"}

    def test_three_single_hop_and_one_multi_hop_per_company(self):
        _, qa_pairs = generate_corpus(1, seed=1)
        hop_types = [qa.hop_type for qa in qa_pairs]
        assert hop_types.count("single_hop") == 3
        assert hop_types.count("multi_hop") == 1

    def test_multi_hop_answer_is_the_acquirers_founder_not_the_targets(self):
        documents, qa_pairs = generate_corpus(1, seed=1)
        profile_doc, acquisition_doc = documents
        multi_hop = next(qa for qa in qa_pairs if qa.hop_type == "multi_hop")

        # The answer must come from the profile document's founder fact...
        assert f"founded by {multi_hop.answer}" in profile_doc
        # ...and the question must reference the acquisition target, which
        # never has its own profile document (ensuring the hop is real).
        target = acquisition_doc.split(" acquired ")[1].split(" in ")[0]
        assert target in multi_hop.question

    def test_single_hop_answers_are_findable_verbatim_in_profile_doc(self):
        documents, qa_pairs = generate_corpus(1, seed=2)
        profile_doc = documents[0]
        for qa in qa_pairs:
            if qa.hop_type == "single_hop":
                assert qa.answer in profile_doc


class TestSubjectField:
    def test_every_pair_names_its_company_as_subject(self):
        _, qa_pairs = generate_corpus(20, seed=1)
        assert all(qa.subject for qa in qa_pairs)

    def test_subject_appears_in_single_hop_question(self):
        _, qa_pairs = generate_corpus(20, seed=1)
        single = [qa for qa in qa_pairs if qa.hop_type == "single_hop"]
        assert all(qa.subject in qa.question for qa in single)

    def test_subject_and_answer_both_appear_in_the_profile_document(self):
        documents, qa_pairs = generate_corpus(20, seed=1)
        for qa in qa_pairs:
            assert any(qa.subject in d and qa.answer in d for d in documents)
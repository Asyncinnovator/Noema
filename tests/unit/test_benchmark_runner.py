"""Tests for benchmarks/runner.py.

noema.KnowledgeBase (spaCy + embeddings) is replaced with a lightweight
fake, and the baselines' embedding model and the tokenizer are mocked for
the same reasons as in test_baselines.py.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from dataset import QAPair
from runner import FULL_CONTEXT, NAIVE_CHUNK_RAG, NOEMA, _score, run_benchmark

compress_module = importlib.import_module("noema.compress")


class _KeywordModel:
    VOCAB: ClassVar[list[str]] = ["ceo", "found", "headquarter", "acquire", "other"]

    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            v = np.zeros(len(self.VOCAB))
            low = t.lower()
            for i, kw in enumerate(self.VOCAB):
                if kw in low:
                    v[i] = 1.0
            if not v.any():
                v[-1] = 1.0
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs, dtype="float32")


class _WhitespaceEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()


class _FakeKnowledgeBase:
    """Returns the documents whose company name appears in the query."""

    instances: ClassVar[list[_FakeKnowledgeBase]] = []

    def __init__(self, embedding_model: str, spacy_model: str) -> None:
        self.docs: list[str] = []
        self.ingest_batch_size = None
        self.retrieve_kwargs: dict = {}
        _FakeKnowledgeBase.instances.append(self)

    def ingest_stream(self, documents, batch_size=100, **_):
        self.ingest_batch_size = batch_size
        self.docs = list(documents)

    def retrieve(self, query, **kwargs):
        self.retrieve_kwargs = kwargs
        hits = [d for d in self.docs if " ".join(d.split()[:2]) in query]
        return SimpleNamespace(text="\n".join(hits))


@pytest.fixture
def patched_run():
    _FakeKnowledgeBase.instances = []
    with (
        patch("runner.KnowledgeBase", _FakeKnowledgeBase),
        patch("baselines._load_model", return_value=_KeywordModel()),
        patch.object(compress_module, "_load_token_encoding", return_value=_WhitespaceEncoding()),
    ):
        yield


class TestScore:
    def _qa(self) -> QAPair:
        return QAPair(
            question="Where is Nova Labs headquartered?",
            answer="Austin",
            hop_type="single_hop",
            subject="Nova Labs",
        )

    def test_requires_subject_alongside_answer(self, patched_run):
        qa = self._qa()
        miss = _score("s", qa, "Delta Labs is headquartered in Austin.", 0.1)
        hit = _score("s", qa, "Nova Labs is headquartered in Austin.", 0.1)
        assert not miss.hit
        assert hit.hit

    def test_counts_tokens_when_not_given(self, patched_run):
        result = _score("s", self._qa(), "one two three", 0.1)
        assert result.token_count == 3

    def test_explicit_token_count_overrides_counting(self, patched_run):
        result = _score("s", self._qa(), "one two three", 0.1, token_count=99)
        assert result.token_count == 99

    def test_carries_question_metadata(self, patched_run):
        result = _score("sys", self._qa(), "x", 0.25)
        assert (result.system, result.hop_type, result.latency_seconds) == (
            "sys",
            "single_hop",
            0.25,
        )


class TestRunBenchmark:
    def test_counts_match_the_generated_corpus(self, patched_run):
        run = run_benchmark(num_companies=5)
        assert run.num_companies == 5
        assert run.document_count == 10
        assert run.qa_count == 20

    def test_every_question_is_run_through_every_system(self, patched_run):
        run = run_benchmark(num_companies=5)
        assert len(run.results) == 20 * 3
        assert {s.system for s in run.summaries} == {NOEMA, NAIVE_CHUNK_RAG, FULL_CONTEXT}
        assert all(s.query_count == 20 for s in run.summaries)

    def test_full_context_always_hits_with_the_whole_corpus_as_cost(self, patched_run):
        run = run_benchmark(num_companies=5)
        full = next(s for s in run.summaries if s.system == FULL_CONTEXT)
        assert full.hit_rate == 1.0
        assert full.multi_hop_hit_rate == 1.0
        # Same token count on every query: the corpus is tokenized once.
        counts = {r.token_count for r in run.results if r.system == FULL_CONTEXT}
        assert len(counts) == 1
        assert counts.pop() > max(
            r.token_count for r in run.results if r.system == NAIVE_CHUNK_RAG
        )

    def test_same_seed_gives_identical_results(self, patched_run):
        first = run_benchmark(num_companies=4, seed=7)
        second = run_benchmark(num_companies=4, seed=7)
        assert [(r.system, r.question, r.hit, r.token_count) for r in first.results] == [
            (r.system, r.question, r.hit, r.token_count) for r in second.results
        ]

    def test_settings_are_forwarded_to_the_knowledge_base(self, patched_run):
        run_benchmark(num_companies=3, batch_size=7, top_k=2, hops=2, max_tokens=33)
        kb = _FakeKnowledgeBase.instances[0]
        assert kb.ingest_batch_size == 7
        assert kb.retrieve_kwargs == {"top_k": 2, "hops": 2, "max_tokens": 33}
        assert len(kb.docs) == 6

    def test_progress_messages_are_reported(self, patched_run):
        messages: list[str] = []
        run_benchmark(num_companies=3, on_progress=messages.append)
        assert messages[0].startswith("Generated 6 documents")
        assert any("Scored" in m for m in messages)

    def test_runs_without_a_progress_callback(self, patched_run):
        assert run_benchmark(num_companies=2).results
"""Orchestrates a full benchmark run: generates a synthetic corpus, builds
noema.KnowledgeBase plus the two baselines (plain-chunk RAG and full-context
stuffing) from it, runs every question through all three, and collects
results.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from baselines import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    FullContextBaseline,
    NaiveChunkRAG,
)
from dataset import QAPair, generate_corpus
from metrics import QueryResult, SystemSummary, answer_hit, summarize

from noema.compress import count_tokens
from noema.parser.entities import DEFAULT_MODEL as DEFAULT_SPACY_MODEL
from noema.retrieval.assemble import DEFAULT_MAX_TOKENS
from noema.storage.knowledge_base import DEFAULT_BATCH_SIZE, KnowledgeBase

NOEMA = "noema"
NAIVE_CHUNK_RAG = "naive_chunk_rag"
FULL_CONTEXT = "full_context"


@dataclass
class BenchmarkRun:
    """Everything produced by one run_benchmark() call."""

    num_companies: int
    document_count: int
    qa_count: int
    results: list[QueryResult] = field(default_factory=list)
    summaries: list[SystemSummary] = field(default_factory=list)


def _time_retrieve(fn, question: str) -> tuple[str, float]:
    start = time.perf_counter()
    text = fn(question)
    elapsed = time.perf_counter() - start
    return text, elapsed


def _score(
    system: str,
    qa: QAPair,
    text: str,
    elapsed: float,
    token_count: int | None = None,
) -> QueryResult:
    """Score one retrieval. Pass `token_count` when the text is reused
    across queries (full-context) to avoid re-tokenizing it every time.
    """
    return QueryResult(
        system=system,
        question=qa.question,
        answer=qa.answer,
        hop_type=qa.hop_type,
        retrieved_text=text,
        token_count=count_tokens(text) if token_count is None else token_count,
        hit=answer_hit(text, qa.answer, qa.subject),
        latency_seconds=elapsed,
    )


def run_benchmark(
    num_companies: int = 50,
    seed: int = 42,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    hops: int = 1,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    spacy_model: str = DEFAULT_SPACY_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    on_progress: Callable[[str], None] | None = None,
) -> BenchmarkRun:
    """Generate a `num_companies`-company corpus and run every question
    through noema.KnowledgeBase and both baselines, built from the exact
    same documents so the comparison is apples-to-apples.
    """

    def report(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    documents, qa_pairs = generate_corpus(num_companies, seed=seed)
    report(f"Generated {len(documents)} documents, {len(qa_pairs)} questions")

    report("Building noema.KnowledgeBase...")
    kb = KnowledgeBase(embedding_model=embedding_model, spacy_model=spacy_model)
    kb.ingest_stream(documents, batch_size=batch_size)

    report("Building NaiveChunkRAG baseline...")
    naive_rag = NaiveChunkRAG(chunk_tokens=chunk_tokens, embedding_model=embedding_model)
    naive_rag.build(documents)

    report("Building FullContextBaseline...")
    full_context = FullContextBaseline()
    full_context.build(documents)
    full_context_tokens = count_tokens(full_context.retrieve(""))

    def noema_retrieve(question: str) -> str:
        return kb.retrieve(question, top_k=top_k, hops=hops, max_tokens=max_tokens).text

    def naive_retrieve(question: str) -> str:
        return naive_rag.retrieve(question, max_tokens=max_tokens, top_k=top_k)

    def full_context_retrieve(question: str) -> str:
        return full_context.retrieve(question)

    # Only the full-context text is identical for every query.
    fixed_token_counts = {FULL_CONTEXT: full_context_tokens}

    systems = [
        (NOEMA, noema_retrieve),
        (NAIVE_CHUNK_RAG, naive_retrieve),
        (FULL_CONTEXT, full_context_retrieve),
    ]

    results: list[QueryResult] = []
    for i, qa in enumerate(qa_pairs):
        for system_name, retrieve_fn in systems:
            text, elapsed = _time_retrieve(retrieve_fn, qa.question)
            results.append(
                _score(system_name, qa, text, elapsed, fixed_token_counts.get(system_name))
            )
        if on_progress is not None and (i + 1) % max(1, len(qa_pairs) // 10) == 0:
            report(f"Scored {i + 1}/{len(qa_pairs)} questions")

    return BenchmarkRun(
        num_companies=num_companies,
        document_count=len(documents),
        qa_count=len(qa_pairs),
        results=results,
        summaries=summarize(results),
    )
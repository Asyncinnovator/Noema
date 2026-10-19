"""Per-query results and aggregate scoring for a benchmark run.

Retrieval accuracy is scored by answer presence: a query "hits" if its
known ground-truth answer string appears in the retrieved text, together
with the question's subject entity when one is given. Requiring the subject
matters because many answers (e.g. cities) are shared by several companies,
so the answer alone could match text about a different company. This is a
simple, automatic proxy that needs no external grading model -- it works
because benchmarks/dataset.py generates every question alongside its exact
answer and subject strings, not as a general-purpose QA metric.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryResult:
    """One (system, question) pair's outcome."""

    system: str
    question: str
    answer: str
    hop_type: str  # "single_hop" or "multi_hop"
    retrieved_text: str
    token_count: int
    hit: bool
    latency_seconds: float


@dataclass
class SystemSummary:
    """Aggregate stats for one system across an entire benchmark run."""

    system: str
    query_count: int
    avg_tokens: float
    hit_rate: float
    single_hop_hit_rate: float
    multi_hop_hit_rate: float
    avg_latency_seconds: float


def answer_hit(retrieved_text: str, answer: str, subject: str = "") -> bool:
    """Whether `answer` (and `subject`, if non-empty) appear in
    `retrieved_text`, case-insensitively.
    """
    text = retrieved_text.lower()
    if answer.strip().lower() not in text:
        return False
    subject = subject.strip().lower()
    return not subject or subject in text


def _hit_rate(results: list[QueryResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit) / len(results)


def summarize(results: list[QueryResult]) -> list[SystemSummary]:
    """Group `results` by system and compute aggregate stats for each,
    in first-seen system order.
    """
    systems: list[str] = []
    by_system: dict[str, list[QueryResult]] = {}
    for result in results:
        if result.system not in by_system:
            systems.append(result.system)
            by_system[result.system] = []
        by_system[result.system].append(result)

    summaries = []
    for system in systems:
        system_results = by_system[system]
        single_hop = [r for r in system_results if r.hop_type == "single_hop"]
        multi_hop = [r for r in system_results if r.hop_type == "multi_hop"]
        summaries.append(
            SystemSummary(
                system=system,
                query_count=len(system_results),
                avg_tokens=sum(r.token_count for r in system_results) / len(system_results),
                hit_rate=_hit_rate(system_results),
                single_hop_hit_rate=_hit_rate(single_hop),
                multi_hop_hit_rate=_hit_rate(multi_hop),
                avg_latency_seconds=sum(r.latency_seconds for r in system_results)
                / len(system_results),
            )
        )
    return summaries
"""Tests for benchmarks/metrics.py: answer_hit and summarize."""

from __future__ import annotations

import pytest
from metrics import QueryResult, answer_hit, summarize


def _result(
    system: str = "a",
    hop_type: str = "single_hop",
    hit: bool = True,
    tokens: int = 10,
    latency: float = 0.1,
) -> QueryResult:
    return QueryResult(
        system=system,
        question="q",
        answer="x",
        hop_type=hop_type,
        retrieved_text="",
        token_count=tokens,
        hit=hit,
        latency_seconds=latency,
    )


class TestAnswerHit:
    def test_answer_present(self):
        assert answer_hit("Nova Robotics is headquartered in Austin.", "Austin")

    def test_answer_absent(self):
        assert not answer_hit("Nova Robotics is headquartered in Denver.", "Austin")

    def test_case_insensitive(self):
        assert answer_hit("elena marsh founded it", "Elena Marsh")

    def test_answer_whitespace_is_stripped(self):
        assert answer_hit("founded by Elena Marsh", "  Elena Marsh ")

    def test_empty_text_never_hits(self):
        assert not answer_hit("", "Austin")

    def test_subject_and_answer_both_present(self):
        text = "Nova Robotics is headquartered in Austin."
        assert answer_hit(text, "Austin", subject="Nova Robotics")

    def test_answer_without_subject_is_a_miss(self):
        # Another company's headquarters sentence contains the same city.
        text = "Delta Labs is headquartered in Austin."
        assert not answer_hit(text, "Austin", subject="Nova Robotics")

    def test_subject_without_answer_is_a_miss(self):
        text = "Nova Robotics builds warehouse robots."
        assert not answer_hit(text, "Austin", subject="Nova Robotics")

    def test_subject_match_is_case_insensitive(self):
        assert answer_hit("nova robotics: austin", "Austin", subject="Nova Robotics")

    def test_empty_subject_is_ignored(self):
        assert answer_hit("headquartered in Austin", "Austin", subject="")

    def test_subject_defaults_to_not_required(self):
        assert answer_hit("headquartered in Austin", "Austin")


class TestSummarize:
    def test_empty_results_give_no_summaries(self):
        assert summarize([]) == []

    def test_groups_by_system_in_first_seen_order(self):
        results = [_result("b"), _result("a"), _result("b")]
        summaries = summarize(results)
        assert [s.system for s in summaries] == ["b", "a"]
        assert [s.query_count for s in summaries] == [2, 1]

    def test_overall_hit_rate(self):
        results = [_result(hit=True), _result(hit=False), _result(hit=True), _result(hit=True)]
        assert summarize(results)[0].hit_rate == pytest.approx(0.75)

    def test_hit_rates_split_by_hop_type(self):
        results = [
            _result(hop_type="single_hop", hit=True),
            _result(hop_type="single_hop", hit=True),
            _result(hop_type="multi_hop", hit=False),
            _result(hop_type="multi_hop", hit=True),
        ]
        summary = summarize(results)[0]
        assert summary.single_hop_hit_rate == pytest.approx(1.0)
        assert summary.multi_hop_hit_rate == pytest.approx(0.5)

    def test_missing_hop_type_gives_zero_rate_not_error(self):
        summary = summarize([_result(hop_type="single_hop")])[0]
        assert summary.multi_hop_hit_rate == 0.0

    def test_average_tokens_and_latency(self):
        results = [_result(tokens=10, latency=0.2), _result(tokens=30, latency=0.4)]
        summary = summarize(results)[0]
        assert summary.avg_tokens == pytest.approx(20.0)
        assert summary.avg_latency_seconds == pytest.approx(0.3)

    def test_systems_are_scored_independently(self):
        results = [_result("a", hit=True), _result("b", hit=False)]
        by_name = {s.system: s for s in summarize(results)}
        assert by_name["a"].hit_rate == 1.0
        assert by_name["b"].hit_rate == 0.0
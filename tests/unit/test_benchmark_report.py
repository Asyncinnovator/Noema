"""Tests for benchmarks/report.py and the benchmarks/run_benchmark.py CLI."""

from __future__ import annotations

import csv
import io
import json
from unittest.mock import patch

import pytest
import run_benchmark as script
from metrics import QueryResult, summarize
from report import MAX_SAVED_TEXT_CHARS, print_summary_table, save_results
from rich.console import Console
from runner import BenchmarkRun
from typer.testing import CliRunner


def _result(system: str, tokens: int, hit: bool = True, text: str = "t") -> QueryResult:
    return QueryResult(
        system=system,
        question="q?",
        answer="a",
        hop_type="single_hop",
        retrieved_text=text,
        token_count=tokens,
        hit=hit,
        latency_seconds=0.002,
    )


def _run(results: list[QueryResult]) -> BenchmarkRun:
    return BenchmarkRun(
        num_companies=3,
        document_count=6,
        qa_count=12,
        results=results,
        summaries=summarize(results),
    )


def _render(run: BenchmarkRun) -> str:
    buffer = io.StringIO()
    print_summary_table(run, console=Console(file=buffer, width=200))
    return buffer.getvalue()


class TestSummaryTable:
    def test_lists_every_system_and_the_run_size(self):
        output = _render(_run([_result("noema", 10), _result("full_context", 100)]))
        assert "noema" in output
        assert "full_context" in output
        assert "3 companies" in output
        assert "12 questions" in output

    def test_reports_fewer_tokens_than_a_larger_baseline(self):
        output = _render(_run([_result("noema", 25), _result("full_context", 100)]))
        assert "75% fewer w/ noema" in output

    def test_reports_more_tokens_than_a_smaller_baseline(self):
        output = _render(_run([_result("noema", 150), _result("naive_chunk_rag", 100)]))
        assert "50% more w/ noema" in output

    def test_no_comparison_note_without_a_noema_row(self):
        output = _render(_run([_result("naive_chunk_rag", 100)]))
        assert "w/ noema" not in output

    def test_hit_rates_are_shown_as_percentages(self):
        output = _render(_run([_result("noema", 10, hit=True), _result("noema", 10, hit=False)]))
        assert "50.0%" in output


class TestSaveResults:
    def test_writes_csv_and_json(self, tmp_path):
        run = _run([_result("noema", 10), _result("full_context", 100)])
        csv_path, json_path = save_results(run, str(tmp_path / "out"))
        assert csv_path.exists()
        assert json_path.exists()

    def test_csv_has_one_row_per_result(self, tmp_path):
        run = _run([_result("noema", 10), _result("naive_chunk_rag", 20)])
        csv_path, _ = save_results(run, str(tmp_path))
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["system"] for r in rows] == ["noema", "naive_chunk_rag"]
        assert rows[0]["token_count"] == "10"

    def test_json_includes_summary_and_results(self, tmp_path):
        run = _run([_result("noema", 10)])
        _, json_path = save_results(run, str(tmp_path))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["qa_count"] == 12
        assert payload["summaries"][0]["system"] == "noema"
        assert len(payload["results"]) == 1

    def test_long_retrieved_text_is_truncated(self, tmp_path):
        long_text = "x" * (MAX_SAVED_TEXT_CHARS * 10)
        run = _run([_result("full_context", 999, text=long_text)])
        csv_path, json_path = save_results(run, str(tmp_path))
        with csv_path.open(encoding="utf-8") as f:
            saved_csv = next(csv.DictReader(f))["retrieved_text"]
        saved_json = json.loads(json_path.read_text(encoding="utf-8"))["results"][0]
        for saved in (saved_csv, saved_json["retrieved_text"]):
            assert len(saved) == MAX_SAVED_TEXT_CHARS + 3
            assert saved.endswith("...")

    def test_short_retrieved_text_is_kept_whole(self, tmp_path):
        run = _run([_result("noema", 5, text="short")])
        _, json_path = save_results(run, str(tmp_path))
        saved = json.loads(json_path.read_text(encoding="utf-8"))["results"][0]
        assert saved["retrieved_text"] == "short"

    def test_does_not_modify_the_run_itself(self, tmp_path):
        long_text = "x" * (MAX_SAVED_TEXT_CHARS * 2)
        run = _run([_result("full_context", 1, text=long_text)])
        save_results(run, str(tmp_path))
        assert run.results[0].retrieved_text == long_text

    def test_creates_nested_output_directory(self, tmp_path):
        target = tmp_path / "a" / "b"
        save_results(_run([_result("noema", 1)]), str(target))
        assert target.is_dir()


class TestRunBenchmarkScript:
    @pytest.fixture
    def fake_run(self):
        run = _run([_result("noema", 10), _result("full_context", 100)])
        with patch.object(script, "run_benchmark", return_value=run) as mock:
            yield mock

    def test_runs_and_saves_results(self, fake_run, tmp_path):
        out = tmp_path / "res"
        result = CliRunner().invoke(script.app, ["-n", "7", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "results.csv").exists()
        assert (out / "results.json").exists()
        assert fake_run.call_args.kwargs["num_companies"] == 7

    def test_options_are_forwarded(self, fake_run, tmp_path):
        args = [
            "--seed", "3", "--top-k", "2", "--hops", "2", "--chunk-tokens", "40",
            "--batch-size", "9", "--max-tokens", "123", "--embedding-model", "m",
            "-o", str(tmp_path),
        ]  # fmt: skip
        result = CliRunner().invoke(script.app, args)
        assert result.exit_code == 0, result.output
        kwargs = fake_run.call_args.kwargs
        assert kwargs["seed"] == 3
        assert kwargs["top_k"] == 2
        assert kwargs["hops"] == 2
        assert kwargs["chunk_tokens"] == 40
        assert kwargs["batch_size"] == 9
        assert kwargs["max_tokens"] == 123
        assert kwargs["embedding_model"] == "m"

    def test_prints_the_summary_table(self, fake_run, tmp_path):
        result = CliRunner().invoke(script.app, ["-o", str(tmp_path)])
        assert "noema benchmark" in result.output
        assert "Saved" in result.output
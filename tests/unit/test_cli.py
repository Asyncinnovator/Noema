"""Tests for noema.cli: the compress / build-graph / query / inspect /
benchmark subcommands, run through typer's CliRunner.

Embeddings and the tokenizer are mocked for the same reason as elsewhere
in this suite (huggingface.co/openaipublic.blob.core.windows.net aren't
reachable in this sandbox).
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from noema.canonicalize import dedup
from noema.cli import app
from noema.retrieval import vector_search
from noema.storage import bundle
from noema.validator import roundtrip

compress_module = importlib.import_module("noema.compress")


class _ExactMatchModel:
    """Orthogonal per distinct (normalized) string -- no accidental merges."""

    def encode(self, texts, normalize_embeddings=True):
        base = {}
        for t in texts:
            key = t.lower().strip()
            if key not in base:
                base[key] = len(base)
        dim = max(len(base), 2)
        vecs = np.zeros((len(texts), dim), dtype="float32")
        for i, t in enumerate(texts):
            vecs[i, base[t.lower().strip()]] = 1.0
        return vecs


class _FakeEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture
def patched_pipeline():
    with (
        patch.object(dedup, "_load_model", return_value=_ExactMatchModel()),
        patch.object(roundtrip, "_load_model", return_value=_ExactMatchModel()),
        patch.object(bundle, "_load_model", return_value=_ExactMatchModel()),
        patch.object(vector_search, "_load_model", return_value=_ExactMatchModel()),
        patch.object(compress_module, "_load_token_encoding", return_value=_FakeEncoding()),
    ):
        yield


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


_DOC_1 = "Larry Page founded Google in 1998."
_DOC_2 = "Sundar Pichai is the CEO of Google."


@pytest.fixture
def two_doc_files(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc1.write_text(_DOC_1, encoding="utf-8")
    doc2.write_text(_DOC_2, encoding="utf-8")
    return doc1, doc2


@pytest.fixture
def built_bundle(runner, patched_pipeline, two_doc_files, tmp_path):
    doc1, doc2 = two_doc_files
    bundle_path = tmp_path / "kb.noema"
    result = runner.invoke(
        app, ["build-graph", str(doc1), str(doc2), "--output", str(bundle_path)]
    )
    assert result.exit_code == 0, result.output
    return bundle_path


class TestCompress:
    def test_exits_zero_on_success(self, runner, patched_pipeline):
        result = runner.invoke(app, ["compress", _DOC_1])
        assert result.exit_code == 0, result.output

    def test_output_contains_notation_and_stats(self, runner, patched_pipeline):
        result = runner.invoke(app, ["compress", _DOC_1])
        assert "Google" in result.output
        assert "original tokens" in result.output
        assert "notation tokens" in result.output

    def test_reads_from_file_with_flag(self, runner, patched_pipeline, tmp_path):
        doc = tmp_path / "input.txt"
        doc.write_text(_DOC_1, encoding="utf-8")
        result = runner.invoke(app, ["compress", "--file", str(doc)])
        assert result.exit_code == 0, result.output
        assert "Google" in result.output

    def test_neither_text_nor_file_errors(self, runner, patched_pipeline):
        result = runner.invoke(app, ["compress"])
        assert result.exit_code != 0


class TestBuildGraph:
    def test_exits_zero_on_success(self, runner, patched_pipeline, two_doc_files, tmp_path):
        doc1, doc2 = two_doc_files
        bundle_path = tmp_path / "kb.noema"
        result = runner.invoke(
            app, ["build-graph", str(doc1), str(doc2), "--output", str(bundle_path)]
        )
        assert result.exit_code == 0, result.output

    def test_bundle_file_is_created(self, built_bundle):
        assert built_bundle.exists()

    def test_reports_entity_and_relation_counts(self, runner, patched_pipeline, two_doc_files, tmp_path):
        doc1, doc2 = two_doc_files
        bundle_path = tmp_path / "kb.noema"
        result = runner.invoke(
            app, ["build-graph", str(doc1), str(doc2), "--output", str(bundle_path)]
        )
        assert "entities" in result.output
        assert "relations" in result.output

    def test_missing_input_file_errors(self, runner, patched_pipeline, tmp_path):
        bundle_path = tmp_path / "kb.noema"
        result = runner.invoke(
            app, ["build-graph", str(tmp_path / "does_not_exist.txt"), "--output", str(bundle_path)]
        )
        assert result.exit_code != 0

    def test_batch_size_is_respected(self, runner, patched_pipeline, two_doc_files, tmp_path):
        doc1, doc2 = two_doc_files
        bundle_path = tmp_path / "kb.noema"
        result = runner.invoke(
            app,
            [
                "build-graph",
                str(doc1),
                str(doc2),
                "--output",
                str(bundle_path),
                "--batch-size",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert bundle_path.exists()


class TestInspect:
    def test_exits_zero_on_success(self, runner, built_bundle):
        result = runner.invoke(app, ["inspect", str(built_bundle)])
        assert result.exit_code == 0, result.output

    def test_shows_entity_and_relation_counts(self, runner, built_bundle):
        result = runner.invoke(app, ["inspect", str(built_bundle)])
        assert "entities" in result.output
        assert "relations" in result.output

    def test_shows_embedding_and_spacy_model(self, runner, built_bundle):
        result = runner.invoke(app, ["inspect", str(built_bundle)])
        assert "all-MiniLM-L6-v2" in result.output
        assert "en_core_web_sm" in result.output

    def test_missing_bundle_errors(self, runner, tmp_path):
        result = runner.invoke(app, ["inspect", str(tmp_path / "missing.noema")])
        assert result.exit_code != 0


class TestQuery:
    def test_exits_zero_on_success(self, runner, patched_pipeline, built_bundle):
        result = runner.invoke(app, ["query", str(built_bundle), "Who is the CEO of Google?"])
        assert result.exit_code == 0, result.output

    def test_output_contains_retrieved_notation(self, runner, patched_pipeline, built_bundle):
        result = runner.invoke(app, ["query", str(built_bundle), "Who is the CEO of Google?"])
        assert "Google" in result.output

    def test_output_reports_token_count(self, runner, patched_pipeline, built_bundle):
        result = runner.invoke(app, ["query", str(built_bundle), "Who is the CEO of Google?"])
        assert "tokens" in result.output

    def test_natural_language_format(self, runner, patched_pipeline, built_bundle):
        result = runner.invoke(
            app,
            ["query", str(built_bundle), "Who is the CEO of Google?", "--format", "natural_language"],
        )
        assert result.exit_code == 0, result.output

    def test_invalid_format_errors(self, runner, patched_pipeline, built_bundle):
        result = runner.invoke(
            app, ["query", str(built_bundle), "Who is the CEO of Google?", "--format", "bogus"]
        )
        assert result.exit_code != 0


class TestBenchmark:
    def test_exits_zero_on_success(self, runner, patched_pipeline, built_bundle, two_doc_files):
        doc1, doc2 = two_doc_files
        result = runner.invoke(
            app,
            [
                "benchmark",
                str(built_bundle),
                "-d",
                str(doc1),
                "-d",
                str(doc2),
                "-q",
                "Who is the CEO of Google?",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_output_has_a_row_per_query(self, runner, patched_pipeline, built_bundle, two_doc_files):
        doc1, _ = two_doc_files
        result = runner.invoke(
            app,
            [
                "benchmark",
                str(built_bundle),
                "-d",
                str(doc1),
                "-q",
                "Who is the CEO of Google?",
                "-q",
                "Who founded Google?",
            ],
        )
        assert result.output.count("Google?") == 2

    def test_reports_average_reduction(self, runner, patched_pipeline, built_bundle, two_doc_files):
        doc1, _ = two_doc_files
        result = runner.invoke(
            app,
            ["benchmark", str(built_bundle), "-d", str(doc1), "-q", "Who is the CEO of Google?"],
        )
        assert "Average reduction" in result.output

    def test_missing_query_option_errors(self, runner, patched_pipeline, built_bundle, two_doc_files):
        doc1, _ = two_doc_files
        result = runner.invoke(app, ["benchmark", str(built_bundle), "-d", str(doc1)])
        assert result.exit_code != 0
"""Renders a BenchmarkRun as a rich console table and saves raw per-query
results plus the summary to CSV/JSON for later analysis.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from runner import BenchmarkRun

# Saved retrieved_text is cut to this many characters: the full-context
# baseline returns the whole corpus for every query, which would otherwise
# be repeated once per question in the output files.
MAX_SAVED_TEXT_CHARS = 300


def _token_comparison(noema_tokens: float, other_tokens: float) -> str:
    """How noema's average token count compares to another system's."""
    change = (1 - noema_tokens / other_tokens) * 100
    if change >= 0:
        return f" ({change:.0f}% fewer w/ noema)"
    return f" ({-change:.0f}% more w/ noema)"


def _saved_row(result) -> dict:
    row = asdict(result)
    text = row["retrieved_text"]
    if len(text) > MAX_SAVED_TEXT_CHARS:
        row["retrieved_text"] = text[:MAX_SAVED_TEXT_CHARS] + "..."
    return row


def print_summary_table(run: BenchmarkRun, console: Console | None = None) -> None:
    console = console or Console()

    table = Table(
        title=f"noema benchmark -- {run.num_companies} companies, "
        f"{run.document_count} documents, {run.qa_count} questions"
    )
    table.add_column("system")
    table.add_column("avg tokens", justify="right")
    table.add_column("hit rate", justify="right")
    table.add_column("single-hop", justify="right")
    table.add_column("multi-hop", justify="right")
    table.add_column("avg latency (ms)", justify="right")

    noema_tokens = next((s.avg_tokens for s in run.summaries if s.system == "noema"), None)

    for summary in run.summaries:
        token_note = ""
        if noema_tokens and summary.system != "noema" and summary.avg_tokens:
            token_note = _token_comparison(noema_tokens, summary.avg_tokens)
        table.add_row(
            summary.system,
            f"{summary.avg_tokens:.0f}{token_note}",
            f"{summary.hit_rate * 100:.1f}%",
            f"{summary.single_hop_hit_rate * 100:.1f}%",
            f"{summary.multi_hop_hit_rate * 100:.1f}%",
            f"{summary.avg_latency_seconds * 1000:.1f}",
        )

    console.print(table)


def save_results(run: BenchmarkRun, output_dir: str) -> tuple[Path, Path]:
    """Write per-query results to results.csv and the full run (including
    the summary) to results.json under `output_dir`, with retrieved text
    truncated to MAX_SAVED_TEXT_CHARS. Returns (csv_path, json_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "question",
                "answer",
                "hop_type",
                "token_count",
                "hit",
                "latency_seconds",
                "retrieved_text",
            ],
        )
        writer.writeheader()
        for result in run.results:
            writer.writerow(_saved_row(result))

    json_path = out_dir / "results.json"
    payload = {
        "num_companies": run.num_companies,
        "document_count": run.document_count,
        "qa_count": run.qa_count,
        "summaries": [asdict(s) for s in run.summaries],
        "results": [_saved_row(r) for r in run.results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return csv_path, json_path
"""Run the noema benchmark suite: KnowledgeBase vs. a plain-chunk RAG
baseline vs. naive full-context stuffing, on a synthetic corpus of fictional
companies with known-answer questions (see dataset.py for how the corpus
and questions are generated).

Scales with --num-companies: each company contributes 2 documents and 4
questions, so --num-companies 500 means a 1,000-document corpus and 2,000
scored questions.

Before running (same as main.py):
    uv sync
    uv run python -m spacy download en_core_web_sm

Then:
    uv run benchmarks/run_benchmark.py --num-companies 50
    uv run benchmarks/run_benchmark.py --num-companies 500 --output-dir results/large
"""

from __future__ import annotations

import typer
from baselines import DEFAULT_EMBEDDING_MODEL
from report import print_summary_table, save_results
from rich.console import Console
from runner import run_benchmark

from noema.parser.entities import DEFAULT_MODEL as DEFAULT_SPACY_MODEL
from noema.retrieval.assemble import DEFAULT_MAX_TOKENS
from noema.storage.knowledge_base import DEFAULT_BATCH_SIZE

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    num_companies: int = typer.Option(
        50, "--num-companies", "-n", help="Corpus scale: 2 documents and 4 questions each."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed, for a reproducible corpus."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, "--max-tokens"),
    top_k: int = typer.Option(5, "--top-k", help="Vector search candidates per query."),
    hops: int = typer.Option(1, "--hops", help="noema.KnowledgeBase graph expansion distance."),
    chunk_tokens: int = typer.Option(60, "--chunk-tokens", help="Baseline chunk size."),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, "--batch-size", help="Documents canonicalized together per batch."
    ),
    spacy_model: str = typer.Option(DEFAULT_SPACY_MODEL, "--spacy-model"),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model"),
    output_dir: str = typer.Option(
        "benchmark_results", "--output-dir", "-o", help="Where to save results.csv/.json."
    ),
) -> None:
    """Generate a synthetic corpus and benchmark all three systems against it."""
    with console.status("Running benchmark...") as status:

        def on_progress(message: str) -> None:
            status.update(message)
            console.log(message)

        run = run_benchmark(
            num_companies=num_companies,
            seed=seed,
            max_tokens=max_tokens,
            top_k=top_k,
            hops=hops,
            chunk_tokens=chunk_tokens,
            batch_size=batch_size,
            spacy_model=spacy_model,
            embedding_model=embedding_model,
            on_progress=on_progress,
        )

    console.print()
    print_summary_table(run, console=console)

    csv_path, json_path = save_results(run, output_dir)
    console.print()
    console.print(f"[green]Saved[/green] {csv_path} and {json_path}")


if __name__ == "__main__":
    app()
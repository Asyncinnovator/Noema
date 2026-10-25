"""Command-line interface: noema compress / build-graph / query / inspect / benchmark.

A thin layer over the public API (noema.compress, noema.KnowledgeBase) with
rich-formatted output and a progress bar for long-running ingestion. Install
with the project itself; the `noema` command becomes available on PATH.
"""

# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

import noema
from noema.parser.entities import DEFAULT_MODEL as DEFAULT_SPACY_MODEL
from noema.retrieval.assemble import DEFAULT_MAX_TOKENS
from noema.storage.bundle import DEFAULT_EMBEDDING_MODEL
from noema.storage.knowledge_base import DEFAULT_BATCH_SIZE

app = typer.Typer(
    name="noema",
    help="Convert text into compact logical form for token-efficient RAG.",
    no_args_is_help=True,
)
console = Console()


def _read_text(text: str | None, file: Path | None) -> str:
    """Resolve input text from either a direct argument or a --file option."""
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is not None:
        return text
    raise typer.BadParameter("Provide TEXT or --file PATH")


@app.command()
def compress(
    text: str = typer.Argument(
        None, help="Text to compress. Omit this and pass --file instead."
    ),
    file: Path = typer.Option(
        None, "--file", "-f", exists=True, readable=True, help="Read text from a file."
    ),
    spacy_model: str = typer.Option(DEFAULT_SPACY_MODEL, "--spacy-model"),
) -> None:
    """Compress TEXT into compact notation and print the result."""
    input_text = _read_text(text, file)

    with console.status("Compressing..."):
        result = noema.compress(input_text, spacy_model=spacy_model)

    console.print(result.notation)
    console.print()

    stats = Table(show_header=False, box=None)
    stats.add_row("original tokens", str(result.original_tokens))
    stats.add_row("notation tokens", str(result.notation_tokens))
    stats.add_row("saved", f"{result.saved_percent:.1f}%")
    stats.add_row("confidence", f"{result.confidence:.3f}")
    console.print(stats)


@app.command(name="build-graph")
def build_graph(
    documents: list[Path] = typer.Argument(
        ..., exists=True, readable=True, help="One or more text files to ingest."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Where to save the resulting .noema bundle."
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, "--batch-size", help="Documents canonicalized together per batch."
    ),
    spacy_model: str = typer.Option(DEFAULT_SPACY_MODEL, "--spacy-model"),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model"),
) -> None:
    """Ingest one or more text files into a new .noema knowledge base."""
    texts = [path.read_text(encoding="utf-8") for path in documents]
    kb = noema.KnowledgeBase(embedding_model=embedding_model, spacy_model=spacy_model)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Ingesting documents", total=len(texts))

        def on_batch(_batch_index: int, report: noema.IngestReport) -> None:
            progress.update(task_id, advance=report.documents_ingested)

        report = kb.ingest_stream(texts, batch_size=batch_size, on_batch=on_batch)

    kb.save(str(output))
    console.print(
        f"[green]Saved[/green] {output} -- "
        f"{report.entities_after} entities, {report.relations_after} relations"
    )


@app.command()
def query(
    bundle: Path = typer.Argument(..., exists=True, readable=True, help="A .noema bundle."),
    question: str = typer.Argument(..., help="The question to retrieve context for."),
    top_k: int = typer.Option(5, "--top-k", help="Vector search candidates."),
    hops: int = typer.Option(1, "--hops", help="Graph expansion distance from each candidate."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, "--max-tokens"),
    format: str = typer.Option(
        "notation", "--format", help="'notation' (token-efficient) or 'natural_language'."
    ),
) -> None:
    """Retrieve context for QUESTION from BUNDLE and print it."""
    with console.status("Loading bundle..."):
        kb = noema.KnowledgeBase.load(str(bundle))

    with console.status("Retrieving..."):
        result = kb.retrieve(
            question, top_k=top_k, hops=hops, max_tokens=max_tokens, format=format
        )

    console.print(result.text)
    console.print()
    truncated_note = " (truncated by max-tokens)" if result.truncated else ""
    console.print(f"[dim]{result.token_count} tokens{truncated_note}[/dim]")


@app.command()
def inspect(
    bundle: Path = typer.Argument(..., exists=True, readable=True, help="A .noema bundle."),
) -> None:
    """Print summary statistics for a .noema bundle without loading it fully."""
    loaded = noema.load_bundle(str(bundle))
    manifest = loaded.manifest

    table = Table(title=str(bundle), show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("schema version", str(manifest.get("schema_version")))
    table.add_row("created at", str(manifest.get("created_at")))
    table.add_row("spacy model", str(manifest.get("spacy_model")))
    table.add_row("embedding model", str(manifest.get("embedding_model")))
    table.add_row("entities", str(manifest.get("entity_count")))
    table.add_row("relations", str(manifest.get("relation_count")))
    table.add_row("events", str(manifest.get("event_count")))
    table.add_row("facts indexed", str(manifest.get("fact_count")))
    table.add_row("source documents", str(len(manifest.get("source_document_ids", []))))
    table.add_row("file size", f"{bundle.stat().st_size:,} bytes")
    console.print(table)


@app.command()
def benchmark(
    bundle: Path = typer.Argument(..., exists=True, readable=True, help="A .noema bundle."),
    document: list[Path] = typer.Option(
        ...,
        "--document",
        "-d",
        exists=True,
        readable=True,
        help="A source document for the naive full-context baseline (repeat for more).",
    ),
    queries: list[str] = typer.Option(
        ..., "--query", "-q", help="A question to benchmark (repeat for more)."
    ),
    top_k: int = typer.Option(5, "--top-k"),
    hops: int = typer.Option(1, "--hops"),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, "--max-tokens"),
) -> None:
    """Compare retrieve() token counts against a naive full-document baseline."""
    with console.status("Loading bundle..."):
        kb = noema.KnowledgeBase.load(str(bundle))

    baseline_text = "\n".join(path.read_text(encoding="utf-8") for path in document)
    baseline_tokens = noema.count_tokens(baseline_text)

    table = Table(title="noema retrieval vs. naive full-context baseline")
    table.add_column("query")
    table.add_column("retrieved tokens", justify="right")
    table.add_column("baseline tokens", justify="right")
    table.add_column("reduction", justify="right")

    reductions = []
    for question in queries:
        result = kb.retrieve(question, top_k=top_k, hops=hops, max_tokens=max_tokens)
        reduction = 1 - (result.token_count / baseline_tokens) if baseline_tokens else 0.0
        reductions.append(reduction)
        table.add_row(question, str(result.token_count), str(baseline_tokens), f"{reduction * 100:.1f}%")

    console.print(table)
    if reductions:
        average = sum(reductions) / len(reductions) * 100
        console.print(f"\nAverage reduction across {len(reductions)} queries: {average:.1f}%")


if __name__ == "__main__":
    app()
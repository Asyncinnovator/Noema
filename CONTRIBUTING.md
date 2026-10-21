# Contributing

## Setup

```bash
git clone <repository-url>
cd noema
uv sync --extra rdf --extra neo4j
```

Python 3.13 or newer is required.

## Tests

```bash
uv run pytest
```

The unit tests mock the embedding models, the spaCy and tokenizer loading, and external services,
so they run without network access. Tests for optional integrations are skipped when their extra
is not installed. For example, `tests/unit/test_rdf_backend.py` is skipped without `rdflib`, so
install the `rdf` extra to run it.

When you test code that embeds a corpus and a query in separate calls, use a fake embedding model
with a fixed vocabulary, so every call returns vectors of the same dimension. A real model always
does.

`noema/__init__.py` re-exports the `compress` function, which shadows the `noema.compress`
submodule. To patch the module in tests, use `importlib.import_module("noema.compress")`.

## Linting

```bash
uv run ruff check .
```

## Project layout

```
src/noema/
|-- parser/          entities, relations, coreference, transformer relations
|-- graph/           SemanticGraph
|-- canonicalize/    entity deduplication
|-- notation/        encoder and decoder
|-- validator/       round-trip validation (embedding and NLI)
|-- retrieval/       vector search, graph expansion, assembly
|-- storage/         bundle format, KnowledgeBase, Neo4j and RDF backends
|-- integrations/    LangChain, LlamaIndex, token counting
|-- compress.py      noema.compress()
|-- schema.py        data types
`-- cli.py           the noema command
benchmarks/          synthetic corpus, baselines, metrics, runner, report
tests/unit/          unit tests
```

## Pull requests

- Keep changes focused and add tests for new behavior.
- Run `uv run pytest` and `uv run ruff check .` before opening a pull request.
- Describe what changed and why.

## License

Noema is licensed under the Apache License 2.0. Contributions are accepted under the same license.

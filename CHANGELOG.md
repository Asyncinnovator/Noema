# Changelog

All notable changes to Noema are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is `0`, the
public API may change between minor versions, and breaking changes are listed under **Changed**
or **Removed**.


## [0.1.0] - 2026-10-24

First public release. The package is published on PyPI as `noema-rag` and imported as `noema`.

### Added

- **Compression:** `noema.compress()` converts text into compact notation and returns a
  round-trip confidence score, token counts and the percentage of tokens saved.
- **Parsing:** entity extraction with spaCy and rule-based relation extraction, with provenance
  (source document and character span) and a confidence value on every item.
- **Canonicalization:** exact-match and embedding-similarity entity merging within the same entity
  type, with a configurable similarity threshold.
- **Notation:** a line-oriented notation grammar with `encode_graph()`, `parse_notation()` and
  `decode()`.
- **Validation:** embedding-based round-trip validation with `validate()`, and an entailment-based
  check with `noema.validator.nli.validate_nli()`.
- **Knowledge base:** `KnowledgeBase` with `ingest()`, `ingest_stream()` (batched, with
  checkpoints and progress callbacks), `save()`, `load()` and incremental updates that merge into
  the existing graph.
- **Storage:** the single-file `.noema` bundle (graph, embeddings, similarity index and manifest).
- **Retrieval:** `kb.retrieve()` with vector search over facts, graph expansion by hops, and
  notation assembly under a token budget, in `notation` or `natural_language` format.
- **Persistent backends:** the `GraphBackend` interface, with optional `Neo4jBackend` and
  `RDFBackend`, plus `sync_to_backend()` and `load_from_backend()`.
- **Advanced NLP (optional):** coreference resolution (`noema[advanced-nlp]`), transformer-based
  relation extraction and a hybrid extractor.
- **Integrations:** `NoemaRetriever` for LangChain, `NoemaLlamaIndexRetriever` for LlamaIndex, and
  OpenAI and Anthropic token counting helpers.
- **CLI:** the `noema` command with `compress`, `build-graph`, `query`, `inspect` and `benchmark`.
- **Benchmarks:** a seeded synthetic corpus with known-answer questions (including multi-hop),
  baselines for plain-chunk RAG and full-context stuffing, and `benchmarks/run_benchmark.py`.
- **Documentation:** a Wiki with installation, guides, a CLI reference and an API reference.
- **Project files:** contributing guide, code of conduct, security policy, governance, issue and
  pull request templates, and the Apache-2.0 license.

### Known limitations

- The built-in parser extracts entities and relations only. Events are supported in the data
  model, notation and backends but are not extracted automatically.
- Rule-based relation extraction handles simple subject-verb-object sentences and is English
  specific. Coreference and transformer relation extraction are separate functions and are not
  used by `compress()` or `ingest()`.
- Embedding-based validation cannot detect reversed claims. Use `validate_nli()` for that.
- The spaCy English model is not installed with the package. Run
  `python -m spacy download en_core_web_sm` once after installing.
- Python 3.13 or newer is required.

[0.1.0]: https://github.com/Asyncinnovator/Noema/releases/tag/v0.1.0
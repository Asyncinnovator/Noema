# Changelog

All notable changes to Noema are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the major version is `0`, the
public API may change between minor versions, and breaking changes are listed under **Changed**
or **Removed**.


## [0.1.1] - 2026-11-27

Bug-fix and hardening release. It fixes three ways the knowledge base could silently end up with
wrong facts, makes saving crash-safe, and stops `.noema` bundles from controlling what gets loaded.
Upgrading is recommended for everyone, and in particular for anyone who loads bundles they did not
create. One behavior change affects loading bundles built with a custom embedding model (see
**Changed**).

### Fixed

- **Negation:** a negated sentence no longer produces the fact it denies. "Google did not acquire
  YouTube" and "Larry Page never founded Yahoo" previously produced `acquire(Google, YouTube)` and
  `found(Larry Page, Yahoo)`. Clauses negated on the verb ("did not", "never", "n't") or on a main
  argument ("no", "neither ... nor") are now skipped.
- **Surname merging:** a bare surname no longer merges distinct people. With "David Chen", "Wei
  Chen" and "Chen", all three used to collapse into one entity. A bare surname is now attached to a
  full name only when exactly one full name matches.
- **Name normalization:** "C++" and "C#" no longer normalize to the same key, and "AT&T" no longer
  matches "ATT". Names made only of business suffixes ("Group", "Company", "Holdings Inc.") no
  longer normalize to an empty string and merge with each other.
- **Crash-safe saves:** `save_bundle()`, `KnowledgeBase.save()`, `ingest_stream()` checkpoints and
  `RDFBackend.save()` now write to a temporary file and rename it into place. Previously a failure
  partway through replaced the existing bundle or `.ttl` file with a truncated one.
- **CLI output:** `noema query` printed results as Rich markup, so an entity name such as `[/bar]`
  could raise an error. Output is now printed literally.

### Security

- **Bundle loading:** `load_bundle()` now caps how far each entry may expand (a roughly 100 KB
  bundle could previously expand to 100 MB or more in memory), validates the manifest, and checks
  that the graph, fact ids, embeddings and similarity index agree with each other. Any problem
  raises the new `noema.BundleError` instead of a `KeyError`, `AttributeError` or faiss error.
  The limits can be raised with `max_json_bytes` and `max_binary_bytes`.
- **Model names in bundles:** `KnowledgeBase.load()` no longer follows the embedding and spaCy model
  names recorded in a bundle's manifest, because a name read from a file could point the next
  query or ingest at an arbitrary Hugging Face repository or local path.
- **Dependencies:** the minimum `torch` version is now 2.6.0, which fixes the `torch.load`
  remote-code-execution issue CVE-2025-32434.

### Changed

- **`KnowledgeBase.load()` and model names.** The spaCy model recorded in a bundle is restored only
  when it looks like an installed package name (for example `en_core_web_md`); anything else, such
  as a path or URL, is ignored with a warning and the default is used. The embedding model is
  restored only when it is the default, `all-MiniLM-L6-v2`. A bundle built with any other
  embedding model now raises `BundleError` unless you pass `embedding_model=...` or
  `trust_manifest_models=True`. Bundles built with the default models load exactly as before.
- `noema query` and `noema benchmark` accept `--trust-bundle-models` for the same purpose, and
  `noema query`, `noema benchmark` and `noema inspect` print a short error instead of a traceback
  for a bad bundle.
- The transformer relation extractor accepts a `revision` argument so the Hugging Face model can be
  pinned to a specific commit.

### Added

- `noema.BundleError`, raised for malformed, inconsistent or oversized bundles. It is a subclass of
  `ValueError`.
- `trust_manifest_models`, `embedding_model` and `spacy_model` keyword arguments on
  `KnowledgeBase.load()`.

### Known limitations

- Negation handling skips the whole sentence, so "Google did not acquire Yahoo but acquired
  YouTube" yields neither fact. Hedged or reported statements ("denied acquiring", "plans to
  acquire") are not detected.
- The `transformers` version in `uv.lock` (4.57.6) has published advisories whose fixes are only in
  5.x, which `fastcoref` does not yet support. The relevant code paths are not used by Noema.
- The similarity index reader in faiss is not hardened against hostile files. Only load bundles
  from sources you trust.
- The 0.1.0 limitations otherwise still apply.

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

[0.1.1]: https://github.com/Asyncinnovator/Noema/releases/tag/v0.1.1
[0.1.0]: https://github.com/Asyncinnovator/Noema/releases/tag/v0.1.0
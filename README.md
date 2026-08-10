<div align="center">

<img src="docs/logo-name.png" alt="Noema logo" width="100%">

# Noema

### Convert text into compact logical form — fewer tokens, no lost context, built for RAG

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/noema/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Build status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Tests](https://img.shields.io/badge/tests-57%2F57%20passing-brightgreen)](#)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

</div>

---

## Overview

**Noema** is a Python library for knowledge graph construction and semantic compression, built for token-efficient Retrieval-Augmented Generation (RAG) pipelines.

Instead of treating documents as flat strings, Noema parses text into a structured logical form — entities, relations, and events — and organizes that structure into a canonical, deduplicated knowledge graph. The same compact representation that reduces token usage also serves as the retrieval layer for RAG, so token savings and retrieval quality come from one underlying design rather than two separate systems bolted together.

Where a typical RAG pipeline stores and re-embeds raw text chunks, Noema stores canonical facts, collapses redundant mentions of the same fact into a single node, and generates compact notation only for the sub-graph relevant to a given query. A round-trip validation step decodes that notation back into natural language and checks it against the original meaning, so compression comes with a confidence guarantee rather than an assumption of safety.

### Core Capabilities

- **Text → semantic graph parsing** — entities, relations, and events extracted from natural language, with coreference resolution
- **Canonicalization & deduplication** — the same fact stated multiple ways, across one document or many, collapses into a single canonical node
- **Compact symbolic notation** — a low-token representation of graph facts, suitable for direct injection into LLM context
- **Hybrid retrieval** — vector similarity search combined with multi-hop graph traversal, not just nearest-chunk matching
- **Round-trip validation** — notation is decoded back to natural language and scored for semantic equivalence before it's trusted
- **Incremental knowledge base updates** — new documents merge into an existing graph instead of accumulating as duplicates
- **Framework integrations** — drop-in components for LangChain and LlamaIndex
- **CLI** — command-line tools for compression, graph building, querying, and inspection

### Project Status

Noema is under active development, following a phased build plan from core parsing and graph construction through compact notation, storage, retrieval, framework integrations, and a CLI.

### License

Noema is licensed under the [Apache License 2.0](LICENSE).
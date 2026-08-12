"""Noema: knowledge graph and semantic compression library for token-efficient RAG.

Public API is assembled here as each phase lands.
"""

# NOTE: re-exported as `canonicalize_entities`, not `canonicalize`. The
# subpackage is itself named `noema.canonicalize`, and Python auto-binds
# that name to the submodule the moment it's imported. Re-exporting the
# function under the identical name `canonicalize` here would silently
# rebind and clobber that submodule reference in noema's namespace, breaking
# `import noema.canonicalize.dedup` for anyone importing the package fresh.
# The function is still available as `noema.canonicalize.canonicalize(...)`
# via the subpackage's own __init__.py, where no such clash exists.
from noema.canonicalize.dedup import canonicalize as canonicalize_entities
from noema.graph import SemanticGraph
from noema.parser import extract_entities, extract_relations
from noema.notation import encode_graph, decode, parse_notation, ParsedNotation
from noema.validator import validate, ValidationResult
from noema.compress import count_tokens, compress
from noema.storage.bundle import load_bundle, save_bundle, LoadedBundle
from noema.storage.knowledge_base import KnowledgeBase, IngestReport
from noema.retrieval import (
    search_facts,
    embed_query,
    FactMatch,
    expand_from_facts,
    relations_within,
    ExpandedSubgraph,
    assemble,
    AssembledContext,
)
from noema.schema import (
    Document,
    Entity,
    EntityType,
    Event,
    ParsedDocument,
    Relation,
    SourceSpan,
)

__all__ = [
    "Document",
    "Entity",
    "EntityType",
    "Event",
    "ParsedDocument",
    "Relation",
    "SourceSpan",
    "SemanticGraph",
    "extract_entities",
    "extract_relations",
    "canonicalize_entities",
    "encode_graph",
    "decode",
    "parse_notation",
    "ParsedNotation",
    "validate",
    "ValidationResult",
    "count_tokens",
    "compress",
    "load_bundle",
    "save_bundle",
    "LoadedBundle",
    "KnowledgeBase",
    "IngestReport",
    "search_facts",
    "embed_query",
    "FactMatch",
    "expand_from_facts",
    "relations_within",
    "ExpandedSubgraph",
    "assemble",
    "AssembledContext",
]
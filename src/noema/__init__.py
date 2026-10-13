"""Noema: knowledge graph and semantic compression library for token-efficient RAG."""

# Re-exported as canonicalize_entities to avoid shadowing the
# noema.canonicalize submodule name.
from noema.canonicalize.dedup import canonicalize as canonicalize_entities
from noema.compress import compress, count_tokens
from noema.graph import SemanticGraph
from noema.notation import decode, encode_graph, parse_notation
from noema.parser import extract_entities, extract_relations
from noema.schema import (
    Document,
    Entity,
    EntityType,
    Event,
    ParsedDocument,
    Relation,
    SourceSpan,
)
from noema.storage.backends import GraphBackend
from noema.storage.bundle import LoadedBundle, load_bundle, save_bundle
from noema.storage.knowledge_base import IngestReport, KnowledgeBase
from noema.validator import validate

__all__ = [
    "Document",
    "Entity",
    "EntityType",
    "Event",
    "GraphBackend",
    "IngestReport",
    "KnowledgeBase",
    "LoadedBundle",
    "ParsedDocument",
    "Relation",
    "SemanticGraph",
    "SourceSpan",
    "canonicalize_entities",
    "compress",
    "count_tokens",
    "decode",
    "encode_graph",
    "extract_entities",
    "extract_relations",
    "load_bundle",
    "parse_notation",
    "save_bundle",
    "validate",
]
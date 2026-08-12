"""Hybrid retrieval: vector search -> graph expansion -> notation assembly."""

from noema.retrieval.assemble import AssembledContext, assemble
from noema.retrieval.expand import ExpandedSubgraph, expand_from_facts, relations_within
from noema.retrieval.vector_search import FactMatch, embed_query, search_facts

__all__ = [
    "search_facts",
    "embed_query",
    "FactMatch",
    "expand_from_facts",
    "relations_within",
    "ExpandedSubgraph",
    "assemble",
    "AssembledContext",
]
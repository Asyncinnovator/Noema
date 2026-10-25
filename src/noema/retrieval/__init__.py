"""Hybrid retrieval: vector search -> graph expansion -> notation assembly."""

from noema.retrieval.assemble import AssembledContext, assemble
from noema.retrieval.expand import ExpandedSubgraph, expand_from_facts, relations_within
from noema.retrieval.vector_search import FactMatch, embed_query, search_facts

__all__ = [
    "AssembledContext",
    "ExpandedSubgraph",
    "FactMatch",
    "assemble",
    "embed_query",
    "expand_from_facts",
    "relations_within",
    "search_facts",
]
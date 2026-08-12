"""Sub-graph -> notation assembly with a token budget.

Phase 4 scope: the final stage of "vector search -> graph expansion ->
sub-graph -> notation". Takes an ExpandedSubgraph (from retrieval.expand)
and greedily assembles as much of it as fits within max_tokens, in priority
order:

  1. Facts that directly matched the query (seed_relation_ids) before facts
     merely reachable near them.
  2. Among non-seed facts, closer facts (lower hop-distance from the nearest
     seed entity) before farther ones.

This is priority-ordered greedy packing, not an optimal knapsack solve --
assembly stops at the FIRST fact that would exceed the budget rather than
skipping it to try a smaller, lower-priority one afterward. Given the
ordering already reflects relevance to the query, taking the highest-
priority prefix that fits is the right tradeoff here: a knapsack-optimal
pack could end up swapping in several low-relevance facts instead of one
higher-relevance one, purely because they happened to be smaller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noema.compress import count_tokens
from noema.graph.builder import SemanticGraph
from noema.notation.encoder import encode_graph
from noema.retrieval.expand import ExpandedSubgraph, relations_within
from noema.schema import Relation

DEFAULT_MAX_TOKENS = 200
DEFAULT_TOKEN_MODEL = "cl100k_base"


@dataclass
class AssembledContext:
    """Final retrieval output: the notation text plus what went into it."""

    notation: str = ""
    entity_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)
    token_count: int = 0
    truncated: bool = False


def _priority_key(relation: Relation, expanded: ExpandedSubgraph) -> tuple[int, int, str]:
    """Sort key; lower sorts first (= higher priority): seed facts (0) before
    non-seed facts (1), then by the farther of the relation's two endpoint
    distances (closer facts first), then by id for a stable, deterministic
    order when everything else ties.
    """
    is_seed = 0 if relation.id in expanded.seed_relation_ids else 1
    distance = max(
        expanded.entity_distance.get(relation.subject_id, 0),
        expanded.entity_distance.get(relation.object_id, 0),
    )
    return (is_seed, distance, relation.id)


def _render(graph: SemanticGraph, entity_ids: set[str], relation_ids: set[str]) -> str:
    """Render notation for EXACTLY the given entities/relations.

    Deliberately not encode_graph(graph, entity_ids=entity_ids) directly --
    that would auto-include every relation the full graph has between those
    entities, even ones not yet committed during greedy assembly (see
    assemble()'s loop), which would make the running token count reflect
    facts the budget check hasn't actually approved yet.
    """
    subset = SemanticGraph()
    for entity_id in entity_ids:
        entity = graph.get_entity(entity_id)
        if entity is not None:
            subset.add_entity(entity)
    for relation in graph.relations:
        if relation.id in relation_ids:
            subset.add_relation(relation)
    return encode_graph(subset)


def assemble(
    graph: SemanticGraph,
    expanded: ExpandedSubgraph,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_model: str = DEFAULT_TOKEN_MODEL,
) -> AssembledContext:
    """Greedily pack the expanded sub-graph's relations into notation text,
    highest priority first, stopping at the token budget.

    `truncated` is True whenever at least one candidate relation existed but
    didn't make it in -- either because it alone would have exceeded
    max_tokens, or because an earlier, higher-priority relation used up the
    remaining budget first.
    """
    candidates = relations_within(graph, expanded.entity_ids)
    candidates.sort(key=lambda r: _priority_key(r, expanded))

    included_entity_ids: set[str] = set()
    included_relation_ids: set[str] = set()
    notation = ""
    truncated = False

    for relation in candidates:
        trial_entity_ids = included_entity_ids | {relation.subject_id, relation.object_id}
        trial_relation_ids = included_relation_ids | {relation.id}
        trial_notation = _render(graph, trial_entity_ids, trial_relation_ids)

        if count_tokens(trial_notation, model=token_model) > max_tokens:
            truncated = True
            break

        included_entity_ids = trial_entity_ids
        included_relation_ids = trial_relation_ids
        notation = trial_notation

    return AssembledContext(
        notation=notation,
        entity_ids=included_entity_ids,
        relation_ids=included_relation_ids,
        token_count=count_tokens(notation, model=token_model),
        truncated=truncated,
    )

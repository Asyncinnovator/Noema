"""Tests for noema.retrieval.assemble: sub-graph -> notation with a token
budget. tiktoken is mocked since its remote vocab isn't reachable in this
sandbox.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

from noema.graph.builder import SemanticGraph
from noema.retrieval.assemble import AssembledContext, assemble
from noema.retrieval.expand import ExpandedSubgraph, expand_from_facts
from noema.schema import Entity, EntityType, Relation

compress_module = importlib.import_module("noema.compress")


class _WhitespaceEncoding:
    """Whitespace-split stand-in for a tiktoken.Encoding."""

    def encode(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture
def patched_tokenizer():
    with patch.object(compress_module, "_load_token_encoding", return_value=_WhitespaceEncoding()):
        yield


def _entity(name: str) -> Entity:
    return Entity(name=name, type=EntityType.OTHER)


def _build_star() -> tuple[SemanticGraph, Entity, list[Entity], list[Relation]]:
    graph = SemanticGraph()
    center = _entity("Center")
    leaves = [_entity(f"Leaf{i}") for i in range(3)]
    graph.add_entity(center)
    for leaf in leaves:
        graph.add_entity(leaf)
    relations = [
        Relation(subject_id=leaf.id, predicate="connects_to", object_id=center.id)
        for leaf in leaves
    ]
    for r in relations:
        graph.add_relation(r)
    return graph, center, leaves, relations


class TestAssembleBasic:
    def test_generous_budget_includes_everything(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        result = assemble(graph, expanded, max_tokens=1000)
        assert len(result.relation_ids) == 3
        assert result.truncated is False

    def test_result_type_is_assembled_context(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        result = assemble(graph, expanded, max_tokens=1000)
        assert isinstance(result, AssembledContext)

    def test_notation_contains_included_entities(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        result = assemble(graph, expanded, max_tokens=1000)
        assert "Center" in result.notation
        assert "Leaf0" in result.notation

    def test_token_count_matches_notation(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        result = assemble(graph, expanded, max_tokens=1000)
        assert result.token_count == len(result.notation.split())


class TestAssemblePriority:
    def test_seed_relation_always_included_under_tight_budget(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        result = assemble(graph, expanded, max_tokens=15)
        assert relations[0].id in result.relation_ids

    def test_tight_budget_truncates(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        result = assemble(graph, expanded, max_tokens=15)
        assert result.truncated is True
        assert len(result.relation_ids) < 3

    def test_non_seed_facts_ordered_by_distance(self, patched_tokenizer):
        # Chain: seed in the middle, one non-seed fact closer than another
        graph = SemanticGraph()
        a, b, c, d = (_entity(n) for n in "ABCD")
        for e in (a, b, c, d):
            graph.add_entity(e)
        r_seed = Relation(subject_id=b.id, predicate="seed", object_id=c.id)
        r_near = Relation(subject_id=a.id, predicate="near", object_id=b.id)  # distance 1 from seed
        graph.add_relation(r_seed)
        graph.add_relation(r_near)
        expanded = expand_from_facts(graph, [r_seed.id], hops=1)

        result = assemble(graph, expanded, max_tokens=1000)
        assert r_seed.id in result.relation_ids
        assert r_near.id in result.relation_ids


class TestAssembleEdgeCases:
    def test_zero_max_tokens_yields_empty_notation(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        expanded = expand_from_facts(graph, [relations[0].id], hops=1)
        result = assemble(graph, expanded, max_tokens=0)
        assert result.notation == ""
        assert result.truncated is True

    def test_no_seeds_yields_empty_untruncated_result(self, patched_tokenizer):
        graph, center, leaves, relations = _build_star()
        result = assemble(graph, ExpandedSubgraph(), max_tokens=1000)
        assert result.notation == ""
        assert result.truncated is False

    def test_committed_relations_only_not_full_induced_subgraph(self, patched_tokenizer):
        # Two relations share both endpoints in the entity set, but only one
        # is a seed/candidate -- assembly must not silently pull in a
        # relation that wasn't part of the ranked candidate list.
        graph = SemanticGraph()
        a, b = _entity("A"), _entity("B")
        graph.add_entity(a)
        graph.add_entity(b)
        r1 = Relation(subject_id=a.id, predicate="rel_one", object_id=b.id)
        graph.add_relation(r1)
        expanded = expand_from_facts(graph, [r1.id], hops=1)
        result = assemble(graph, expanded, max_tokens=1000)
        assert "rel_one" in result.notation
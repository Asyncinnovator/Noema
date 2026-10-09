"""Entity canonicalization: merge duplicate nodes into one canonical entity.

Exact-name matching, followed by embedding-similarity clustering within
each EntityType, using a small sentence-transformers model. Entities only
merge within the same EntityType to avoid false merges like a person named
Washington matching the city. Coreference resolution ("it", "the company")
is handled separately upstream in parser.coref.

Operates on a whole SemanticGraph: the parser assigns each entity mention a
fresh id per document, so the same real-world thing appearing across
multiple documents starts out as multiple distinct nodes. This module
merges those into one canonical Entity and remaps every Relation/Event
that referenced an old id to the canonical id.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from noema.graph.builder import SemanticGraph
from noema.schema import Entity, EntityType, Event, Relation

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.85

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

# Common business-entity suffixes, stripped during normalization so
# "Google Inc." and "Google" become an exact match before embeddings.
_BUSINESS_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "co", "company", "group", "holdings", "plc",
}


def _normalize_name(name: str) -> str:
    """'Google, Inc.' -> 'google' -- for cheap exact-match comparison
    before falling back to embeddings.
    """
    name = _PUNCT.sub("", name.lower())
    name = _WHITESPACE.sub(" ", name).strip()
    words = name.split()
    while words and words[-1] in _BUSINESS_SUFFIXES:
        words.pop()
    return " ".join(words)


@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class _UnionFind:
    """Minimal disjoint-set for clustering entity ids into merge groups."""

    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = defaultdict(list)
        for item in self._parent:
            clusters[self.find(item)].append(item)
        return clusters


def _is_surname_match(short_name: str, long_name: str) -> bool:
    """True if `short_name` is a single word matching the last word of
    `long_name`, e.g. 'chen' matching 'david chen'. Scoped to
    EntityType.PERSON only: the same heuristic on ORG names would be
    risky -- "Labs" shouldn't auto-match "Bright Sensor Labs".
    """
    short_words = short_name.split()
    long_words = long_name.split()
    return len(short_words) == 1 and len(long_words) > 1 and short_words[0] == long_words[-1]


def _merge_cluster(entities: list[Entity]) -> Entity:
    """Collapse a cluster of duplicate Entity objects into one canonical
    Entity. For PERSON, prefers the most complete name (most words), with
    mention count as a tiebreak. For everything else, prefers the
    most-attested name (most mentions).
    """
    entity_type = entities[0].type
    if entity_type == EntityType.PERSON:
        canonical = max(entities, key=lambda e: (len(e.name.split()), len(e.mentions)))
    else:
        canonical = max(entities, key=lambda e: len(e.mentions))

    merged_mentions = []
    merged_aliases: list[str] = []
    seen = set()

    for entity in entities:
        merged_mentions.extend(entity.mentions)
        names_to_consider = entity.aliases if entity.id == canonical.id else [entity.name, *entity.aliases]
        for alias in names_to_consider:
            if alias != canonical.name and alias not in seen:
                seen.add(alias)
                merged_aliases.append(alias)

    return Entity(
        id=canonical.id,
        name=canonical.name,
        type=canonical.type,
        aliases=merged_aliases,
        mentions=merged_mentions,
        confidence=min(e.confidence for e in entities),
    )


def canonicalize(
    graph: SemanticGraph,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> SemanticGraph:
    """Return a new SemanticGraph with duplicate entities merged.

    Three passes, all scoped within each EntityType (a PERSON and an ORG
    never merge):

      1. Exact match on normalized name.
      2. Surname matching, PERSON entities only (see _is_surname_match).
      3. Embedding cosine similarity between remaining distinct names.
    """
    by_type: dict[str, list[Entity]] = defaultdict(list)
    for entity in graph.entities:
        by_type[entity.type].append(entity)

    id_remap: dict[str, str] = {}
    canonical_entities: dict[str, Entity] = {}

    for entity_type, group in by_type.items():
        uf = _UnionFind([e.id for e in group])

        by_normalized: dict[str, list[Entity]] = defaultdict(list)
        for entity in group:
            by_normalized[_normalize_name(entity.name)].append(entity)
        for same_name_entities in by_normalized.values():
            for entity in same_name_entities[1:]:
                uf.union(same_name_entities[0].id, entity.id)

        distinct_names = list(by_normalized.keys())

        if entity_type == EntityType.PERSON:
            for name_a in distinct_names:
                for name_b in distinct_names:
                    if name_a == name_b:
                        continue
                    if _is_surname_match(name_a, name_b):
                        uf.union(by_normalized[name_a][0].id, by_normalized[name_b][0].id)

        if len(distinct_names) > 1:
            model = _load_model(model_name)
            embeddings = model.encode(distinct_names, normalize_embeddings=True)
            sims = embeddings @ embeddings.T
            for i in range(len(distinct_names)):
                for j in range(i + 1, len(distinct_names)):
                    if sims[i, j] >= similarity_threshold:
                        rep_i = by_normalized[distinct_names[i]][0].id
                        rep_j = by_normalized[distinct_names[j]][0].id
                        uf.union(rep_i, rep_j)

        for cluster_ids in uf.groups().values():
            cluster_entities = [e for e in group if e.id in cluster_ids]
            canonical = _merge_cluster(cluster_entities)
            canonical_entities[canonical.id] = canonical
            for entity in cluster_entities:
                id_remap[entity.id] = canonical.id

    return _rebuild_graph(graph, canonical_entities, id_remap)


def _rebuild_graph(
    graph: SemanticGraph,
    canonical_entities: dict[str, Entity],
    id_remap: dict[str, str],
) -> SemanticGraph:
    """Build a fresh graph from canonical entities, with relations and
    events remapped from old ids to canonical ids. Relations that become
    self-loops after merging are dropped; relations collapsing onto the
    same (subject, predicate, object) triple are merged.
    """
    new_graph = SemanticGraph()
    for entity in canonical_entities.values():
        new_graph.add_entity(entity)

    merged_relations: dict[tuple[str, str, str], Relation] = {}
    for relation in graph.relations:
        subject_id = id_remap[relation.subject_id]
        object_id = id_remap[relation.object_id]
        if subject_id == object_id:
            continue
        key = (subject_id, relation.predicate, object_id)
        existing = merged_relations.get(key)
        if existing is not None:
            existing.mentions.extend(relation.mentions)
            existing.confidence = max(existing.confidence, relation.confidence)
            continue
        merged_relations[key] = Relation(
            subject_id=subject_id,
            predicate=relation.predicate,
            object_id=object_id,
            mentions=list(relation.mentions),
            confidence=relation.confidence,
            valid_from=relation.valid_from,
            valid_to=relation.valid_to,
        )
    for relation in merged_relations.values():
        new_graph.add_relation(relation)

    for event in graph.events:
        remapped_participants = list(dict.fromkeys(id_remap[p] for p in event.participant_ids))
        new_graph.add_event(
            Event(
                id=event.id,
                label=event.label,
                participant_ids=remapped_participants,
                time=event.time,
                mentions=list(event.mentions),
                confidence=event.confidence,
            )
        )

    return new_graph
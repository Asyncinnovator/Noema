"""KnowledgeBase: the graph-backed, persistent, incrementally-updatable
store described in the project plan's core function #2 and #4.

    kb = noema.KnowledgeBase()
    kb.ingest(documents)             # parses, dedupes, canonicalizes
    kb.save("company_kb.noema")
    ...
    kb = noema.KnowledgeBase.load("company_kb.noema")
    kb.ingest(new_documents)         # new facts merge into existing entities

Incremental ingestion strategy: each .ingest() call re-runs canonicalize()
over the UNION of the existing graph and the newly parsed documents, rather
than only deduping the new batch against itself. This is what makes "Google"
mentioned in an ingest() call today merge with "Google" ingested last week
into the same node, instead of accumulating a fresh duplicate every call --
the exact requirement Phase 3 sets out ("avoid duplicate nodes").

The tradeoff: this re-canonicalizes the whole knowledge base on every
ingest(), which is correct but not scalable -- cost grows with total KB
size, not just the new batch. That's an accepted Phase 3 limitation, not an
oversight: efficient incremental canonicalization (e.g. only comparing new
entities against an existing embedding index instead of all-pairs) is a
Phase 7 (scale) concern once KBs are large enough for it to matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noema.canonicalize.dedup import DEFAULT_SIMILARITY_THRESHOLD, canonicalize
from noema.graph.builder import SemanticGraph
from noema.notation.decoder import decode
from noema.parser.entities import DEFAULT_MODEL as SPACY_DEFAULT_MODEL
from noema.parser.entities import extract_entities
from noema.parser.relations import extract_relations
from noema.retrieval.assemble import DEFAULT_MAX_TOKENS, DEFAULT_TOKEN_MODEL, assemble
from noema.retrieval.expand import expand_from_facts
from noema.retrieval.vector_search import FactMatch, search_facts
from noema.schema import Document, ParsedDocument
from noema.storage.bundle import DEFAULT_EMBEDDING_MODEL, build_fact_index, load_bundle, save_bundle


@dataclass
class IngestReport:
    """Summary of what one .ingest() call did -- the "dedup stats" the
    project plan's manifest.json describes, surfaced back to the caller
    rather than only written to disk.
    """

    documents_ingested: int
    entities_before: int
    entities_after: int
    entities_parsed: int
    relations_before: int
    relations_after: int

    @property
    def entities_merged(self) -> int:
        """How many entities were folded into another entity (by exact-match
        or embedding-similarity dedup) during this ingest.

        NOT simply `entities_before - entities_after` -- ingest() almost
        always *grows* the graph (new documents contribute new entities
        alongside whatever merges also happen), so that difference is
        usually negative and clamps to a misleading 0 even when real merges
        occurred. The correct count compares the total RAW candidate pool
        for this ingest (existing entities + newly parsed ones, before
        canonicalization) against what remained after: entities_parsed
        exists specifically so this property can compute that correctly.
        """
        raw_candidate_count = self.entities_before + self.entities_parsed
        return max(0, raw_candidate_count - self.entities_after)


@dataclass
class RetrievalResult:
    """Result of kb.retrieve(query, ...).

    `text` is what the project plan's usage example injects into an LLM
    prompt (`context.text`) -- its content depends on `format`: the raw
    notation for format="notation" (the default, and the token-efficient
    choice), or a decoded natural-language rendering for
    format="natural_language". `notation` is always the raw notation
    regardless of format, for the same reason CompressResult always
    includes decoded_text: costs nothing to include, and makes a surprising
    result debuggable.
    """

    text: str
    notation: str
    entity_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)
    token_count: int = 0
    truncated: bool = False
    fact_matches: list[FactMatch] = field(default_factory=list)


def _normalize_documents(documents: str | Document | list[str | Document]) -> list[Document]:
    if isinstance(documents, (str, Document)):
        documents = [documents]
    return [d if isinstance(d, Document) else Document(text=d) for d in documents]


def _merge_graphs(*graphs: SemanticGraph) -> SemanticGraph:
    """Combine multiple already-built graphs into one, entities first (across
    all graphs) then relations then events, since SemanticGraph.add_relation/
    add_event require their referenced entities to already be present.
    """
    merged = SemanticGraph()
    for graph in graphs:
        for entity in graph.entities:
            merged.add_entity(entity)
    for graph in graphs:
        for relation in graph.relations:
            merged.add_relation(relation)
    for graph in graphs:
        for event in graph.events:
            merged.add_event(event)
    return merged


class KnowledgeBase:
    """A persistent, incrementally-updatable knowledge graph.

    Wraps a SemanticGraph plus the embedding model/similarity settings used
    to canonicalize it, and the storage functions to save/load it as a
    .noema bundle.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        spacy_model: str = SPACY_DEFAULT_MODEL,
    ) -> None:
        self._graph = SemanticGraph()
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        # Selects which spaCy pipeline does NER and dependency parsing for
        # every document this KnowledgeBase ever ingests -- see
        # compress()'s spacy_model docstring for what switching this does
        # and does not buy you (entity extraction becomes genuinely
        # multilingual-capable; relation extraction stays English-tuned).
        self._spacy_model = spacy_model

        # Cached fact (relation) embedding index for retrieve(). Lazily
        # built on first retrieve() call, invalidated on every ingest()
        # (the graph -- and therefore the facts -- changed), and restored
        # cheaply on load() by reusing what load_bundle() already computed
        # rather than re-embedding every fact from scratch right after a load.
        self._fact_ids: list[str] | None = None
        self._fact_index = None

    def _invalidate_fact_index(self) -> None:
        self._fact_ids = None
        self._fact_index = None

    def _ensure_fact_index(self):
        if self._fact_index is None:
            self._fact_ids, _embeddings, self._fact_index = build_fact_index(
                self._graph, model_name=self._embedding_model
            )
        return self._fact_ids, self._fact_index

    def ingest(self, documents: str | Document | list[str | Document]) -> IngestReport:
        """Parse, canonicalize, and merge new documents into the knowledge base.

        Accepts a single string, a single Document, or a list mixing either
        -- matching the flexibility of noema.compress()'s plain-text input
        while still allowing pre-built Document objects (with metadata) when
        the caller has them.
        """
        docs = _normalize_documents(documents)

        new_graph = SemanticGraph()
        for document in docs:
            entities = extract_entities(document, model_name=self._spacy_model)
            relations = extract_relations(document, entities, model_name=self._spacy_model)
            new_graph.add_document(
                ParsedDocument(document=document, entities=entities, relations=relations)
            )

        entities_before = len(self._graph.entities)
        relations_before = len(self._graph.relations)
        entities_parsed = len(new_graph.entities)

        combined = _merge_graphs(self._graph, new_graph)
        self._graph = canonicalize(
            combined,
            similarity_threshold=self._similarity_threshold,
            model_name=self._embedding_model,
        )
        self._invalidate_fact_index()

        return IngestReport(
            documents_ingested=len(docs),
            entities_before=entities_before,
            entities_after=len(self._graph.entities),
            entities_parsed=entities_parsed,
            relations_before=relations_before,
            relations_after=len(self._graph.relations),
        )

    def save(self, path: str) -> None:
        """Write this knowledge base to a .noema bundle at `path`."""
        save_bundle(
            path, self._graph, model_name=self._embedding_model, spacy_model=self._spacy_model
        )

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        """Load a knowledge base from a .noema bundle at `path`."""
        loaded = load_bundle(path)
        kb = cls(
            embedding_model=loaded.manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            spacy_model=loaded.manifest.get("spacy_model") or SPACY_DEFAULT_MODEL,
        )
        kb._graph = loaded.graph
        # The bundle already carried a fully-built fact index -- reuse it
        # directly rather than re-embedding every fact via _ensure_fact_index()
        # on the first retrieve() call after a load.
        kb._fact_ids = loaded.fact_ids
        kb._fact_index = loaded.index
        return kb

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 1,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        format: str = "notation",
        token_model: str = DEFAULT_TOKEN_MODEL,
    ) -> RetrievalResult:
        """Hybrid retrieval: vector search over facts, then graph expansion,
        then notation assembly under a token budget -- the full "vector
        search -> graph expansion -> relevant sub-graph -> notation"
        pipeline from the project plan's workflow diagram.

        `format` controls what `.text` contains: "notation" (default, and
        the token-efficient choice meant for LLM context injection) or
        "natural_language" (decoded, for cases where the consuming prompt
        expects prose rather than compact notation).
        """
        if format not in ("notation", "natural_language"):
            raise ValueError(
                f"Unknown format {format!r}; expected 'notation' or 'natural_language'"
            )

        fact_ids, index = self._ensure_fact_index()
        matches = search_facts(
            query, fact_ids, index, top_k=top_k, model_name=self._embedding_model
        )

        seed_relation_ids = [match.relation_id for match in matches]
        expanded = expand_from_facts(self._graph, seed_relation_ids, hops=hops)
        assembled = assemble(
            self._graph, expanded, max_tokens=max_tokens, token_model=token_model
        )

        text = decode(assembled.notation) if format == "natural_language" else assembled.notation

        return RetrievalResult(
            text=text,
            notation=assembled.notation,
            entity_ids=assembled.entity_ids,
            relation_ids=assembled.relation_ids,
            token_count=assembled.token_count,
            truncated=assembled.truncated,
            fact_matches=matches,
        )

    @property
    def graph(self) -> SemanticGraph:
        return self._graph

    def __len__(self) -> int:
        return len(self._graph.entities)
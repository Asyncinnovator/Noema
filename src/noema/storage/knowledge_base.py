"""KnowledgeBase: a graph-backed, persistent, incrementally-updatable store.

    kb = noema.KnowledgeBase()
    kb.ingest(documents)             # parses, dedupes, canonicalizes
    kb.save("company_kb.noema")
    ...
    kb = noema.KnowledgeBase.load("company_kb.noema")
    kb.ingest(new_documents)         # new facts merge into existing entities

Each .ingest() call re-runs canonicalize() over the union of the existing
graph and the newly parsed documents, so a repeated entity merges into the
same node instead of duplicating. This re-canonicalizes the whole knowledge
base on every ingest, so cost grows with total KB size, not just the new
batch.

For large or streamed document sets, ingest_stream() chunks the work into
fixed-size batches instead of parsing everything into memory at once. For
knowledge bases too large to keep as a single .noema bundle, sync_to_backend()
and load_from_backend() move the graph to/from a persistent GraphBackend
(e.g. Neo4j or an RDF store) instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
from noema.storage.backends import GraphBackend, read_graph, write_graph
from noema.storage.bundle import DEFAULT_EMBEDDING_MODEL, build_fact_index, load_bundle, save_bundle

# Default chunk size for ingest_stream(): how many documents are parsed and
# canonicalized together before moving to the next chunk.
DEFAULT_BATCH_SIZE = 100


@dataclass
class IngestReport:
    """Summary of what one .ingest() call did."""

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

        Not simply `entities_before - entities_after`, since ingest()
        usually grows the graph with new entities alongside any merges.
        Compares the raw candidate pool (existing + newly parsed, before
        canonicalization) against what remained after.
        """
        raw_candidate_count = self.entities_before + self.entities_parsed
        return max(0, raw_candidate_count - self.entities_after)


@dataclass
class RetrievalResult:
    """Result of kb.retrieve(query, ...).

    `text` depends on `format`: raw notation for format="notation"
    (default, token-efficient), or a decoded natural-language rendering
    for format="natural_language". `notation` is always the raw notation
    regardless of format.
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
    all graphs) then relations then events, since relations/events require
    their referenced entities to already be present.
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
        # spaCy pipeline used for NER/dependency parsing on every document
        # this KnowledgeBase ingests.
        self._spacy_model = spacy_model

        # Cached fact (relation) embedding index for retrieve(). Lazily
        # built on first retrieve() call, invalidated on every ingest(),
        # and restored directly on load() from what load_bundle() computed.
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

        Accepts a single string, a single Document, or a list mixing either.
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

    def ingest_stream(
        self,
        documents: Iterable[str | Document],
        batch_size: int = DEFAULT_BATCH_SIZE,
        checkpoint_path: str | None = None,
        on_batch: Callable[[int, IngestReport], None] | None = None,
    ) -> IngestReport:
        """Ingest a large or streamed set of documents in fixed-size batches.

        `documents` may be any iterable, including a generator that reads
        from disk lazily, so the whole set never needs to be resident in
        memory at once. Each batch is parsed and canonicalized with a
        regular ingest() call, so cost per batch still grows with total KB
        size, but the batch itself stays a fixed, bounded size instead of
        the whole document set being parsed into memory in one pass.

        If `checkpoint_path` is given, the knowledge base is saved there
        after every batch, so a crash partway through only loses the
        batch in progress. `on_batch`, if given, is called after each
        batch as `on_batch(batch_index, batch_report)`.

        Returns one IngestReport summarizing totals across all batches.
        """
        entities_before_all = len(self._graph.entities)
        relations_before_all = len(self._graph.relations)
        total_docs = 0
        total_entities_parsed = 0
        batch_index = 0
        batch: list[str | Document] = []

        def flush(pending: list[str | Document]) -> None:
            nonlocal batch_index, total_docs, total_entities_parsed
            if not pending:
                return
            report = self.ingest(pending)
            total_docs += report.documents_ingested
            total_entities_parsed += report.entities_parsed
            batch_index += 1
            if checkpoint_path is not None:
                self.save(checkpoint_path)
            if on_batch is not None:
                on_batch(batch_index, report)

        for document in documents:
            batch.append(document)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        flush(batch)

        return IngestReport(
            documents_ingested=total_docs,
            entities_before=entities_before_all,
            entities_after=len(self._graph.entities),
            entities_parsed=total_entities_parsed,
            relations_before=relations_before_all,
            relations_after=len(self._graph.relations),
        )

    def save(self, path: str) -> None:
        """Write this knowledge base to a .noema bundle at `path`."""
        save_bundle(
            path, self._graph, model_name=self._embedding_model, spacy_model=self._spacy_model
        )

    @classmethod
    def load(cls, path: str) -> KnowledgeBase:
        """Load a knowledge base from a .noema bundle at `path`."""
        loaded = load_bundle(path)
        kb = cls(
            embedding_model=loaded.manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            spacy_model=loaded.manifest.get("spacy_model") or SPACY_DEFAULT_MODEL,
        )
        kb._graph = loaded.graph
        # Reuse the bundle's prebuilt fact index instead of re-embedding
        # on the first retrieve() call after a load.
        kb._fact_ids = loaded.fact_ids
        kb._fact_index = loaded.index
        return kb

    def sync_to_backend(self, backend: GraphBackend) -> None:
        """Push this knowledge base's full current graph to a persistent
        backend (e.g. Neo4jBackend, RDFBackend), replacing whatever that
        backend held before. Use this once a knowledge base outgrows what's
        comfortable to keep as a single .noema bundle.
        """
        write_graph(backend, self._graph)

    @classmethod
    def load_from_backend(
        cls,
        backend: GraphBackend,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        spacy_model: str = SPACY_DEFAULT_MODEL,
    ) -> KnowledgeBase:
        """Build a knowledge base by reading a persistent backend's full
        contents into memory.
        """
        kb = cls(embedding_model=embedding_model, spacy_model=spacy_model)
        kb._graph = read_graph(backend)
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
        then notation assembly under a token budget.

        `format` controls what `.text` contains: "notation" (default,
        token-efficient) or "natural_language" (decoded prose).
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
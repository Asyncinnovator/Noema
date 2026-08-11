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

from dataclasses import dataclass

from noema.canonicalize.dedup import DEFAULT_SIMILARITY_THRESHOLD, canonicalize
from noema.graph.builder import SemanticGraph
from noema.parser.entities import extract_entities
from noema.parser.relations import extract_relations
from noema.schema import Document, ParsedDocument
from noema.storage.bundle import DEFAULT_EMBEDDING_MODEL, load_bundle, save_bundle


@dataclass
class IngestReport:
    """Summary of what one .ingest() call did -- the "dedup stats" the
    project plan's manifest.json describes, surfaced back to the caller
    rather than only written to disk.
    """

    documents_ingested: int
    entities_before: int
    entities_after: int
    relations_before: int
    relations_after: int

    @property
    def entities_merged(self) -> int:
        """How many entities were absorbed into existing/other entities
        during this ingest, rather than staying as their own node.
        """
        return max(0, self.entities_before - self.entities_after) if self.entities_before else 0


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
    ) -> None:
        self._graph = SemanticGraph()
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold

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
            entities = extract_entities(document)
            relations = extract_relations(document, entities)
            new_graph.add_document(
                ParsedDocument(document=document, entities=entities, relations=relations)
            )

        entities_before = len(self._graph.entities)
        relations_before = len(self._graph.relations)

        combined = _merge_graphs(self._graph, new_graph)
        self._graph = canonicalize(
            combined,
            similarity_threshold=self._similarity_threshold,
            model_name=self._embedding_model,
        )

        return IngestReport(
            documents_ingested=len(docs),
            entities_before=entities_before,
            entities_after=len(self._graph.entities),
            relations_before=relations_before,
            relations_after=len(self._graph.relations),
        )

    def save(self, path: str) -> None:
        """Write this knowledge base to a .noema bundle at `path`."""
        save_bundle(path, self._graph, model_name=self._embedding_model)

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        """Load a knowledge base from a .noema bundle at `path`."""
        loaded = load_bundle(path)
        kb = cls(embedding_model=loaded.manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
        kb._graph = loaded.graph
        return kb

    @property
    def graph(self) -> SemanticGraph:
        return self._graph

    def __len__(self) -> int:
        return len(self._graph.entities)
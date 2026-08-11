"""The .noema bundle format: a zip archive with four entries.

    company_kb.noema
    |-- graph.json        canonical entities + relations + events, with
    |                     per-mention provenance and confidence (already
    |                     present on Entity/Relation/Event -- this module
    |                     just serializes what schema.py already tracks)
    |-- embeddings.bin    float32 vectors, one row per relation ("fact"),
    |                     row order matching manifest.json's fact_ids
    |-- index.faiss       a faiss IndexFlatIP over those same vectors, for
    |                     fast similarity search at retrieval time (Phase 4)
    `-- manifest.json     schema version, counts, embedding model name/dim,
                          fact_ids (the row-order -> Relation.id mapping),
                          source document ids

Per the project plan: "embeddings.bin / index.faiss index the compact
notation form of each fact (not raw text) -- smaller and more precise to
search than embedding whole paragraphs." A "fact" here is one Relation.
Each fact's embedded text is that single relation rendered through
notation.encoder in isolation (just its two entities + itself) -- not the
relation's raw source sentence -- so what gets indexed is exactly the same
compact representation compress() produces, keeping the token-savings and
retrieval-quality properties on the same representation rather than two
separate ones (see project plan, "Why Noema Is Better").

IndexFlatIP (exact inner-product search over normalized vectors, i.e. exact
cosine similarity) is used rather than an approximate index -- correctness
over speed for now. Swapping in an approximate index (IVF, HNSW) for large
knowledge bases is a Phase 7 (scale) concern, not Phase 3's.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from noema.graph.builder import SemanticGraph
from noema.notation.encoder import encode_graph
from noema.schema import Entity, Event, Relation

SCHEMA_VERSION = "1"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Dimension of the default model above. Used only for the zero-fact edge
# case, so an index can still be constructed without invoking the model
# just to learn its output size for an empty knowledge base.
DEFAULT_EMBEDDING_DIM = 384

_GRAPH_ENTRY = "graph.json"
_EMBEDDINGS_ENTRY = "embeddings.bin"
_INDEX_ENTRY = "index.faiss"
_MANIFEST_ENTRY = "manifest.json"


@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@dataclass
class LoadedBundle:
    graph: SemanticGraph
    fact_ids: list[str]
    embeddings: np.ndarray
    index: faiss.Index
    manifest: dict


# -- graph <-> dict -----------------------------------------------------


def graph_to_dict(graph: SemanticGraph) -> dict:
    """Serialize a SemanticGraph's entities/relations/events to plain dicts,
    suitable for json.dumps. Uses each schema type's own pydantic
    serialization rather than hand-rolling field lists, so this stays in
    sync with schema.py automatically as fields are added.
    """
    return {
        "entities": [e.model_dump(mode="json") for e in graph.entities],
        "relations": [r.model_dump(mode="json") for r in graph.relations],
        "events": [ev.model_dump(mode="json") for ev in graph.events],
    }


def graph_from_dict(data: dict) -> SemanticGraph:
    """Inverse of graph_to_dict. Entities are added before relations/events
    that reference them, matching SemanticGraph.add_relation/add_event's
    requirement that referenced entities already exist in the graph.
    """
    graph = SemanticGraph()
    for entity_data in data.get("entities", []):
        graph.add_entity(Entity.model_validate(entity_data))
    for relation_data in data.get("relations", []):
        graph.add_relation(Relation.model_validate(relation_data))
    for event_data in data.get("events", []):
        graph.add_event(Event.model_validate(event_data))
    return graph


# -- fact embeddings ------------------------------------------------------


def _fact_notation(graph: SemanticGraph, relation: Relation) -> str:
    """Render one relation, in isolation, as its own tiny notation block --
    just its subject and object entities plus itself. This is what gets
    embedded for that fact: the same compact form compress() would produce
    for it, not its raw source sentence.
    """
    subject = graph.get_entity(relation.subject_id)
    obj = graph.get_entity(relation.object_id)
    isolated = SemanticGraph()
    isolated.add_entity(subject)
    isolated.add_entity(obj)
    isolated.add_relation(relation)
    return encode_graph(isolated)


def build_fact_index(
    graph: SemanticGraph, model_name: str = DEFAULT_EMBEDDING_MODEL
) -> tuple[list[str], np.ndarray, faiss.Index]:
    """Embed every relation ("fact") in the graph and build a faiss index
    over the result.

    Returns (fact_ids, embeddings, index) where fact_ids[i] is the
    Relation.id whose vector is embeddings[i] / index row i -- this mapping
    is what manifest.json's fact_ids field preserves across save/load, since
    faiss indices don't carry external ids on their own with IndexFlatIP.
    """
    relations = graph.relations
    if not relations:
        empty = np.zeros((0, DEFAULT_EMBEDDING_DIM), dtype="float32")
        return [], empty, faiss.IndexFlatIP(DEFAULT_EMBEDDING_DIM)

    texts = [_fact_notation(graph, r) for r in relations]
    model = _load_model(model_name)
    embeddings = np.asarray(model.encode(texts, normalize_embeddings=True), dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    fact_ids = [r.id for r in relations]
    return fact_ids, embeddings, index


# -- bundle pack / unpack --------------------------------------------------


def save_bundle(path: str, graph: SemanticGraph, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    """Write a SemanticGraph to a .noema bundle at `path`."""
    fact_ids, embeddings, index = build_fact_index(graph, model_name)

    source_document_ids = sorted(
        {mention.doc_id for entity in graph.entities for mention in entity.mentions}
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_name,
        "embedding_dim": embeddings.shape[1] if embeddings.size else DEFAULT_EMBEDDING_DIM,
        "entity_count": len(graph.entities),
        "relation_count": len(graph.relations),
        "event_count": len(graph.events),
        "fact_count": len(fact_ids),
        "fact_ids": fact_ids,
        "source_document_ids": source_document_ids,
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(_GRAPH_ENTRY, json.dumps(graph_to_dict(graph)))
        bundle.writestr(_EMBEDDINGS_ENTRY, embeddings.tobytes())
        bundle.writestr(_INDEX_ENTRY, faiss.serialize_index(index).tobytes())
        bundle.writestr(_MANIFEST_ENTRY, json.dumps(manifest, indent=2))


def load_bundle(path: str) -> LoadedBundle:
    """Read a .noema bundle at `path` back into its parts."""
    with zipfile.ZipFile(path, "r") as bundle:
        graph_data = json.loads(bundle.read(_GRAPH_ENTRY))
        manifest = json.loads(bundle.read(_MANIFEST_ENTRY))
        embeddings_bytes = bundle.read(_EMBEDDINGS_ENTRY)
        index_bytes = bundle.read(_INDEX_ENTRY)

    graph = graph_from_dict(graph_data)

    dim = manifest["embedding_dim"]
    embeddings = (
        np.frombuffer(embeddings_bytes, dtype="float32").reshape(-1, dim)
        if embeddings_bytes
        else np.zeros((0, dim), dtype="float32")
    )
    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype="uint8"))

    return LoadedBundle(
        graph=graph,
        fact_ids=manifest["fact_ids"],
        embeddings=embeddings,
        index=index,
        manifest=manifest,
    )
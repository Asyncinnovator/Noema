"""The .noema bundle format: a zip archive with four entries.

    company_kb.noema
    |-- graph.json        canonical entities + relations + events, with
    |                     per-mention provenance and confidence
    |-- embeddings.bin    float32 vectors, one row per relation ("fact"),
    |                     row order matching manifest.json's fact_ids
    |-- index.faiss       a faiss IndexFlatIP over those same vectors, for
    |                     similarity search at retrieval time
    `-- manifest.json     schema version, counts, embedding model name/dim,
                          fact_ids (row-order -> Relation.id mapping),
                          source document ids

A "fact" is one Relation. Each fact's embedded text is that relation
rendered through notation.encoder in isolation (just its two entities plus
itself), not its raw source sentence. IndexFlatIP (exact inner-product
search over normalized vectors) is used rather than an approximate index,
favoring correctness over speed.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from noema.graph.builder import SemanticGraph
from noema.notation.encoder import encode_graph
from noema.schema import Entity, Event, Relation
from noema.storage._atomic import atomic_write

SCHEMA_VERSION = "1"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Dimension of the default model above, used for the zero-fact edge case so
# an empty index can be built without invoking the model.
DEFAULT_EMBEDDING_DIM = 384

_GRAPH_ENTRY = "graph.json"
_EMBEDDINGS_ENTRY = "embeddings.bin"
_INDEX_ENTRY = "index.faiss"
_MANIFEST_ENTRY = "manifest.json"

# Upper bounds on how much a single bundle entry may expand to in memory.
# zip deflate reaches ~1000:1, so without a cap a ~100 KB file can force a
# multi-hundred-MB read. Both are overridable per load_bundle() call for
# legitimately huge knowledge bases.
DEFAULT_MAX_JSON_BYTES = 512 * 1024**2
DEFAULT_MAX_BINARY_BYTES = 4 * 1024**3
_MAX_EMBEDDING_DIM = 8192


class BundleError(ValueError):
    """A .noema bundle is malformed, internally inconsistent, or exceeds the
    size limits. Always raised in place of the KeyError / AttributeError /
    BadZipFile / faiss RuntimeError a hostile or corrupted file would
    otherwise surface as.
    """


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
    suitable for json.dumps.
    """
    return {
        "entities": [e.model_dump(mode="json") for e in graph.entities],
        "relations": [r.model_dump(mode="json") for r in graph.relations],
        "events": [ev.model_dump(mode="json") for ev in graph.events],
    }


def graph_from_dict(data: dict) -> SemanticGraph:
    """Inverse of graph_to_dict. Entities are added before relations/events
    that reference them, since SemanticGraph requires referenced entities
    to already exist in the graph.
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
    """Render one relation, in isolation, as its own tiny notation block
    (its subject and object entities plus itself). This is what gets
    embedded for that fact, not its raw source sentence.
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
    Relation.id whose vector is embeddings[i] / index row i. This mapping
    is preserved in manifest.json's fact_ids field across save/load, since
    IndexFlatIP doesn't carry external ids on its own.
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


def save_bundle(
    path: str,
    graph: SemanticGraph,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    spacy_model: str | None = None,
) -> None:
    """Write a SemanticGraph to a .noema bundle at `path`.

    `spacy_model` is recorded in the manifest as metadata about which
    spaCy pipeline produced this graph's entities/relations, so
    KnowledgeBase.load() can restore the same setting for future ingests.
    """
    fact_ids, embeddings, index = build_fact_index(graph, model_name)

    source_document_ids = sorted(
        {mention.doc_id for entity in graph.entities for mention in entity.mentions}
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model": model_name,
        "spacy_model": spacy_model,
        "embedding_dim": embeddings.shape[1] if embeddings.size else DEFAULT_EMBEDDING_DIM,
        "entity_count": len(graph.entities),
        "relation_count": len(graph.relations),
        "event_count": len(graph.events),
        "fact_count": len(fact_ids),
        "fact_ids": fact_ids,
        "source_document_ids": source_document_ids,
    }

    # Everything is serialized before the file is touched, and written via a
    # temp file + atomic rename, so a failure at any point leaves an existing
    # bundle at `path` intact instead of replacing it with a truncated one.
    graph_json = json.dumps(graph_to_dict(graph))
    embeddings_bytes = embeddings.tobytes()
    index_bytes = faiss.serialize_index(index).tobytes()
    manifest_json = json.dumps(manifest, indent=2)

    with atomic_write(path) as fh, zipfile.ZipFile(
        fh, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        bundle.writestr(_GRAPH_ENTRY, graph_json)
        bundle.writestr(_EMBEDDINGS_ENTRY, embeddings_bytes)
        bundle.writestr(_INDEX_ENTRY, index_bytes)
        bundle.writestr(_MANIFEST_ENTRY, manifest_json)


def _read_entry(bundle: zipfile.ZipFile, name: str, limit: int) -> bytes:
    """Read one bundle entry, refusing missing entries and ones that would
    expand past `limit` bytes. The declared size is checked first (Python's
    zipfile never yields more than the declared size), and the read itself
    is bounded as well.
    """
    try:
        info = bundle.getinfo(name)
    except KeyError:
        raise BundleError(f"Not a valid .noema bundle: missing entry {name!r}") from None
    if info.file_size > limit:
        raise BundleError(
            f"Bundle entry {name!r} expands to {info.file_size:,} bytes, over the "
            f"{limit:,}-byte limit"
        )
    with bundle.open(info) as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise BundleError(f"Bundle entry {name!r} is larger than the {limit:,}-byte limit")
    return data


def _load_json(data: bytes, name: str) -> dict:
    try:
        parsed = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise BundleError(f"Bundle entry {name!r} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BundleError(f"Bundle entry {name!r} must be a JSON object")
    return parsed


def _validate_manifest(manifest: dict) -> None:
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise BundleError(
            f"Unsupported bundle schema_version {version!r}; this version of noema reads "
            f"{SCHEMA_VERSION!r}"
        )
    dim = manifest.get("embedding_dim")
    if not isinstance(dim, int) or isinstance(dim, bool) or not 0 < dim <= _MAX_EMBEDDING_DIM:
        raise BundleError(f"Invalid embedding_dim in manifest: {dim!r}")
    fact_ids = manifest.get("fact_ids")
    if not isinstance(fact_ids, list) or not all(isinstance(f, str) for f in fact_ids):
        raise BundleError("Manifest fact_ids must be a list of strings")
    if len(set(fact_ids)) != len(fact_ids):
        raise BundleError("Manifest fact_ids contains duplicates")
    for key in ("embedding_model", "spacy_model"):
        value = manifest.get(key)
        if value is not None and not isinstance(value, str):
            raise BundleError(f"Manifest {key} must be a string or null")


def load_bundle(
    path: str,
    *,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_binary_bytes: int = DEFAULT_MAX_BINARY_BYTES,
) -> LoadedBundle:
    """Read a .noema bundle at `path` back into its parts.

    The file is validated rather than trusted: entry sizes are capped, the
    manifest is schema-checked, and the graph, fact ids, embedding matrix and
    faiss index must agree with each other. Anything wrong raises
    `BundleError`. Note that faiss's own deserializer is not hardened against
    hostile input, so bundles from untrusted sources should still not be
    loaded.
    """
    try:
        with zipfile.ZipFile(path, "r") as bundle:
            graph_data = _load_json(_read_entry(bundle, _GRAPH_ENTRY, max_json_bytes), _GRAPH_ENTRY)
            manifest = _load_json(
                _read_entry(bundle, _MANIFEST_ENTRY, max_json_bytes), _MANIFEST_ENTRY
            )
            embeddings_bytes = _read_entry(bundle, _EMBEDDINGS_ENTRY, max_binary_bytes)
            index_bytes = _read_entry(bundle, _INDEX_ENTRY, max_binary_bytes)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"Not a valid .noema bundle (bad zip file): {exc}") from exc

    _validate_manifest(manifest)
    dim = manifest["embedding_dim"]
    fact_ids = manifest["fact_ids"]

    try:
        graph = graph_from_dict(graph_data)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BundleError(f"Bundle graph.json is malformed: {exc}") from exc

    relation_ids = {r.id for r in graph.relations}
    unknown = [f for f in fact_ids if f not in relation_ids]
    if unknown:
        raise BundleError(f"Manifest fact_ids reference unknown relations: {unknown[:3]}")

    expected_bytes = len(fact_ids) * dim * 4
    if len(embeddings_bytes) != expected_bytes:
        raise BundleError(
            f"embeddings.bin is {len(embeddings_bytes):,} bytes; expected {expected_bytes:,} "
            f"for {len(fact_ids)} facts x {dim} dims"
        )
    embeddings = np.frombuffer(embeddings_bytes, dtype="float32").reshape(-1, dim)

    try:
        index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype="uint8"))
    except Exception as exc:  # faiss raises bare RuntimeError / SWIG errors
        raise BundleError(f"Bundle index.faiss could not be read: {exc}") from exc
    if not isinstance(index, faiss.IndexFlat) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise BundleError("Bundle index.faiss is not an inner-product flat index")
    if index.d != dim or index.ntotal != len(fact_ids):
        raise BundleError(
            f"index.faiss holds {index.ntotal} vectors of dim {index.d}; manifest says "
            f"{len(fact_ids)} facts of dim {dim}"
        )

    return LoadedBundle(
        graph=graph,
        fact_ids=fact_ids,
        embeddings=embeddings,
        index=index,
        manifest=manifest,
    )
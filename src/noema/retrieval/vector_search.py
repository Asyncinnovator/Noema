"""Vector search over a knowledge base's fact embeddings.

Given a query string, embeds it with the same model used to build the fact
index (storage.bundle.build_fact_index) and returns the top-k most similar
facts (relations) by cosine similarity. Independent of KnowledgeBase: only
needs a query string plus a faiss index and the fact_ids list mapping
index rows back to Relation.id.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@dataclass
class FactMatch:
    """One search hit: a fact (Relation.id) and its similarity to the query."""

    relation_id: str
    score: float


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a query string, normalized the same way fact embeddings are
    (see build_fact_index), so cosine similarity via inner product is
    directly comparable between the two.
    """
    model = _load_model(model_name)
    vector = model.encode([query], normalize_embeddings=True)
    return np.asarray(vector, dtype="float32")


def search_facts(
    query: str,
    fact_ids: list[str],
    index: faiss.Index,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> list[FactMatch]:
    """Return the top-k facts (relations) most similar to `query`, ranked
    by descending similarity score.

    Returns fewer than top_k results if the knowledge base has fewer facts
    than top_k. faiss pads short result sets with index -1, which is
    filtered out here.
    """
    if not fact_ids or index.ntotal == 0:
        return []

    query_vector = embed_query(query, model_name)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)

    matches = []
    for score, row in zip(scores[0], indices[0]):
        if row == -1:
            continue
        matches.append(FactMatch(relation_id=fact_ids[row], score=float(score)))
    return matches
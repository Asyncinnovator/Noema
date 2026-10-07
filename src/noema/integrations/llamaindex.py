"""LlamaIndex integration: `NoemaLlamaIndexRetriever`, a `BaseRetriever`
backed by a Noema `KnowledgeBase`.

Requires the optional `llamaindex` extra:

    pip install noema[llamaindex]

As with the LangChain integration, the `llama-index-core` import below is
wrapped in a try/except so this module always imports cleanly; only
instantiating the class without that extra raises `ImportError`.

Example
-------
    from noema import KnowledgeBase
    from noema.integrations.llamaindex import NoemaLlamaIndexRetriever

    kb = KnowledgeBase.load("company_kb.noema")
    retriever = NoemaLlamaIndexRetriever(kb=kb, top_k=5, hops=1, max_tokens=200)

    nodes = retriever.retrieve("Who is the CEO of Google?")
    print(nodes[0].get_content())

    # Drop into a LlamaIndex query engine:
    #   from llama_index.core.query_engine import RetrieverQueryEngine
    #   query_engine = RetrieverQueryEngine.from_args(retriever)
"""

from __future__ import annotations

from typing import Any

from noema.storage.knowledge_base import KnowledgeBase

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _LLAMA_INDEX_AVAILABLE = False
    BaseRetriever = object  # type: ignore[assignment,misc]
    NodeWithScore = Any  # type: ignore[assignment,misc]
    QueryBundle = Any  # type: ignore[assignment,misc]
    TextNode = Any  # type: ignore[assignment,misc]

_INSTALL_HINT = (
    "NoemaLlamaIndexRetriever requires the optional `llamaindex` extra. "
    "Install it with:\n\n    pip install noema[llamaindex]\n"
)


class NoemaLlamaIndexRetriever(BaseRetriever):  # type: ignore[misc]
    """Adapts `KnowledgeBase.retrieve()` to LlamaIndex's `BaseRetriever`
    interface (`_retrieve(QueryBundle) -> list[NodeWithScore]`).

    Like the LangChain adapter, this wraps Noema's already-assembled,
    token-budgeted sub-graph as a single `TextNode` rather than splitting
    it into multiple nodes -- the de-duplication and budget assembly
    already happened inside `kb.retrieve()`. The node's score is the best
    (highest) fact-similarity score from `result.fact_matches`, or `1.0`
    when there were no scored matches but the budget assembly still
    produced text (e.g. a single-fact graph).
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        top_k: int = 5,
        hops: int = 1,
        max_tokens: int = 200,
        format: str = "notation",
        **kwargs: Any,
    ) -> None:
        if not _LLAMA_INDEX_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        self._kb = kb
        self._top_k = top_k
        self._hops = hops
        self._max_tokens = max_tokens
        self._format = format
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: "QueryBundle") -> list["NodeWithScore"]:
        result = self._kb.retrieve(
            query_bundle.query_str,
            top_k=self._top_k,
            hops=self._hops,
            max_tokens=self._max_tokens,
            format=self._format,
        )
        if not result.text:
            return []

        best_score = max((m.score for m in result.fact_matches), default=1.0)
        node = TextNode(
            text=result.text,
            metadata={
                "entity_ids": sorted(result.entity_ids),
                "relation_ids": sorted(result.relation_ids),
                "token_count": result.token_count,
                "truncated": result.truncated,
                "source": "noema",
            },
        )
        return [NodeWithScore(node=node, score=best_score)]

    async def _aretrieve(self, query_bundle: "QueryBundle") -> list["NodeWithScore"]:
        # Synchronous under the hood for the same reason as the LangChain
        # adapter's async path -- no I/O here to actually await.
        return self._retrieve(query_bundle)
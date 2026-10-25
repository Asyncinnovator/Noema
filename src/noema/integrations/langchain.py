"""LangChain integration: `NoemaRetriever`, a `BaseRetriever` backed by a
Noema `KnowledgeBase`.

Requires the optional `langchain` extra:

    pip install noema-rag[langchain]

Importing this module never fails for lack of that extra -- the
`langchain-core` import is wrapped in a try/except, and the class body
falls back to plain `object`. Only instantiating `NoemaRetriever` without
`langchain-core` installed raises `ImportError` with the install hint.

Example
-------
    from noema import KnowledgeBase
    from noema.integrations.langchain import NoemaRetriever

    kb = KnowledgeBase.load("company_kb.noema")
    retriever = NoemaRetriever(kb=kb, top_k=5, hops=1, max_tokens=200)

    docs = retriever.invoke("Who is the CEO of Google?")
    print(docs[0].page_content)   # compact notation, ready for prompt injection
"""

from __future__ import annotations

from typing import Any

from noema.storage.knowledge_base import KnowledgeBase

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _LANGCHAIN_AVAILABLE = False
    BaseRetriever = object  # type: ignore[assignment,misc]
    Document = Any  # type: ignore[assignment,misc]
    CallbackManagerForRetrieverRun = Any  # type: ignore[assignment,misc]

_INSTALL_HINT = (
    "NoemaRetriever requires the optional `langchain` extra. "
    "Install it with:\n\n    pip install noema-rag[langchain]\n"
)


class NoemaRetriever(BaseRetriever):  # type: ignore[misc]
    """Adapts `KnowledgeBase.retrieve()` to LangChain's `BaseRetriever`
    interface, so a Noema knowledge base can be dropped into any LangChain
    chain or agent that expects a retriever.

    Each call returns at most one `Document`, since retrieval already
    assembles a single token-budgeted sub-graph rather than independent
    chunks. `page_content` is `result.text`; `metadata` carries
    `entity_ids`, `relation_ids`, `token_count`, `truncated`, and
    `fact_matches`.
    """

    kb: KnowledgeBase
    top_k: int = 5
    hops: int = 1
    max_tokens: int = 200
    format: str = "notation"

    # pydantic reads model_config as a plain dict, not a ClassVar.
    model_config = {"arbitrary_types_allowed": True}  # noqa: RUF012

    def __init__(self, **data: Any) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        super().__init__(**data)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._retrieve_as_documents(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # KnowledgeBase.retrieve() is synchronous (no I/O to await on).
        return self._retrieve_as_documents(query)

    def _retrieve_as_documents(self, query: str) -> list[Document]:
        result = self.kb.retrieve(
            query,
            top_k=self.top_k,
            hops=self.hops,
            max_tokens=self.max_tokens,
            format=self.format,
        )
        if not result.text:
            return []
        return [
            Document(
                page_content=result.text,
                metadata={
                    "entity_ids": sorted(result.entity_ids),
                    "relation_ids": sorted(result.relation_ids),
                    "token_count": result.token_count,
                    "truncated": result.truncated,
                    "fact_matches": [
                        {"relation_id": m.relation_id, "score": m.score}
                        for m in result.fact_matches
                    ],
                    "source": "noema",
                },
            )
        ]
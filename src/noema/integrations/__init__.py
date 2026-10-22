"""Drop-in integrations for existing RAG frameworks.

Submodules import heavy, optional third-party packages (`langchain-core`,
`llama-index-core`, `anthropic`) lazily inside a try/except, so importing
them never fails for a missing framework -- only instantiating the
retriever class does, with a message naming the extra to install. None of
this package is imported by `noema/__init__.py`, so a plain
`pip install noema-rag` never pulls these frameworks in.

    pip install noema-rag[langchain]
    pip install noema-rag[llamaindex]
    pip install noema-rag[anthropic]

    from noema import KnowledgeBase
    from noema.integrations.langchain import NoemaRetriever

    kb = KnowledgeBase.load("company_kb.noema")
    retriever = NoemaRetriever(kb=kb)
    docs = retriever.invoke("Who is Google's CEO?")
"""

from __future__ import annotations
"""Phase 6: drop-in integrations for existing RAG frameworks.

Submodules in this package import heavy, optional third-party packages
(`langchain-core`, `llama-index-core`, `anthropic`) lazily, inside a
try/except at module import time -- importing `noema.integrations.langchain`
or `noema.integrations.llamaindex` never fails just because that framework
isn't installed; only *instantiating* the retriever class does, with a
message telling you which extra to install. None of this package is
imported by `noema/__init__.py`, so a plain `pip install noema` (the basic
install) never pulls any of these frameworks in -- you opt in per
integration:

    pip install noema[langchain]
    pip install noema[llamaindex]
    pip install noema[anthropic]

Usage mirrors the project plan's example:

    from noema import KnowledgeBase
    from noema.integrations.langchain import NoemaRetriever

    kb = KnowledgeBase.load("company_kb.noema")
    retriever = NoemaRetriever(kb=kb)
    docs = retriever.invoke("Who is Google's CEO?")
"""

from __future__ import annotations
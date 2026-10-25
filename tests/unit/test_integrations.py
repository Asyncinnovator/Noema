"""Tests for noema.integrations (the framework integration adapters).

`langchain-core`, `llama-index-core`, and `anthropic` are all optional
third-party packages, so these tests don't assume any are installed:

- `TestLangChainUnavailable` / `TestLlamaIndexUnavailable` run only when the
  respective package is absent, checking the ImportError-with-install-hint
  behavior.
- `TestLangChainRetriever` / `TestLlamaIndexRetriever` run only when the
  respective package is installed, exercising the real adapter classes
  against a mocked `KnowledgeBase`.
- `TestCountTokensOpenAI` / `TestCountTokensAnthropic` mock tiktoken/anthropic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from noema.retrieval.vector_search import FactMatch
from noema.storage.knowledge_base import KnowledgeBase, RetrievalResult

try:
    import langchain_core  # noqa: F401

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

try:
    import llama_index.core  # noqa: F401

    _HAS_LLAMA_INDEX = True
except ImportError:
    _HAS_LLAMA_INDEX = False

try:
    import anthropic  # noqa: F401

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


def _fake_result(text: str = "#1 Google :ORG\n#2 Sundar Pichai :PERSON\n#2 -ceo_of-> #1") -> RetrievalResult:
    return RetrievalResult(
        text=text,
        notation=text,
        entity_ids={"ent_1", "ent_2"},
        relation_ids={"rel_1"},
        token_count=17,
        truncated=False,
        fact_matches=[FactMatch(relation_id="rel_1", score=0.61)],
    )


def _mock_kb(result: RetrievalResult | None = None) -> MagicMock:
    kb = MagicMock(spec=KnowledgeBase)
    kb.retrieve.return_value = result if result is not None else _fake_result()
    return kb


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_HAS_LANGCHAIN, reason="langchain-core is installed")
class TestLangChainUnavailable:
    def test_instantiation_raises_import_error_with_install_hint(self):
        from noema.integrations.langchain import NoemaRetriever

        with pytest.raises(ImportError, match="pip install noema-rag\\[langchain\\]"):
            NoemaRetriever(kb=_mock_kb())

    def test_module_itself_imports_cleanly(self):
        # Importing the module never requires the extra, only instantiating
        # the class.
        import noema.integrations.langchain  # noqa: F401


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
class TestLangChainRetriever:
    def test_returns_one_document_with_result_text(self):
        from noema.integrations.langchain import NoemaRetriever

        kb = _mock_kb()
        retriever = NoemaRetriever(kb=kb, top_k=3, hops=2, max_tokens=150)
        docs = retriever.invoke("Who is the CEO of Google?")

        assert len(docs) == 1
        assert docs[0].page_content == _fake_result().text
        kb.retrieve.assert_called_once_with(
            "Who is the CEO of Google?", top_k=3, hops=2, max_tokens=150, format="notation"
        )

    def test_metadata_carries_retrieval_details(self):
        from noema.integrations.langchain import NoemaRetriever

        retriever = NoemaRetriever(kb=_mock_kb())
        docs = retriever.invoke("query")

        meta = docs[0].metadata
        assert meta["entity_ids"] == ["ent_1", "ent_2"]
        assert meta["relation_ids"] == ["rel_1"]
        assert meta["token_count"] == 17
        assert meta["truncated"] is False
        assert meta["fact_matches"] == [{"relation_id": "rel_1", "score": 0.61}]
        assert meta["source"] == "noema"

    def test_empty_result_text_yields_no_documents(self):
        from noema.integrations.langchain import NoemaRetriever

        empty = _fake_result(text="")
        retriever = NoemaRetriever(kb=_mock_kb(empty))
        docs = retriever.invoke("query with no extractable facts")

        assert docs == []

    def test_async_path_delegates_to_same_logic(self):
        # Driven with asyncio.run() directly rather than pytest-asyncio,
        # which isn't a project dependency.
        import asyncio

        from noema.integrations.langchain import NoemaRetriever

        retriever = NoemaRetriever(kb=_mock_kb())
        docs = asyncio.run(retriever.ainvoke("query"))

        assert len(docs) == 1
        assert docs[0].page_content == _fake_result().text

    def test_format_is_forwarded_to_retrieve(self):
        from noema.integrations.langchain import NoemaRetriever

        kb = _mock_kb()
        retriever = NoemaRetriever(kb=kb, format="natural_language")
        retriever.invoke("query")

        _, kwargs = kb.retrieve.call_args
        assert kwargs["format"] == "natural_language"


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_HAS_LLAMA_INDEX, reason="llama-index-core is installed")
class TestLlamaIndexUnavailable:
    def test_instantiation_raises_import_error_with_install_hint(self):
        from noema.integrations.llamaindex import NoemaLlamaIndexRetriever

        with pytest.raises(ImportError, match="pip install noema-rag\\[llamaindex\\]"):
            NoemaLlamaIndexRetriever(kb=_mock_kb())

    def test_module_itself_imports_cleanly(self):
        import noema.integrations.llamaindex  # noqa: F401


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama-index-core not installed")
class TestLlamaIndexRetriever:
    def test_retrieve_returns_one_scored_node(self):
        from llama_index.core.schema import QueryBundle

        from noema.integrations.llamaindex import NoemaLlamaIndexRetriever

        kb = _mock_kb()
        retriever = NoemaLlamaIndexRetriever(kb=kb, top_k=3, hops=2, max_tokens=150)
        nodes = retriever.retrieve(QueryBundle(query_str="Who is the CEO of Google?"))

        assert len(nodes) == 1
        assert nodes[0].node.get_content() == _fake_result().text
        assert nodes[0].score == pytest.approx(0.61)
        kb.retrieve.assert_called_once_with(
            "Who is the CEO of Google?", top_k=3, hops=2, max_tokens=150, format="notation"
        )

    def test_empty_result_text_yields_no_nodes(self):
        from llama_index.core.schema import QueryBundle

        from noema.integrations.llamaindex import NoemaLlamaIndexRetriever

        empty = _fake_result(text="")
        retriever = NoemaLlamaIndexRetriever(kb=_mock_kb(empty))
        nodes = retriever.retrieve(QueryBundle(query_str="query"))

        assert nodes == []

    def test_no_fact_matches_falls_back_to_score_one(self):
        from llama_index.core.schema import QueryBundle

        from noema.integrations.llamaindex import NoemaLlamaIndexRetriever

        result = _fake_result()
        result.fact_matches = []
        retriever = NoemaLlamaIndexRetriever(kb=_mock_kb(result))
        nodes = retriever.retrieve(QueryBundle(query_str="query"))

        assert nodes[0].score == 1.0


# ---------------------------------------------------------------------------
# Provider token-counting helpers
# ---------------------------------------------------------------------------


class TestCountTokensOpenAI:
    def test_empty_text_returns_zero(self):
        from noema.integrations.tokens import count_tokens_openai

        assert count_tokens_openai("") == 0
        assert count_tokens_openai("   ") == 0

    def test_counts_via_tiktoken_encoding_for_model(self):
        from noema.integrations.tokens import count_tokens_openai

        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = [1, 2, 3, 4]
        with patch("tiktoken.encoding_for_model", return_value=fake_encoding) as enc_for_model:
            count = count_tokens_openai("hello there", model="gpt-4o")

        enc_for_model.assert_called_once_with("gpt-4o")
        assert count == 4

    def test_unknown_model_falls_back_to_cl100k_base(self):
        from noema.integrations.tokens import count_tokens_openai

        fake_encoding = MagicMock()
        fake_encoding.encode.return_value = [1, 2, 3]
        with patch("tiktoken.encoding_for_model", side_effect=KeyError("nope")), patch(
            "tiktoken.get_encoding", return_value=fake_encoding
        ) as get_encoding:
            count = count_tokens_openai("hello", model="some-future-model")

        get_encoding.assert_called_once_with("cl100k_base")
        assert count == 3


class TestCountTokensAnthropic:
    def test_empty_text_returns_zero(self):
        from noema.integrations.tokens import count_tokens_anthropic

        assert count_tokens_anthropic("") == 0

    def test_without_anthropic_package_falls_back_with_warning(self):
        import noema.integrations.tokens as tokens_module

        with (
            patch.object(tokens_module, "_count_tokens_cl100k", return_value=42) as fallback,
            patch.dict("sys.modules", {"anthropic": None}),
            pytest.warns(UserWarning, match="pip install noema-rag\\[anthropic\\]"),
        ):
            count = tokens_module.count_tokens_anthropic("hello world")

        fallback.assert_called_once_with("hello world")
        assert count == 42

    @pytest.mark.skipif(not _HAS_ANTHROPIC, reason="anthropic package not installed")
    def test_uses_anthropic_client_when_available(self):
        from noema.integrations.tokens import count_tokens_anthropic

        fake_response = MagicMock(input_tokens=7)
        fake_client = MagicMock()
        fake_client.messages.count_tokens.return_value = fake_response

        with patch("anthropic.Anthropic", return_value=fake_client) as client_cls:
            count = count_tokens_anthropic("hello world", model="claude-sonnet-5", api_key="sk-test")

        client_cls.assert_called_once_with(api_key="sk-test")
        fake_client.messages.count_tokens.assert_called_once_with(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hello world"}],
        )
        assert count == 7

    @pytest.mark.skipif(not _HAS_ANTHROPIC, reason="anthropic package not installed")
    def test_api_failure_falls_back_with_warning(self):
        import noema.integrations.tokens as tokens_module

        fake_client = MagicMock()
        fake_client.messages.count_tokens.side_effect = RuntimeError("no api key")

        with (
            patch("anthropic.Anthropic", return_value=fake_client),
            patch.object(tokens_module, "_count_tokens_cl100k", return_value=11) as fallback,
            pytest.warns(UserWarning, match="cl100k_base-based estimate"),
        ):
            count = tokens_module.count_tokens_anthropic("hello world")

        fallback.assert_called_once_with("hello world")
        assert count == 11
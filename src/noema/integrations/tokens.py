"""Provider-specific token counting helpers (Phase 6).

`noema.compress()` and `KnowledgeBase.retrieve(max_tokens=...)` both budget
tokens using tiktoken's cl100k_base encoding by default -- a reasonable
approximation for most LLMs, and exactly right for OpenAI's GPT-3.5/4-era
models. These helpers let you count (and, by passing `token_model` through
to the rest of the library, budget) against a *specific* provider's own
notion of a token instead of that default:

    import noema
    from noema.integrations.tokens import count_tokens_openai, count_tokens_anthropic

    count_tokens_openai(text, model="gpt-4o")
    count_tokens_anthropic(text, model="claude-sonnet-5")

Both are plain functions with no required third-party dependency beyond
what's already in the base install (`tiktoken`) -- `count_tokens_anthropic`
only *optionally* uses the `anthropic` package (the `noema[anthropic]`
extra) for an exact count; without it, it falls back to the same
cl100k_base estimate `noema.count_tokens` uses, with a warning that it's an
approximation.
"""

from __future__ import annotations

import warnings

from noema.compress import count_tokens as _count_tokens_cl100k


def count_tokens_openai(text: str, model: str = "gpt-4o") -> int:
    """Count tokens the way a given OpenAI model would, via tiktoken.

    Falls back to the cl100k_base encoding for any model name tiktoken
    doesn't recognize (e.g. a very new model it hasn't been updated for
    yet), which is the same encoding `noema.count_tokens` uses by default.
    """
    if not text or not text.strip():
        return 0

    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_tokens_anthropic(
    text: str,
    model: str = "claude-sonnet-5",
    api_key: str | None = None,
) -> int:
    """Count tokens the way a given Anthropic (Claude) model would.

    Claude's real tokenizer isn't published, so an exact count requires
    calling Anthropic's `count_tokens` API (via the `anthropic` package,
    the `noema[anthropic]` extra, and a valid API key -- passed explicitly
    or read from the `ANTHROPIC_API_KEY` environment variable by the SDK).

    Without the package installed, or if the API call fails for any
    reason (no key, network, rate limit), this falls back to noema's own
    cl100k_base-based estimate and emits a `UserWarning` saying so -- it
    keeps working, but the number is an approximation, not an exact count.
    """
    if not text or not text.strip():
        return 0

    try:
        import anthropic
    except ImportError:
        warnings.warn(
            "The `anthropic` package isn't installed, so count_tokens_anthropic() "
            "is returning a cl100k_base-based estimate rather than an exact "
            "Claude token count. Install it with `pip install noema[anthropic]` "
            "for exact counts.",
            stacklevel=2,
        )
        return _count_tokens_cl100k(text)

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        response = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
    except Exception as exc:  # noqa: BLE001 - any SDK/auth/network failure
        warnings.warn(
            f"count_tokens_anthropic() couldn't get an exact count from the "
            f"Anthropic API ({exc}); returning a cl100k_base-based estimate "
            "instead.",
            stacklevel=2,
        )
        return _count_tokens_cl100k(text)
"""Shared fixtures for the Phase 1 test suite.

The spaCy model (en_core_web_sm) is loaded once per test session via the
package's own cached loader rather than re-loaded per test -- loading is
the slow part (~100ms+) and Phase 1's parser tests run it many times.
"""

from __future__ import annotations

import pytest

from noema.parser.entities import _load_model as _load_spacy_model
from noema.schema import Document


@pytest.fixture(scope="session", autouse=True)
def _warm_spacy_model():
    """Load the spaCy model once at the start of the test session so the
    first parser test doesn't eat a one-time load cost in its own timing,
    and so a missing model fails fast with a clear pytest error instead of
    surfacing inside an unrelated test.
    """
    _load_spacy_model()


@pytest.fixture
def make_document():
    """Factory fixture: make_document("some text") -> Document."""

    def _make(text: str) -> Document:
        return Document(text=text)

    return _make
"""Live retrieval test against the real, already-populated ChromaDB.

Marked `live` so it is deselected in CI; run locally when you have indexed docs:

    pytest -m live
    # or
    RUN_LIVE_TESTS=1 pytest tests/test_retrieval_live.py

Uses .env (PERSISTENT_STORAGE, COLLECTION_NAME, COLLECTION_PATH).
"""

import os

import pytest

from chroma import RagClient

pytestmark = pytest.mark.live

RUN_LIVE = os.getenv("RUN_LIVE_TESTS") or os.getenv("TEST_USE_REAL_DB")
if not RUN_LIVE:
    pytest.skip(
        "set RUN_LIVE_TESTS=1 to run against the real ChromaDB", allow_module_level=True
    )


@pytest.fixture(scope="module")
def client():
    return RagClient(
        name=os.getenv("COLLECTION_NAME", "my-collection"),
        persistent_storage=os.getenv("PERSISTENT_STORAGE", "chroma_db"),
        collection_path=os.getenv("COLLECTION_PATH", "source_docs"),
    )


def test_context_non_empty(client):
    context = client.get_context("What are the main guidelines?", n_results=5)
    assert isinstance(context, str)
    assert context.strip()


def test_rule_query_includes_rule(client):
    results = client.retriever.get_query_results("What is PRE30-C?", n_results=5)
    assert results
    assert any("PRE30-C" in r["content"] for r in results)
    assert "PRE30-C" in results[0]["content"]


def test_semantic_query_relevant(client):
    context = client.get_context("security and macros", n_results=5).lower()
    assert context
    assert any(term in context for term in ("security", "macro", "rule", "code"))

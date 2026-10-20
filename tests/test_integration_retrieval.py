"""Integration test: chunk -> index -> retrieve, end to end, in-memory.

This is the test that would have caught the original retrieval problem: with
rule-aware chunking, a query naming a rule id must return a chunk that actually
contains that rule's definition and examples - not a fragment.
"""

import threading

import chromadb
import pytest

from chroma.hash_manager import FileHashManager
from chroma.indexer import ChromaIndexer
from chroma.retriever import ChromaRetriever
from chroma.text_splitter import TextSplitter

CORPUS = """# Mini Standard

## **1 Introduction**

Introductory prose.

## **2.1 PRE30-C. Do not create a universal character name through concatenation**

### Noncompliant Code Example

Concatenating a universal character name is noncompliant.

### Compliant Solution

Use the character directly instead.

## **2.2 PRE31-C. Avoid side effects in arguments to unsafe macros**

### Noncompliant Code Example

Passing an increment into an unsafe macro is noncompliant.

### Compliant Solution

Evaluate the argument first.

## **3.1 DCL30-C. Declare objects with appropriate storage durations**

### Noncompliant Code Example

Returning a pointer to a local object is noncompliant.

### Compliant Solution

Use static or caller-provided storage.
"""


@pytest.fixture
def rag(tmp_path):
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("integration")
    hash_manager = FileHashManager(tmp_path / "hashes.json")
    splitter = TextSplitter(chunk_size=2000, chunk_overlap=200)
    indexer = ChromaIndexer(collection, threading.Lock(), splitter, hash_manager)

    doc = tmp_path / "standard.md"
    doc.write_text(CORPUS, encoding="utf-8")
    indexer.index_files([str(doc)])

    return collection, ChromaRetriever(collection, max_distance=None)


class TestEndToEndRuleRetrieval:
    @pytest.mark.parametrize("rule_id", ["PRE30-C", "PRE31-C", "DCL30-C"])
    def test_rule_query_returns_that_rule_first(self, rag, rule_id):
        _, retriever = rag
        results = retriever.get_query_results(f"What is {rule_id}?", n_results=5)
        assert results, f"no results for {rule_id}"
        assert rule_id in results[0]["content"], f"{rule_id} should be the top result"
        assert results[0]["distance"] == 0.0

    def test_rule_chunk_contains_its_examples(self, rag):
        _, retriever = rag
        results = retriever.get_query_results("What is PRE30-C?", n_results=1)
        chunk = results[0]["content"]
        assert "noncompliant" in chunk.lower()
        assert "compliant" in chunk.lower()

    def test_semantic_query_still_returns_context(self, rag):
        _, retriever = rag
        results = retriever.get_query_results(
            "storage duration of local objects", n_results=3
        )
        assert results

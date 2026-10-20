"""Shared pytest fixtures.

Retriever tests run against a lightweight in-memory fake collection so they need no
real ChromaDB, no embedding model and no network access - which is what lets them run
in CI. Integration tests use a real ephemeral ChromaDB with a tiny fixture corpus.
"""

import pytest

from chroma.text_splitter import TextSplitter


# --------------------------------------------------------------- fixture corpora

RULE_DOC = """# Sample Standard

## **1 Introduction**

General text that is not a rule and should be split normally.

## **2.1 PRE30-C. Do not create a universal character name through concatenation**

### Noncompliant Code Example

```c
/* noncompliant */
```

### Compliant Solution

```c
/* compliant */
```

## **2.2 PRE31-C. Avoid side effects in arguments to unsafe macros**

### Noncompliant Code Example

```c
/* noncompliant PRE31 */
```

### Compliant Solution

```c
/* compliant PRE31 */
```

## Appendix A: Bibliography

## [Acton 2006]

A citation entry that belongs to the appendix, not to a rule.
"""


@pytest.fixture
def rule_doc(tmp_path):
    """Write RULE_DOC to a temp .md file and return its path."""
    path = tmp_path / "standard.md"
    path.write_text(RULE_DOC, encoding="utf-8")
    return str(path)


@pytest.fixture
def splitter():
    """TextSplitter with a small chunk_size so oversized rules exercise sub-splitting."""
    return TextSplitter(chunk_size=200, chunk_overlap=20)


class FakeCollection:
    """Minimal stand-in for a chromadb Collection: only query/get used by the retriever.

    Documents are a list of dicts: {id, document, metadata, distance_to_query}.
    `query` returns results ordered by an explicit per-doc rank so tests are
    deterministic; `get` supports the `where={"rule_id": ...}` lookup.
    """

    def __init__(self, docs):
        self.docs = docs

    def get(self, where=None, include=None):
        matched = self.docs
        if where and "rule_id" in where:
            matched = [d for d in self.docs if d["metadata"].get("rule_id") == where["rule_id"]]
        return {
            "ids": [d["id"] for d in matched],
            "documents": [d["document"] for d in matched],
            "metadatas": [d["metadata"] for d in matched],
        }

    def query(self, query_texts=None, n_results=10, include=None):
        ranked = self.docs[:n_results]
        return {
            "ids": [[d["id"] for d in ranked]],
            "documents": [[d["document"] for d in ranked]],
            "metadatas": [[d["metadata"] for d in ranked]],
            "distances": [[d["distance_to_query"] for d in ranked]],
        }


@pytest.fixture
def fake_collection():
    """A fake collection where a rule chunk and a far-away noise chunk coexist.

    PRE30-C is registered as a rule (distance 0.9 -> dropped by default threshold).
    A 'far' unrelated chunk has distance 1.8 -> must be filtered out.
    """
    return FakeCollection(
        [
            {
                "id": "rule_pre30",
                "document": "## **2.1 PRE30-C. Do not create a universal character name**",
                "metadata": {"source": "standard.md", "rule_id": "PRE30-C", "chunk_index": 1},
                "distance_to_query": 0.9,
            },
            {
                "id": "noise_far",
                "document": "An unrelated chunk about something completely different.",
                "metadata": {"source": "other.md", "chunk_index": 0},
                "distance_to_query": 1.8,
            },
        ]
    )

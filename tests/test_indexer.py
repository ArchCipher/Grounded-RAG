"""Tests for the Chroma indexer: chunk ids, rule_id metadata, resilience, re-index.

Uses an in-memory ChromaDB (EphemeralClient) so no persistent state or API key is
required - runs in CI.
"""

import threading

import chromadb
import pytest

from chroma.hash_manager import FileHashManager
from chroma.indexer import ChromaIndexer
from chroma.text_splitter import TextSplitter


@pytest.fixture
def indexer(tmp_path, request):
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(request.node.name)
    hash_manager = FileHashManager(tmp_path / "file_hashes.json")
    splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
    return ChromaIndexer(collection, threading.Lock(), splitter, hash_manager)


class TestChunkId:
    def test_deterministic(self):
        assert ChromaIndexer._generate_md5_hash("t", "/f.md") == ChromaIndexer._generate_md5_hash("t", "/f.md")

    def test_content_and_source_matter(self):
        a = ChromaIndexer._generate_md5_hash("A", "/f.md")
        assert a != ChromaIndexer._generate_md5_hash("B", "/f.md")
        assert a != ChromaIndexer._generate_md5_hash("A", "/g.md")


class TestRuleIdMetadata:
    def test_rule_id_extracted_from_header(self, indexer, tmp_path):
        path = tmp_path / "std.md"
        path.write_text(
            "## **2.1 PRE30-C. Do not create a universal character name**\n\nbody text\n",
            encoding="utf-8",
        )
        indexer.index_files([str(path)])
        got = indexer.collection.get(where={"rule_id": "PRE30-C"})
        assert got["ids"], "expected a chunk tagged with rule_id PRE30-C"

    def test_non_rule_chunk_has_no_rule_id(self, indexer, tmp_path):
        path = tmp_path / "plain.md"
        path.write_text("## Introduction\n\nNo rules here.\n", encoding="utf-8")
        indexer.index_files([str(path)])
        got = indexer.collection.get(include=["metadatas"])
        assert all("rule_id" not in m for m in got["metadatas"])


class TestResilience:
    def test_one_bad_file_does_not_abort_the_rest(self, indexer, tmp_path):
        """A file that fails to split must not stop later files being indexed."""
        good1 = tmp_path / "good1.md"
        good1.write_text("## A\n\ntext\n", encoding="utf-8")
        bad = tmp_path / "bad.txt"  # not .md -> TextSplitter raises
        bad.write_text("not markdown\n", encoding="utf-8")
        good2 = tmp_path / "good2.md"
        good2.write_text("## B\n\ntext\n", encoding="utf-8")

        result = indexer.index_files([str(good1), str(bad), str(good2)])

        assert result.errors, "the bad file should be reported in errors"
        indexed_names = {str(p).split("/")[-1] for p in result.files}
        assert "good2.md" in indexed_names, "good2 must still be indexed after bad.txt"


class TestReIndex:
    def test_unchanged_file_is_skipped(self, indexer, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("## A\n\ntext\n", encoding="utf-8")
        first = indexer.index_files([str(path)])
        second = indexer.index_files([str(path)])
        assert len(first.files) == 1
        assert second.files == [], "unchanged file should be skipped on re-index"

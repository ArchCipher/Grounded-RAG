"""Tests for ChromaDB retrieval: rule boost, dedup, distance filtering, result cap.

Run against a fake in-memory collection (see conftest) so no database or API key is
needed - this is what makes them CI-friendly.
"""

import pytest

from chroma.retriever import ChromaRetriever, normalize_rule_id


def _ids(results):
    return [r["metadata"].get("rule_id") or r["content"][:20] for r in results]


class TestRuleBoost:
    def test_rule_query_prepends_rule_chunk_at_distance_zero(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        results = retriever.get_query_results("What is PRE30-C?", n_results=5)
        assert results, "expected at least the boosted rule chunk"
        assert "PRE30-C" in results[0]["content"]
        assert results[0]["distance"] == 0.0

    def test_non_rule_query_has_no_boost(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        results = retriever.get_query_results("tell me about macros", n_results=5)
        assert all(
            "PRE30-C" not in r["content"] or r["distance"] != 0.0 for r in results
        )


class TestDistanceFiltering:
    def test_far_chunk_is_dropped(self, fake_collection):
        """Noise chunk at distance 1.8 must be filtered by the default threshold."""
        retriever = ChromaRetriever(fake_collection, max_distance=1.0)
        results = retriever.get_query_results("anything", n_results=5)
        assert all(r["distance"] <= 1.0 for r in results)
        assert all("completely different" not in r["content"] for r in results)

    def test_threshold_none_keeps_everything(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        results = retriever.get_query_results("anything", n_results=5)
        assert any(r["distance"] == 1.8 for r in results)

    def test_rule_boost_survives_threshold(self, fake_collection):
        """Boosted rule chunks (distance 0.0) are always kept, even with a strict threshold."""
        retriever = ChromaRetriever(fake_collection, max_distance=0.1)
        results = retriever.get_query_results("What is PRE30-C?", n_results=5)
        assert any("PRE30-C" in r["content"] and r["distance"] == 0.0 for r in results)


class TestResultCapAndDedup:
    def test_result_count_capped(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        results = retriever.get_query_results("anything", n_results=1)
        assert len(results) == 1

    def test_no_duplicate_ids(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        results = retriever.get_query_results("What is PRE30-C?", n_results=10)
        contents = [r["content"] for r in results]
        assert len(contents) == len(set(contents))


class TestRuleIdNormalization:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("explain PRE31-C", "PRE31-C"),
            ("explain pre31c", "PRE31-C"),
            ("what is PRE 31 C?", "PRE31-C"),
            ("pre31-cpp", "PRE31-CPP"),
            ("DCL30-CPP", "DCL30-CPP"),
            ("tell me dcl30cpp", "DCL30-CPP"),
        ],
    )
    def test_normalizes_variants(self, text, expected):
        assert normalize_rule_id(text) == expected

    def test_none_when_no_rule(self):
        assert normalize_rule_id("what are the safe coding rules?") is None

    def test_boost_fires_for_unhyphenated_query(self, fake_collection):
        """The real bug: typing 'pre30c' must still boost the PRE30-C chunk."""
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        results = retriever.get_query_results("explain pre30c", n_results=5)
        assert results
        assert results[0]["distance"] == 0.0
        assert "PRE30-C" in results[0]["content"]


class TestMissingRuleNotice:
    def test_no_notice_when_rule_present(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        assert retriever.missing_rule_notice("explain PRE30-C") == ""

    def test_no_notice_without_a_rule_id(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        assert retriever.missing_rule_notice("what are the safe coding rules?") == ""


class TestContextFormatting:
    def test_context_labels_sources(self, fake_collection):
        retriever = ChromaRetriever(fake_collection, max_distance=None)
        results = retriever.get_query_results("What is PRE30-C?", n_results=2)
        context = retriever.get_context(results)
        assert "[source 1:" in context

    def test_empty_results_give_empty_context(self):
        retriever = ChromaRetriever(None, max_distance=None)  # type: ignore[arg-type]
        assert retriever.get_context([]) == ""

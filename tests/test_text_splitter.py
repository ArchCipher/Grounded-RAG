"""Tests for rule-aware chunking.

These are the tests that justify the whole retrieval rework: they prove a rule's
header and its examples stay together, and that oversized rules are sub-split with
the header repeated so every part stays retrievable.
"""

from chroma.text_splitter import RULE_HEADER_PATTERN, TextSplitter


def _rule_chunks(chunks):
    return [c for c in chunks if RULE_HEADER_PATTERN.match(c)]


class TestRuleAwareSplitting:
    def test_rule_header_scoped_to_one_chunk(self, rule_doc, splitter):
        """PRE30-C header, its noncompliant example and compliant solution share a chunk."""
        chunks = splitter.split(rule_doc)
        pre30 = [c for c in chunks if "PRE30-C" in c and RULE_HEADER_PATTERN.match(c)]
        assert pre30, "expected a chunk beginning with the PRE30-C rule header"
        chunk = pre30[0]
        assert "noncompliant" in chunk.lower()
        assert "compliant" in chunk.lower()

    def test_each_rule_yields_a_chunk_starting_with_its_header(
        self, rule_doc, splitter
    ):
        chunks = splitter.split(rule_doc)
        headers = [c.splitlines()[0] for c in chunks if RULE_HEADER_PATTERN.match(c)]
        assert any("PRE30-C" in h for h in headers)
        assert any("PRE31-C" in h for h in headers)

    def test_appendix_not_absorbed_into_last_rule(self, rule_doc, splitter):
        """The bibliography appendix must not end up inside a rule chunk."""
        chunks = splitter.split(rule_doc)
        wrong = [
            c
            for c in _rule_chunks(chunks)
            if "Acton 2006" in c and c.splitlines()[0].strip().startswith("## **")
        ]
        assert not wrong, "appendix text leaked into a rule chunk"

    def test_falls_back_when_no_rules(self, tmp_path):
        path = tmp_path / "plain.md"
        path.write_text(
            "## Intro\n\nSome text.\n\n## Detail\n\nMore text.\n", encoding="utf-8"
        )
        chunks = TextSplitter(chunk_size=50, chunk_overlap=5).split(str(path))
        assert chunks
        assert not any(RULE_HEADER_PATTERN.match(c) for c in chunks)


class TestOversizedRuleSubSplit:
    def test_header_repeated_on_every_part(self, tmp_path):
        """A rule larger than chunk_size splits into parts that each restate the header."""
        big = "## **9.1 BIG30-C. A very large rule**\n\n" + ("word " * 400)
        path = tmp_path / "big.md"
        path.write_text(big, encoding="utf-8")
        chunks = TextSplitter(chunk_size=200, chunk_overlap=20).split(str(path))
        parts = [c for c in chunks if "BIG30-C" in c]
        assert len(parts) > 1, "expected the oversized rule to be split into parts"
        for part in parts:
            assert "BIG30-C" in part, "every part must still reference the rule id"
        assert any("(part " in p for p in parts), "sub-parts should be labelled"

    def test_non_rule_section_still_splits(self, splitter):
        """A long plain section (no rule header) is still size-split."""
        text = "## Intro\n\n" + ("word " * 500)
        chunks = splitter.split_text(text)
        assert len(chunks) > 1


class TestHeaderVariants:
    def test_unbolded_and_bolded_headers_both_match(self):
        for header in (
            "## **2.1 PRE30-C. text**",
            "## 2.1 PRE30-C. text",
            "## **3.2 DCL30-CPP. text**",
        ):
            assert RULE_HEADER_PATTERN.match(header), header

    def test_non_rule_headers_do_not_match(self):
        for header in ("## **1.1 Scope**", "## Introduction", "## **False Positives**"):
            assert not RULE_HEADER_PATTERN.match(header), header

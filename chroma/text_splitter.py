"""Chunking: split on rule headers first, then by size with RecursiveCharacterTextSplitter.

Rule corpora (e.g. SEI CERT C/C++) must keep each rule - its heading, noncompliant
examples and compliant solution - inside a single chunk, otherwise semantic search
cannot surface the whole rule. This splitter:

1. Finds every rule-header boundary (``## **N.M RULE-ID-CPP. ...``).
2. Slices the document at those boundaries: preamble, one slice per rule, tail.
3. Keeps each rule whole when it fits in ``chunk_size``; when a rule is larger it is
   sub-split by ``RecursiveCharacterTextSplitter`` and the rule header is repeated at
   the top of every part so each part stays self-describing and retrievable.
4. Falls back to plain size-splitting for documents that contain no rule headers.

The public ``split(file)`` signature is unchanged: it returns ``list[str]``.
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Rule-header pattern, e.g. "## **2.1 PRE30-C. Do not create ..." (CERT C/C++).
# Kept permissive: optional bold, any leading numbering, C or CPP suffix.
RULE_HEADER_PATTERN = re.compile(
    r"^##[ \t]+\*{0,2}\d+(?:\.\d+)*[ \t]+([A-Z]{3,}\d+-C(?:PP)?)\.[^\n]*",
    re.MULTILINE,
)

# Sections after the rules (APPENDIX, BIBLIOGRAPHY, GLOSSARY). The last rule must not
# swallow these - otherwise a rule chunk absorbs the whole bibliography. Matched on
# their own line so inline mentions ("see Appendix A") are not treated as boundaries.
TERMINAL_SECTION_PATTERN = re.compile(
    r"^##[ \t]+\*{0,2}(?:Appendix|Bibliography|Glossary)\b[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)


class TextSplitter:
    """Split markdown into rule-aligned chunks, sub-splitting oversized rules.

    ``split`` returns a list of chunk strings. Rule chunks begin with their rule
    header; oversized rules are split into parts that each repeat the header.
    """

    DEFAULT_CHUNK_SIZE = 2000
    DEFAULT_CHUNK_OVERLAP = 200

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, file: str) -> list[str]:
        """Read a .md file and return its chunks.

        Rule-bearing documents are sliced at rule boundaries; documents without rule
        headers fall back to chapter-header splitting plus recursive size splitting.
        """
        if not file.endswith(".md"):
            raise ValueError(f"File {file} is not a markdown file")
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        return self.split_text(text)

    def split_text(self, text: str) -> list[str]:
        """Split raw markdown text into chunks (rule-aware when rules are present)."""
        boundaries = list(RULE_HEADER_PATTERN.finditer(text))
        if not boundaries:
            return self._split_without_rules(text)

        # Rule slicing must stop where the trailing sections (appendix, bibliography)
        # begin, so the final rule does not absorb them.
        terminal = TERMINAL_SECTION_PATTERN.search(text)
        rule_zone_end = terminal.start() if terminal else len(text)

        chunks: list[str] = []
        # Text before the first rule header (title, table of contents, intro).
        preamble = text[: boundaries[0].start()].strip()
        if preamble:
            chunks.extend(
                self._split_section(preamble, header=None)
            )
        # One slice per rule, from its header up to the next rule header (or the
        # start of the trailing sections / end of document).
        for i, match in enumerate(boundaries):
            start = match.start()
            if start >= rule_zone_end:
                break
            next_start = (
                boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
            )
            end = min(next_start, rule_zone_end)
            rule_text = text[start:end].strip()
            if rule_text:
                chunks.extend(
                    self._split_section(rule_text, header=match.group(0).strip())
                )
        # Trailing appendix/bibliography/glossary, split by size as normal text.
        if terminal:
            tail = text[terminal.start() :].strip()
            if tail:
                chunks.extend(self._split_section(tail, header=None))
        return chunks

    def _split_without_rules(self, text: str) -> list[str]:
        """Fallback: split on chapter headers then by size."""
        sections = self._split_on_headers(text)
        if not sections:
            sections = [text]
        chunks: list[str] = []
        for section in sections:
            chunks.extend(self.text_splitter.split_text(section))
        return chunks

    def _split_section(self, section: str, header: str | None) -> list[str]:
        """Return section as one chunk, or split-and-reheader if it exceeds chunk_size."""
        if len(section) <= self.chunk_size:
            return [section]
        parts = self.text_splitter.split_text(section)
        if header is None:
            return parts
        # Repeat the rule header on every part so each is self-describing.
        total = len(parts)
        reheadered = []
        for i, part in enumerate(parts, 1):
            if part.lstrip().startswith(header):
                reheadered.append(part)
            else:
                reheadered.append(f"{header} (part {i}/{total})\n\n{part}")
        return reheadered

    @staticmethod
    def _split_on_headers(text: str) -> list[str]:
        """Split on chapter-level headers into sections"""
        header_pattern = r"^## "
        sections = []
        current_section = []
        lines = text.split("\n")
        for line in lines:
            match = re.match(header_pattern, line)
            if not match:
                current_section.append(line)
                continue
            # if header matches
            if current_section:  # add current section to sections
                sections.append("\n".join(current_section))
            # start new section
            current_section = [line]
        if current_section:  # add last section
            sections.append("\n".join(current_section))
        return sections

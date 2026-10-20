"""ChromaDB retrieval: semantic search plus rule-id boost for CERT-style queries."""

import re
from typing import TypedDict

from chromadb import Collection, Metadata, QueryResult

# Default cap on semantic distance. Chroma's default L2 distance grows with
# dissimilarity; chunks above this are near-random and only dilute the prompt.
# Rule-boost chunks are injected with distance 0.0 and always bypass the filter.
DEFAULT_MAX_DISTANCE = 1.0

# Matches a CERT-style rule id loosely so natural typing still hits the rule boost:
# "PRE31-C", "pre31c", "PRE 31 C", "pre31-cpp", "DCL30-CPP".
_RULE_ID_LOOSE = re.compile(r"([A-Z]{3,})\s*(\d+)\s*-?\s*(C(?:PP)?)\b", re.IGNORECASE)


def normalize_rule_id(text: str) -> str | None:
    """Return canonical CERT rule id (e.g. 'PRE31-C') found in text, or None.

    Inserts the hyphen users often omit, so 'pre31c' and 'pre 31 c' both match
    the 'PRE31-C' metadata written at index time.
    """
    match = _RULE_ID_LOOSE.search(text)
    if not match:
        return None
    letters, digits, suffix = match.groups()
    return f"{letters.upper()}{digits}-{suffix.upper()}"


class RetrievalResult(TypedDict):
    content: str
    metadata: Metadata
    distance: float | None


class ChromaRetriever:
    """Retrieves chunks by semantic similarity, prepends rule chunk when message matches rule id."""

    def __init__(
        self, collection: Collection, max_distance: float | None = DEFAULT_MAX_DISTANCE
    ) -> None:
        self.collection = collection
        self.max_distance = max_distance

    def get_context(self, results: list[RetrievalResult]) -> str:
        """Format list of {content, metadata} into a single context string with source labels."""
        if not results:
            return ""
        context_chunks = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            content = result["content"]
            context_chunks.append(f"[source {i}: {source}]\n{content}")
        return "\n\n".join(context_chunks)

    def missing_rule_notice(self, message: str) -> str:
        """If the message names a rule id that is not in the collection, return a notice.

        Gives the model an explicit signal to answer "not in sources" instead of
        guessing from general knowledge when a requested rule is not indexed.
        """
        rule_id = normalize_rule_id(message)
        if not rule_id:
            return ""
        found = self.collection.get(where={"rule_id": rule_id}, include=[])
        if found.get("ids"):
            return ""
        return (
            f"NOTE: The requested rule {rule_id} is NOT present in the source "
            f"documents. Do not answer from general knowledge; state that it is "
            f"not in the sources."
        )

    def get_query_results(self, message: str, n_results: int) -> list[RetrievalResult]:
        """
        Rule-id match first (if any), then semantic search.
        Drop semantic chunks above max_distance, dedupe, return up to n_results.
        """
        retrieved = []
        seen_ids = set()
        self._get_rule_results(message, seen_ids, retrieved)
        results = self.collection.query(
            query_texts=[message],
            n_results=n_results * 2,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents")
        if not documents or not documents[0]:
            return retrieved
        ids = results["ids"][0] if results.get("ids") else []
        for i, doc in enumerate(documents[0]):
            if len(retrieved) >= n_results:
                break
            distance = self._get_distance(results, i)
            if self._is_too_far(distance):
                continue
            if ids and i < len(ids):
                doc_id = ids[i]
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
            retrieved.append(
                {
                    "content": doc,
                    "metadata": self._get_metadata(results, i),
                    "distance": distance,
                }
            )
        return retrieved[:n_results]

    def _is_too_far(self, distance: float | None) -> bool:
        """True if distance exceeds max_distance. None threshold or distance disables filter."""
        if self.max_distance is None or distance is None:
            return False
        return distance > self.max_distance

    def _get_rule_results(
        self, message: str, seen_ids: set[str], retrieved: list[RetrievalResult]
    ) -> None:
        """If message contains a CERT-style rule id, prepend that chunk and mark ids seen."""
        rule_id = normalize_rule_id(message)
        if not rule_id:
            return
        rule_results = self.collection.get(
            where={"rule_id": rule_id}, include=["documents", "metadatas"]
        )
        if not rule_results["ids"]:
            return
        for idx, doc_id in enumerate(rule_results["ids"]):
            seen_ids.add(doc_id)
            documents = rule_results.get("documents")
            if not documents or idx >= len(documents) or not documents[idx]:
                continue
            doc = documents[idx]
            meta = rule_results["metadatas"][idx] if rule_results["metadatas"] else {}
            retrieved.append({"content": doc, "metadata": meta, "distance": 0.0})

    @staticmethod
    def _get_metadata(results: QueryResult, i: int) -> Metadata:
        """Get metadata for i-th document in Chroma query result."""
        metadatas = results.get("metadatas")
        if not metadatas or not metadatas[0]:
            return {}
        if i >= len(metadatas[0]):
            return {}
        return metadatas[0][i]

    @staticmethod
    def _get_distance(results: QueryResult, i: int) -> float | None:
        """Get distance for i-th document in Chroma query result."""
        distances = results.get("distances")
        if not distances or not distances[0]:
            return None
        if i >= len(distances[0]):
            return None
        return distances[0][i]

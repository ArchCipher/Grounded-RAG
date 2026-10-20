#!/usr/bin/env python3
"""MCP server exposing Grounded RAG tools.

Run with:
    uv run python mcp_server.py

Or add to your MCP client config (e.g. Cursor, Claude):
{
  "mcpServers": {
    "grounded-rag": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/grounded-rag"
    }
  }
}

Tools:
- search: retrieve relevant chunks from the indexed corpus
- ask: get a grounded answer constrained to the indexed sources
- get_rule: fetch a specific rule by ID (e.g. PRE30-C) directly from the index
"""

import os

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from chroma import RagClient

load_dotenv()

mcp = MCPServer("grounded-rag")

rag_client = RagClient(
    name=os.getenv("COLLECTION_NAME", "my-collection"),
    persistent_storage=os.getenv("PERSISTENT_STORAGE", "./chroma_db"),
    collection_path=os.getenv("COLLECTION_PATH", "source_docs"),
    hash_filename=os.getenv("HASH_FILE", "file_hashes.json"),
)


@mcp.tool()
def search(query: str, n_results: int = 5) -> list[dict]:
    """Search the indexed documents and return relevant chunks with sources."""
    results = rag_client.retriever.get_query_results(query, n_results)
    return [
        {
            "content": r["content"],
            "source": r["metadata"].get("source", "unknown"),
            "rule_id": r["metadata"].get("rule_id"),
            "distance": r["distance"],
        }
        for r in results
    ]


@mcp.tool()
def ask(question: str, n_results: int = 5) -> str:
    """Ask a question and get an answer grounded strictly in the indexed sources."""
    return rag_client.get_context(question, n_results)


@mcp.tool()
def get_rule(rule_id: str) -> dict:
    """Fetch a rule by ID (e.g. PRE30-C) directly from the index if present."""
    rule_id = rule_id.strip().upper()
    results = rag_client.retriever.collection.get(
        where={"rule_id": rule_id},
        include=["documents", "metadatas"],
    )
    ids = results.get("ids") or []
    if not ids:
        return {
            "found": False,
            "rule_id": rule_id,
            "message": "Rule not found in indexed sources.",
        }
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    return {
        "found": True,
        "rule_id": rule_id,
        "chunks": [
            {
                "content": doc,
                "source": (meta or {}).get("source", "unknown"),
            }
            for doc, meta in zip(documents, metadatas)
        ],
    }


if __name__ == "__main__":
    mcp.run()

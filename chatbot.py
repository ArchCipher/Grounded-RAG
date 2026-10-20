"""
FastAPI client for RAG chatbot

- FastAPI: Python web framework to build APIs
- Uvicorn: ASGI server used to run the FastAPI app
- Pydantic: data validation library for Python
- dotenv/ os: load environment variables from .env file
- google-genai: Google GenAI API client
- chroma: ChromaDB/ RAG client implementation
"""

import asyncio
import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel

from chroma import RagClient

# Logging
logger = logging.getLogger("chatbot")


# Pydantic request and response models
class ChatRequest(BaseModel):
    """Incoming chat message and session identifier."""

    session_id: int
    message: str


class ChatResponse(BaseModel):
    """LLM reply returned from /chat."""

    reply: str


# Constants
SYSTEM_PROMPT = """You are a documentation assistant. Answer ONLY using the context from
the source documents provided below.

Rules:
- Base every statement on the provided context. Do not use outside or general knowledge.
- Only describe an item if its full definition/content is present in the context.
  A passing mention or cross-reference is NOT the item itself — do not describe,
  summarize, or recommend something you only see cited.
- If the context does not contain the answer, reply exactly: "I don't have that
  information in the source documents." Do not guess or approximate.
- When you reference something from the context, cite it exactly as written.
- Be concise."""

# Load environment variables
load_dotenv()

# get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")
# instantiate LLM client: Gemini
genai_client = genai.Client(api_key=api_key)

# instantiate RAG client: ChromaDB
rag_client = RagClient(
    name=os.getenv("COLLECTION_NAME", "my-collection"),
    persistent_storage=os.getenv("PERSISTENT_STORAGE", "./chroma_db"),
    collection_path=os.getenv("COLLECTION_PATH", "source_docs"),
    hash_filename=os.getenv("HASH_FILE", "file_hashes.json"),
)

# instantiate FastAPI app
app = FastAPI()


@app.get("/")
def root() -> dict[str, str]:
    """Health check."""
    return {"message": "Hello World"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Retrieve RAG context and generate reply."""
    reply = generate_response(req.message, rag_client.get_context(req.message))
    return ChatResponse(reply=reply)


def generate_response(message: str, context: str) -> str:
    """Build prompt and call LLM client. Returns reply text or raise HTTPException."""
    if context:
        prompt = f"""Please answer the question based on the context below when relevant:
Context from source documents: {context}
Question: {message}
Answer: """
    else:
        prompt = message
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents={"text": prompt},
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0,  # how random the response is
                "top_p": 0.95,  # probability of selecting the next token
                "top_k": 20,  # number of tokens to consider for the next token
            },
        )
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    text = response.text
    if text is None:
        raise HTTPException(
            status_code=500, detail="LLM returned no text. Please try again."
        )
    return text


async def main() -> None:
    """Configure logging, and run Uvicorn."""
    # Configure logging to file or stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(name)s: %(message)s: %(asctime)s",
    )
    log_file = os.getenv("LOG_FILE")
    if log_file:
        logging.basicConfig(filename=log_file)
    else:
        logging.basicConfig(stream=sys.stdout)
    # Run Uvicorn programmatically
    config = uvicorn.Config(
        "chatbot:app", port=8000
    )  # use reload=True if not production
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

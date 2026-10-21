import logging
import os

from dotenv import load_dotenv

from chroma import RagClient

logger = logging.getLogger("reload_db")


# Load environment variables
load_dotenv()

# instantiate RagClient
rag_client = RagClient(
    name=os.getenv("COLLECTION_NAME", "my-collection"),
    persistent_storage=os.getenv("PERSISTENT_STORAGE", "./chroma_db"),
    collection_path=os.getenv("COLLECTION_PATH", "source_docs"),
    hash_filename=os.getenv("HASH_FILE", "file_hashes.json"),
)


def main() -> None:
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s][%(message)s]",
    )
    # Reload documents
    response = rag_client.reload_collection()
    log = f"Collection reloaded: {len(response.files)} Files indexed: {response.files}, Errors: {response.errors}"
    if response.errors:
        logger.error(log)
    else:
        logger.info(log)


if __name__ == "__main__":
    main()

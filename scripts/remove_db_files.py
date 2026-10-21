import logging
import os

from dotenv import load_dotenv

from chroma import RagClient

logger = logging.getLogger("remove_db_files")


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
    print("Enter file names (one per line):")
    print("If file is within COLLECTION_PATH/file_name, enter only the file_name.")
    print("If file is within COLLECTION_PATH/subdir/file_name, enter subdir/file_name.")
    lines = []
    while True:
        # Get user input
        line = input().strip()
        if not line:
            break
        lines.append(line)
    # Convert to full paths
    base_folder = os.getenv("COLLECTION_PATH", "source_docs")
    files = [os.path.join(base_folder, line) for line in lines]
    # Remove file from collection
    files_removed = rag_client.indexer.remove_files(files=files)
    logger.info(f"Files removed: {files_removed}")


if __name__ == "__main__":
    main()

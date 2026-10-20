"""ChromaDB indexing: chunk documents, add/update/remove in collection, track file mtimes."""

import hashlib
import logging
from pathlib import Path
from threading import Lock

from chromadb import Collection

from chroma.hash_manager import FileHashManager
from chroma.models import CollectionResult
from chroma.text_splitter import RULE_HEADER_PATTERN, TextSplitter

logger = logging.getLogger("ChromaIndexer")


class ChromaIndexer:
    """Indexes markdown files into a ChromaDB collection"""

    def __init__(
        self,
        collection: Collection,
        lock: Lock,
        text_splitter: TextSplitter,
        hash_manager: FileHashManager,
    ):
        self.collection = collection
        self.lock = lock
        self.text_splitter = text_splitter
        self.hash_manager = hash_manager

    def index_files(self, files: list[str]) -> CollectionResult:
        """Index only changed files (by mtime)"""
        files_to_process = self._get_files_to_process(files)
        files_indexed = []
        errors = []
        for file in files_to_process:
            try:
                chunks = self.text_splitter.split(file)
                norm_file = str(Path(file).resolve())
                for chunk_index, chunk in enumerate(chunks):
                    if chunk.strip():
                        self._add_chunk(chunk, norm_file, chunk_index)
                files_indexed.append(file)
                self.hash_manager.update(file)
            except Exception as e:
                errors.append(f"Error processing file {file}: {e}")
                continue
        if files_indexed:
            self.hash_manager.save(self.hash_manager.file_hashes)
        return CollectionResult(files=files_indexed, errors=errors)

    def remove_files(self, files: list[str]) -> list[str]:
        """Delete chunks for given source paths from collection and hash file"""
        files_removed = []
        file_hashes = self.hash_manager.file_hashes
        for file in files:
            norm_file = str(Path(file).resolve())
            try:
                with self.lock:
                    # check if file exists in collection
                    result = self.collection.get(where={"source": norm_file})
                    if not result.get("ids"):
                        continue
                    # delete file from collection
                    self.collection.delete(where={"source": norm_file})
                    files_removed.append(norm_file)
                if norm_file in file_hashes:
                    del file_hashes[norm_file]
            except Exception as e:
                logger.error(f"Error removing docs from chroma: {e}")
        if files_removed:
            self.hash_manager.save(file_hashes)
        return files_removed

    def clear(self) -> None:
        """Delete all documents in collection and clear hash file."""
        try:
            with self.lock:
                existing = self.collection.get(include=[])
                ids = existing.get("ids") or []
                if ids:
                    self.collection.delete(ids=ids)
                self.hash_manager.file_hashes = {}
                self.hash_manager.save(self.hash_manager.file_hashes)
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def _get_files_to_process(self, files: list[str]) -> list[str]:
        """Return list of files that are new or have mtime different from stored hash."""
        files_to_process = []
        for file in files:
            norm_file = str(Path(file).resolve())
            file_stat = Path(norm_file).stat()
            file_mtime = file_stat.st_mtime
            file_hashes = self.hash_manager.file_hashes
            if norm_file in file_hashes and file_hashes[norm_file] == file_mtime:
                continue
            files_to_process.append(norm_file)
        return files_to_process

    def _add_chunk(self, chunk: str, source: str, chunk_index: int) -> None:
        """Add a chunk to collection"""
        chunk_id = self._generate_md5_hash(chunk, source)
        # Extract rule_id from the chunk's leading rule header (if any). Reuses the
        # same pattern the splitter uses so bold/plain/multi-level headers all match.
        first_line = chunk.split("\n", 1)[0]
        rule_id_match = RULE_HEADER_PATTERN.match(first_line)
        with self.lock:
            existing = self.collection.get(ids=[chunk_id], include=["metadatas"])
            # skip if chunk already exists, but ensure rule_id is set if chunk matches
            if existing and existing.get("ids"):
                if not rule_id_match:
                    return
                existing_meta = existing.get("metadatas")
                if existing_meta and existing_meta[0].get(
                    "rule_id"
                ) != rule_id_match.group(1):
                    new_meta = {
                        **existing_meta[0],
                        "source": source,
                        "chunk_index": chunk_index,
                        "rule_id": rule_id_match.group(1),
                    }
                    self.collection.update(
                        documents=[chunk], metadatas=[new_meta], ids=[chunk_id]
                    )
                return
            meta = {"source": source, "chunk_index": chunk_index}
            if rule_id_match:
                meta["rule_id"] = rule_id_match.group(1)
            self.collection.add(documents=[chunk], metadatas=[meta], ids=[chunk_id])

    @staticmethod
    def _generate_md5_hash(text: str, source: str) -> str:
        """Generate chunk id from source path and content."""
        data = f"{source}:{text}"
        return hashlib.md5(data.encode("utf-8")).hexdigest()

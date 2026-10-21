# Grounded RAG – issues & learnings

## 1. Indexing / duplicates

**Issue:** Same file indexed twice, duplicate results.

**Cause:** Inconsistent path representation (sometimes `Path(filename)`, sometimes plain `filename`) → different `source` and different MD5 → same content stored under different ids.

**Fix:** Normalised path everywhere: `norm_file = str(Path(file).resolve())` for hashing and for metadata `source`. Used it in both add and (when applicable) skip/update paths.

**Re-index without wiping DB:** If only code changes (e.g. adding `rule_id`), re-run index; existing chunks get metadata updated via `collection.update()`. Only delete `chroma_db` when **chunk_size**, **chunk_overlap**, or split logic changes.

---

## 2. Chunking & retrieval

**Issue:** Small chunks (500), large PDFs → poor retrieval (distance > 1, content split across chunks).

**Tried:**
- Larger size: `chunk_size 2000`, `overlap 200`, `n_results 8` → 487 chunks; still high distance and split content.
- Split on all markdown headers `#`, `##`, `###` → 1399 chunks, avg length 593 (too many small chunks).
- Split on chapter-level only: pattern `^##\s+\*\*\d+\s` (e.g. `## **2 Preprocessor**`), not `## **2.1 PRE30-C**` → ~15 sections per doc, then `RecursiveCharacterTextSplitter` sub-splits by size → 489 + 349 chunks, avg ~1770. Avoids ~1800 chunks from splitting on every `##`.

**Issue:** Semantic search didn’t always return the specific rule-definition chunk first (e.g. PRE30-C, DCL30-C).

**Fix:** Rule-id metadata + hybrid retrieval:
- On index: detect rule headers (`## **N.M RULE-ID-C.`), set `rule_id` in chunk metadata.
- On query: if message contains a rule ID, run `collection.get(where={"rule_id": rule_id})` and prepend those chunks (distance 0.0); then `collection.query(n_results=n_results*2)`; merge with data deduplication by doc_id and trim to `n_results` (2× so that after dropping duplicates we still fill the list).
- Try/except only around the rule boost so optional path fails silently; semantic query failures propagate.

**Config change:** After changing `chunk_size` / `chunk_overlap` or split logic, run `rm -rf chroma_db` (or equivalent) and re-index so all chunks are recreated with the new strategy.

---

## 3. Performance optimization: Multi-threading

**Issue:** Pdf to markdown conversion and indexing too slow

**Cause:** Sequential file processing

**Fix:** Multi-threading with parallel file processing and PDF conversion

### Benchmark Results

#### Test Configuration
- **Files:** 2 PDFs: 969 pages total (~485 pages avg)
    - converted to 2 markdown: 33,612 lines total (~16,806 lines avg)
- **Test method:** Clean database state (fresh `chroma_db/`)

| Configuration | Time/file | PDF Conv. | Indexing | Speedup | Notes |
|--------------|------------|-----------|----------|---------|-------|
| Sequential | ~706s | - | - | - | Processed .md twice (bug) |
| 2 threads (add_docs only) | 670.98s | - | - | 1.05x | Processed .md twice (bug) |
| Sequential | 301.14s | - | - | 2.3x | After duplicate fix |
| 2 threads (add_docs only) | 302.4s | - | - | 2.3x | After duplicate fix |
| 2 PDF + 2 add_docs threads | ~273.25s | 147.70s | 114.72s | 2.6x | After duplicate fix |

#### Sequential (Baseline)
Total time: ~1413 seconds (1414.29, 1411.97)
Average time per file: ~706 seconds (707.14, 705.98)

### Sequential (After duplicate fix)
- Store file hashes to prevent chunking file again or check modification time
- Normalise path to prevent duplicate entries

#### Multi-threaded (2 PDF threads + 2 add_docs threads)

- ~2.6 times faster than sequential
- **Total time:** ~546.50 seconds (avg)
    - First run: 524.84 seconds
    - Second run: 572.25 seconds
    - Third run: 542.42 seconds
- Average time per file: 273.25 seconds
- Average PDF conversion: 147.70 seconds per PDF
- Average ChromaDB indexing: 114.72 seconds per file

### Findings

1. **Duplicate file fix was the primary bottleneck:** The jump from ~706s to 301.14s per file (sequential with duplicate fix) shows that fixing duplicate `.md` file processing was the main performance issue (~405s improvement per file). Files were being chunked twice before the fix.

2. **Threading add_docs alone provides minimal/no benefit:** With duplicate fix enabled, threading only `add_docs` (sequential PDF conversion) shows essentially no improvement (301.14s → 302.44s per file, within measurement variance). This is likely due to lock contention serializing ChromaDB operations, negating threading benefits.

3. **Threading PDF conversion provides the real benefit:** Adding PDF conversion threading improves performance (~29s per file: 302.44s → 273.25s), achieving **2.6x speedup** over sequential processing.

4. **Optimal configuration:** 2 PDF conversion threads + 2 add_docs threads achieves **2.6x speedup** over sequential processing. The speedup primarily comes from parallel PDF conversion, not from threading add_docs.

### Implementation
- PDF conversion: `ThreadPoolExecutor(max_workers=2)` in `_extract_text_from_pdfs()`
- File processing: `ThreadPoolExecutor(max_workers=2)` in `add_docs_to_chroma()`
- Thread safety: `threading.Lock()` protects ChromaDB operations in `_add_doc_to_chroma()`
- Duplicate file fix: `md_files_from_pdfs` variable to track files from markdown files created from pdfs

---

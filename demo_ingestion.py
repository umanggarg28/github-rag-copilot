"""
demo_ingestion.py — Run the Phase 1 ingestion pipeline on a small public repo.

Usage:
  python demo_ingestion.py

What this does:
  1. Fetches a small public repo via GitHub API (no clone)
  2. Filters files by language rules
  3. Chunks Python files with AST; others with character windows
  4. Embeds all chunks with nomic-embed-code
  5. Upserts into Qdrant Cloud

Run this to verify the full Phase 1 pipeline works end-to-end before
building the FastAPI backend on top of it.
"""

from ingestion.repo_fetcher  import fetch_repo_files, get_repo_metadata
from ingestion.file_filter   import should_index, language_from_path
from ingestion.code_chunker  import chunk_files
from ingestion.embedder      import Embedder
from ingestion.qdrant_store  import QdrantStore

# A small, well-known repo — good for testing
TEST_REPO = "https://github.com/karpathy/micrograd"


def main():
    print("=" * 60)
    print("Cartographer — Ingestion Demo")
    print("=" * 60)

    # ── Step 1: Fetch repo metadata ───────────────────────────────────────────
    print("\n[1/5] Fetching repo metadata...")
    meta = get_repo_metadata(TEST_REPO)
    print(f"  Repo:        {meta['repo']}")
    print(f"  Description: {meta['description']}")
    print(f"  Language:    {meta['language']}")
    print(f"  Stars:       {meta['stars']}")

    # ── Step 2: Fetch filtered files ──────────────────────────────────────────
    print("\n[2/5] Fetching files...")
    files = fetch_repo_files(TEST_REPO, file_filter_fn=should_index)
    print(f"  Files to index: {len(files)}")
    for f in files:
        lang = language_from_path(f["path"])
        print(f"    {f['path']} ({lang}, {f['size']} bytes)")

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    print("\n[3/5] Chunking files...")
    chunks = chunk_files(files)
    print(f"\n  Sample chunks:")
    for chunk in chunks[:3]:
        print(f"    [{chunk['chunk_type']}] {chunk['filepath']}:{chunk['start_line']}–{chunk['end_line']}")
        if chunk['name']:
            print(f"      name: {chunk['name']}")
        print(f"      text preview: {chunk['text'][:120].strip()!r}")

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    print("\n[4/5] Embedding chunks...")
    embedder = Embedder()
    vectors  = embedder.embed_chunks(chunks)
    print(f"  {len(vectors)} vectors × {len(vectors[0])} dims")

    # ── Step 5: Store in Qdrant ───────────────────────────────────────────────
    print("\n[5/5] Storing in Qdrant Cloud...")
    store = QdrantStore()
    store.upsert_chunks(chunks, vectors)

    total = store.count(repo=meta["repo"])
    print(f"\n  Total chunks stored for {meta['repo']}: {total}")
    print(f"\n✓ Ingestion complete. Try querying with the backend API.")


if __name__ == "__main__":
    main()

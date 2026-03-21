# Note 001 — Phase 1: Core Ingestion Pipeline

**Date:** 2026-03-22
**Phase:** 1 — Core Ingestion

---

## What was built

- `ingestion/repo_fetcher.py` — downloads a GitHub repo as a zip (one API call), extracts files in memory, returns filtered file dicts
- `ingestion/file_filter.py` — two-layer filter: excluded directories + included extensions. Skips node_modules, lock files, binaries, build output
- `ingestion/code_chunker.py` — AST chunking for Python (per function/class), character-window fallback for everything else
- `ingestion/embedder.py` — wraps `nomic-ai/nomic-embed-code` with correct search_document/search_query prefixes
- `ingestion/qdrant_store.py` — Qdrant Cloud client: creates collection (dense + sparse), upserts chunks with stable hash IDs
- `backend/config.py` — central settings via .env
- `.env.example` — template for required env vars
- `demo_ingestion.py` — end-to-end pipeline smoke test on karpathy/micrograd

---

## Key decisions

**Zip download over git clone:** One API call vs N calls (one per file). No git dependency on the server. Works on Render free tier.

**AST chunking for Python only:** Python's `ast` module is stdlib — zero dependencies. Multi-language AST (tree-sitter) adds complexity for a learning project. Python + fallback covers the majority of repos cleanly.

**Stable point IDs:** MD5 of `repo::filepath::start_line` — same chunk always gets the same Qdrant point ID. Re-ingesting a repo is safe (upsert overwrites, no duplicates).

**Sparse vectors from hash-tokenisation:** Simple `Counter` of tokens mapped to hash indices. Not true BM25 (no IDF weighting) — that's handled by Qdrant at query time. This keeps the client code simple.

**nomic-embed-code prefixes:** `search_document:` for chunks, `search_query:` for queries. Required by the model — without them similarity scores degrade slightly.

---

## Concepts learned

- **AST (Abstract Syntax Tree):** A tree representation of code structure. `ast.parse()` turns source text into a tree of nodes (FunctionDef, ClassDef, etc.) that you can walk programmatically.
- **Sparse vectors:** Unlike dense vectors (every dimension has a value), sparse vectors only store non-zero positions — efficient for vocabulary-sized spaces (millions of tokens).
- **Qdrant collections:** Like a database table with a declared schema. Must specify vector dimensions and distance metric upfront.
- **Upsert:** Insert-or-update. If a point with the same ID exists, overwrite it. Makes re-ingestion safe without deduplication logic.

---

## What's next

Phase 2: Retrieval & Generation
- `retrieval/retrieval.py` — hybrid search (dense + sparse) via Qdrant
- `backend/services/generation.py` — LLM answer with code-aware system prompt
- FastAPI endpoints: `/ingest`, `/query`, `/search`

# Note 000 — Project Setup

**Date:** 2026-03-22
**PR:** Initial setup (no PR — baseline)

---

## What was set up

- Project structure created: `backend/`, `ingestion/`, `retrieval/`, `notes/`, `.claude/`
- `PLAN.md` written with full architecture, phases, and tech stack decisions
- `CLAUDE.md` written with project instructions for Claude Code
- `LEARN.md` started — will grow as each phase is built
- Git repo initialized

---

## Key architectural decisions

**Why Qdrant over ChromaDB?**
ChromaDB is local-only — data lives on disk and disappears if you redeploy.
Qdrant Cloud has a permanent free tier (1GB), making the app deployable without
paying for storage. It also has native hybrid search (sparse + dense vectors),
eliminating the need for our manual BM25 index.

**Why nomic-embed-code over all-MiniLM-L6-v2?**
`all-MiniLM-L6-v2` was trained on natural language. Code has different patterns:
identifier names, function signatures, call chains. `nomic-embed-code` was
fine-tuned on code and produces better semantic similarity for code queries.

**Why AST chunking over character windows?**
Character windows split wherever they hit the size limit — often mid-function.
A function is the natural unit of code: it has a name, a purpose, inputs/outputs.
Chunking at function boundaries keeps each chunk semantically complete and makes
citations meaningful ("see `embed_text()` in `retrieval/embedder.py`").

---

## What's next

Phase 1: Core ingestion pipeline
- `repo_fetcher.py` — clone public repos
- `file_filter.py` — skip binaries, lock files, node_modules
- `code_chunker.py` — AST-based chunking for Python

# GitHub RAG Copilot — Claude Code Instructions

This file is read by Claude Code at the start of every session.
It tells Claude how to work in this project.

---

## Project Purpose

A RAG system that indexes GitHub repositories and answers questions about code.
This is a **learning project** — prioritise clarity and explanation over brevity.

---

## Architecture at a Glance

```
ingestion/          ← repo fetching, file filtering, AST chunking, embedding
retrieval/          ← Qdrant hybrid search, BM25 sparse vectors
backend/            ← FastAPI: /ingest, /query, /search endpoints
  services/         ← ingestion_service.py, retrieval_service.py, generation.py
  routers/          ← ingest.py, query.py
  models/           ← schemas.py (Pydantic models)
ui/                 ← React + Vite frontend
notes/              ← Updated after every PR (NNN-title.md)
PLAN.md             ← Build plan and phase tracking
LEARN.md            ← Learning guide, updated as features are built
```

---

## Coding Rules

- Write comments explaining **why**, not what — this is a learning project
- Each new concept gets a docstring explaining it from first principles
- Prefer explicit over implicit — avoid magic
- No LangChain, no LlamaIndex — build from scratch so concepts are visible
- Write comments explaining **why**, not what

---

## Notes Convention

After every significant feature (PR-worthy), add an entry to `notes/`:
- Filename: `NNN-short-title.md` (zero-padded, e.g. `001-ingestion.md`)
- Contents: what was built, key decisions, concepts learned, what's next

---

## Running the Project

```bash
# Backend
cd github-rag-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload

# Frontend
cd ui && npm install && npm run dev
```

---

## Environment Variables

```
QDRANT_URL=         # Qdrant Cloud cluster URL
QDRANT_API_KEY=     # Qdrant Cloud API key
GROQ_API_KEY=       # LLM (free)
ANTHROPIC_API_KEY=  # LLM fallback (optional)
GITHUB_TOKEN=       # Optional — increases API rate limit from 60 to 5000 req/hr
```

---

## Slash Commands Available

- `/ingest-repo` — ingest a GitHub repository by URL
- `/search-code` — search the index without generating an answer
- `/add-to-notes` — add a note entry for the current work

---

## Key Design Decisions (don't change without good reason)

- **Qdrant Cloud** for vector storage (not ChromaDB) — enables free deployment
- **AST chunking** at function/class boundaries — not character windows
- **nomic-embed-code** embedding model — code-optimised, not general text
- **Qdrant native hybrid search** — replaces manual BM25 index + RRF fusion
- **No auth required** for public repo ingestion — GitHub API unauthenticated allows 60 req/hr, with token 5000 req/hr

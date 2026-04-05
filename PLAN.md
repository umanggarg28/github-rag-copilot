# Cartographer — Build Plan

A RAG system that indexes GitHub repositories and answers natural language
questions about their code, architecture, and documentation.

---

## Learning Objectives

By the end of this project you will understand:
- How RAG works on source code (not just documents)
- AST-based code chunking vs. fixed character windows
- Code-aware embeddings vs. general text embeddings
- Metadata-rich retrieval (file, function, class, language, line numbers)
- Hosted vector databases (Qdrant Cloud) and why they enable free deployment
- Live deployment: frontend on Vercel, backend on Render, vectors on Qdrant Cloud
- Claude Code features: CLAUDE.md, hooks, slash commands, subagents

---

## Architecture Overview

```
GitHub URL
    │
    ▼
[Ingestion Pipeline]
    ├── Fetch repo via GitHub API (no clone needed for public repos)
    ├── Filter files by language — skip binaries, lock files, node_modules
    ├── Chunk by AST boundaries (functions, classes)
    │       └── fallback: character windows for markdown, config, plain text
    ├── Embed with nomic-embed-code (code-optimised model)
    └── Store in Qdrant Cloud
            └── metadata: repo, filepath, language,
                         function_name, class_name, start_line, end_line

    │
    ▼
[Query Pipeline]
    ├── Embed query with same model
    ├── Hybrid search (dense vector + sparse BM25, native in Qdrant)
    ├── Relevance threshold (reject out-of-domain queries)
    └── LLM generation (Groq / Claude)
            └── citations: filepath + line range
```

---

## Phases

### Phase 1 — Core Ingestion
- [ ] `ingestion/repo_fetcher.py` — fetch file tree + content via GitHub API
- [ ] `ingestion/file_filter.py` — include/exclude rules per language
- [ ] `ingestion/code_chunker.py` — AST-based chunking for Python; character-window fallback for other file types
- [ ] `ingestion/embedder.py` — embed chunks with `nomic-ai/nomic-embed-code`
- [ ] `ingestion/qdrant_store.py` — upsert chunks into Qdrant Cloud collection

### Phase 2 — Retrieval & Generation
- [ ] `retrieval/retrieval.py` — hybrid search using Qdrant's native dense + sparse
- [ ] `backend/services/generation.py` — LLM answer generation with code-aware system prompt
- [ ] `backend/services/ingestion_service.py` — orchestrate full ingestion pipeline
- [ ] FastAPI backend with `/ingest`, `/query`, `/search` endpoints

### Phase 3 — UI
- [ ] React + Vite frontend
- [ ] Repo URL input instead of file upload
- [ ] Citations show filepath + line numbers
- [ ] Syntax-highlighted code chunks in source passages
- [ ] Multi-repo selector in sidebar

### Phase 4 — Live Deployment
- [ ] **Frontend → Vercel** (free, static hosting)
- [ ] **Backend → Render** (free tier — lightweight since no local ML model)
- [ ] **Vector DB → Qdrant Cloud** (permanent free tier, 1GB)
- [ ] **Embeddings → Qdrant's built-in vectoriser** or Voyage AI API (removes model from backend, keeps Render on free tier)
- [ ] Environment variable setup, CORS configuration
- [ ] GitHub Actions CI: lint + deploy on push to main

### Phase 5 — Claude Code Features (Throughout)
- [ ] `CLAUDE.md` — project briefing for Claude Code sessions
- [ ] Hooks — auto-lint on file edit, reminder to update notes after commit
- [ ] Slash commands — `/ingest-repo`, `/search-code`, `/add-to-notes`
- [ ] Subagent patterns — parallel ingestion, expert review before PRs

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Repo fetch | GitHub REST API | No local clone needed; works without git installed |
| Code parsing | `ast` (Python), `tree-sitter` (multi-lang) | Split at function/class boundaries |
| Embeddings | `nomic-ai/nomic-embed-code` | Fine-tuned on code, free, runs locally |
| Vector DB | Qdrant Cloud (free tier) | Permanent free 1GB, native hybrid search, enables deployment |
| LLM | Groq Llama 3.3 70B / Claude Haiku | Fast, cheap/free |
| Backend | FastAPI + Uvicorn | Lightweight, async, auto-docs |
| Frontend | React + Vite | Fast dev server, small production bundle |
| Frontend hosting | Vercel | Free, zero-config for Vite apps |
| Backend hosting | Render | Free tier works once model is removed from server |
| CI/CD | GitHub Actions | Lint and deploy on push |

---

## Deployment Architecture

```
User browser
    │
    ├── Static files ──→ Vercel (free)
    │                        React UI
    │
    └── API calls ──────→ Render (free)
                              FastAPI backend
                                  │
                                  ├──→ Qdrant Cloud (free)
                                  │        Vector storage + hybrid search
                                  │
                                  └──→ Groq API (free)
                                           LLM generation
```

The key insight: by using **Qdrant Cloud** for vector storage and a **remote
embedding API** (instead of running the model on the server), the backend
becomes a lightweight HTTP service with minimal RAM usage — fitting within
Render's free tier (512MB RAM).

---

## Notes Directory

`notes/` is updated after every PR:
- What was built
- Key decisions made
- Concepts learned
- What's next

See `notes/000-project-setup.md` for the first entry.

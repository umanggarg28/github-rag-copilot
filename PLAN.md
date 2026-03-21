# GitHub RAG Copilot — Build Plan

A RAG system that indexes GitHub repositories and answers questions about their
code, architecture, and documentation. Extends the PDF RAG Copilot concepts to
source code.

---

## Learning Objectives

By the end of this project you will understand:
- How RAG applies to code (not just documents)
- AST-based code chunking vs. character-window chunking
- Code-aware embeddings vs. general text embeddings
- Metadata-rich retrieval (file, function, class, language)
- Hosted vector DB (Qdrant Cloud) vs. local (ChromaDB)
- Claude Code features: CLAUDE.md, hooks, slash commands, subagents

---

## Architecture Overview

```
GitHub URL
    │
    ▼
[Ingestion Pipeline]
    ├── Clone / fetch repo via GitHub API
    ├── Filter files (language-aware)
    ├── Chunk by AST boundaries (functions, classes)
    │       └── fallback: character windows (markdown, config)
    ├── Embed with code-optimized model
    └── Store in Qdrant Cloud (vector + metadata)
            └── metadata: repo, filepath, language,
                         function_name, class_name, start_line

    │
    ▼
[Query Pipeline]  ← identical to PDF RAG
    ├── Embed query
    ├── Hybrid search (semantic + BM25 via Qdrant)
    ├── Relevance threshold
    └── LLM generation (Groq / Claude)
            └── citations: filepath + line range
```

---

## Phases

### Phase 1 — Core Ingestion (Week 1)
- [ ] `repo_fetcher.py` — clone or fetch via GitHub API (no auth needed for public repos)
- [ ] `file_filter.py` — include/exclude rules per language, skip binaries/lock files
- [ ] `code_chunker.py` — AST-based chunking for Python; character-window fallback
- [ ] `embedder.py` — reuse from PDF RAG, swap model to `nomic-ai/nomic-embed-code`
- [ ] `qdrant_store.py` — replace ChromaDB with Qdrant client

### Phase 2 — Retrieval & Generation (Week 1–2)
- [ ] `retrieval.py` — hybrid search using Qdrant's native BM25 + vector
- [ ] `generation.py` — reuse from PDF RAG, update system prompt for code answers
- [ ] FastAPI backend with `/ingest`, `/query`, `/search` endpoints

### Phase 3 — UI (Week 2)
- [ ] Reuse PDF RAG UI structure
- [ ] Input: GitHub URL instead of PDF upload
- [ ] Citations show filepath + line numbers instead of page numbers
- [ ] Syntax highlighting for code chunks in source passages

### Phase 4 — Claude Code Features (Throughout)
- [ ] `CLAUDE.md` — project instructions for Claude Code
- [ ] Hooks — auto-lint on edit, auto-update notes on commit
- [ ] Slash commands — `/ingest-repo`, `/search-code`, `/add-to-notes`
- [ ] Subagent patterns — parallel ingestion of multiple repos

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Repo fetch | `gitpython` + GitHub API | No auth for public repos |
| Code parsing | `ast` (Python), `tree-sitter` (multi-lang) | Function/class boundaries |
| Embeddings | `nomic-ai/nomic-embed-code` | Trained on code, free |
| Vector DB | Qdrant Cloud (free tier) | Permanent free, 1GB |
| Keyword search | Qdrant sparse vectors (BM25) | Native, no separate index |
| LLM | Groq Llama 3.3 70B / Claude | Same as PDF RAG |
| Backend | FastAPI | Same as PDF RAG |
| Frontend | React + Vite | Same as PDF RAG |

---

## Key Differences vs PDF RAG

| Concern | PDF RAG | GitHub RAG |
|---|---|---|
| Ingestion source | Local file upload | GitHub URL |
| Chunking strategy | Fixed character windows | AST-aware (function/class) |
| Metadata | source, page | repo, filepath, language, function, line |
| Vector DB | ChromaDB (local) | Qdrant Cloud (hosted) |
| Embedding model | all-MiniLM-L6-v2 | nomic-embed-code |
| Citations | Paper name + page | Filepath + line range |
| Hybrid search | Manual RRF | Qdrant native hybrid |

---

## Notes Directory

`notes/` is updated after every PR with:
- What was built
- Key decisions made
- Concepts learned
- What's next

See `notes/000-project-setup.md` for the first entry.

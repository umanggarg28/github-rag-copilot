---
title: Cartographer
emoji: 🧭
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# Cartographer

**A production-grade RAG system that maps any GitHub repository — built from scratch, without LangChain or LlamaIndex.**

Index any public repo and ask natural-language questions about its code. Cartographer retrieves the exact functions and classes relevant to your question, explains them with source citations, and can autonomously investigate complex questions across multiple files using an agent with 10 specialised tools. It can also generate a full README for any indexed repo on demand.

**Live:** [cartographer-app.vercel.app](https://cartographer-app.vercel.app) · **Backend:** [HuggingFace Spaces](https://huggingface.co/spaces/umanggarg/cartographer)

---

## What Makes This Different

Most RAG tutorials wrap a library and call it done. This project implements every layer manually:

- **Ingestion** — GitHub API → AST-based code chunking → contextual LLM descriptions → dual-vector embedding → Qdrant Cloud
- **Retrieval** — HyDE + query expansion + native hybrid search (dense + BM25) + cross-encoder reranking — each stage independently improving recall and precision
- **Agent** — a ReAct loop with 10 MCP tools, working memory, parallel tool execution, and streaming thought traces
- **UI** — the pipeline is visible: every retrieved chunk, agent thought, tool call, and confidence grade is shown to the user

The result is both a useful tool and a study in how production AI systems are actually built.

---

## AI Pipeline

### Ingestion

```
GitHub URL
  → repo_fetcher.py      Download zip via GitHub API (no disk writes, memory-streamed)
  → file_filter.py       Keep .py .ts .go .rs .md; skip node_modules, dist, *.lock, >100KB
  → code_chunker.py      tree-sitter AST → functions + classes as atomic chunks
                         Falls back to line-windowed sliding chunks for unsupported languages
  → ingestion_service.py (Optional) LLM generates a 1–2 sentence description per chunk
                         prepended before embedding — Anthropic's "contextual retrieval"
  → embedder.py          Nomic nomic-embed-text-v1.5 (768-dim) via API · optional Voyage voyage-code-3 (1024-dim)
  → qdrant_store.py      Each chunk stored with: dense vector + sparse BM25 vector + full payload metadata
```

Every chunk carries: `repo`, `filepath`, `language`, `chunk_type`, `name`, `start_line`, `end_line`, `calls`, `imports`, `base_classes`. This metadata powers search filters, diagram generation, and citation rendering.

**Why AST chunking?** Splitting at character boundaries breaks functions mid-signature. AST chunking extracts complete, self-contained units — every retrieved chunk can stand alone. Function signatures, bodies, and return types are never separated.

**Why contextual retrieval?** A generic function name like `forward()` exists in thousands of codebases. Raw code embeddings capture syntax, not meaning. Prepending an LLM description ("this is the forward pass of the multi-head attention layer in a decoder-only transformer") shifts the embedding toward semantics. Anthropic benchmarks show 35–49% retrieval improvement.

### Retrieval (every query)

```
Question
  ├─ HyDE: LLM generates a hypothetical code snippet that would answer the question
  │        Embed the snippet instead of the question — searches code-space-to-code-space
  │
  ├─ Query Expansion: LLM generates 2 alternative phrasings of the question
  │
  └─ Hybrid Search × 3 variants (Qdrant native, single round-trip per variant):
       dense cosine similarity (semantic) + sparse BM25 (exact identifiers)
       Server-side RRF fusion → top-24 candidates per variant

  → Client-side RRF across all 3 variant result sets → top-24 merged candidates

  → Cohere rerank-v3.5 (cross-encoder): reads (query, chunk) jointly → top-8 final results
    Fallback: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, no API)

  → LLM generation with citations → SSE token stream → browser
```

**Why hybrid search?** Semantic search finds conceptually related code even without exact words. BM25 catches precise identifiers (class names, error types, function names) that semantic search compresses away. Together they consistently outperform either alone.

**Why HyDE?** Natural-language questions and code live in different regions of embedding space. Embedding a *hypothetical answer* instead of the question removes this gap.

**Why reranking?** Bi-encoder retrieval (embed-then-compare) is fast but lossy. Cross-encoders read query and document together — they can reason about interactions between tokens. Too slow for full-corpus scoring, so we narrow to 24 candidates first, then rerank precisely.

### Generation

Top-8 reranked chunks are injected as numbered sources. The LLM cites by `[1]`, `[2]`, etc. Response streams token-by-token via SSE. After generation, a second LLM call grades faithfulness: **high / medium / low**, shown in the UI pipeline bar.

**Multi-provider cascade** (automatic failover, no interruption to user):
```
Cerebras llama-3.3-70b (2600 tok/s, fastest) → Groq → Gemini → OpenRouter → Anthropic
```

---

## Agent Mode

For complex questions requiring multi-file investigation ("trace how data flows from the loader into the model"), single-shot RAG is insufficient. The agent runs a **ReAct loop** — Reason, Act, Observe — calling tools iteratively until it can synthesise a complete answer.

### Architecture: Model Context Protocol (MCP)

The agent communicates with tools via **MCP** — the same open protocol Claude Code uses. A FastMCP server runs inside the FastAPI process at `/mcp`. The agent connects as an MCP client, discovers tools dynamically, and calls them via JSON-RPC.

This means every tool works with any MCP-compatible client, not just our agent.

### 10 Agent Tools

| Tool | What it does |
|------|-------------|
| `search_code` | Full retrieval pipeline (HyDE + hybrid search + rerank) for a natural-language query |
| `search_symbol` | Look up a specific class, function, or variable by exact name |
| `read_file` | Read any indexed file in full |
| `get_file_chunk` | Read a precise line range from a file |
| `list_files` | List all indexed files in a repo or subdirectory |
| `find_callers` | Find every call site of a function across the repo |
| `trace_calls` | Walk the call chain from a function to see what it calls and what calls it |
| `note` | Store a key-value fact in working memory for this session |
| `recall_notes` | Retrieve all stored notes — enables cross-iteration reasoning |
| `draw_diagram` | Generate a Mermaid diagram (flowchart, class, sequence) and stream it to the UI |

**Working memory** (`note`/`recall_notes`) lets the agent accumulate findings across iterations — it doesn't have to re-discover facts it already found. Notes are now **persistent across server restarts** — stored in a Qdrant sidecar collection and loaded at session start. This significantly improves coherence on long investigations.

**Parallel tool execution** — when the agent needs information from multiple independent sources, it fires them concurrently via `asyncio.gather`, not sequentially. This halves wall-clock time for multi-step investigations.

### Streaming Thought Traces

Every agent thought and tool call streams to the UI in real time via SSE. The UI renders a live timeline:
- **Thoughts** (agent reasoning) shown in full while streaming, collapse to a one-liner once the agent moves on — click to expand
- **Tool steps** show tool name, input query/args, and output — individually expandable
- The outer trace collapses after completion so the final answer is prominent

### Agent Model Selection

Three models available in the UI, selectable per conversation:

| Model | Provider | Strengths |
|-------|----------|----------|
| `qwen3-235b` | Cerebras | Best reasoning, 2600 tok/s |
| `gemma-4-31b` | Google (OpenRouter) | Strong code understanding |
| `gemini-flash` | Google | Lowest latency |

---

## Additional Features

### Explore View — Agent-Generated Concept Tour

For any indexed repo, Cartographer runs a **three-phase agent** to generate an interactive concept map that teaches you how to approach the codebase:

```
Phase 1 — MAP        Trace the main pipeline from entry file imports/calls
Phase 2 — INVESTIGATE  Deep-dive into each pipeline stage (one LLM call per file)
Phase 3 — SYNTHESIZE  Convert traced understanding into structured tour JSON
```

Each phase gets only the context it needs — no LLM call is asked to do two hard things simultaneously. This mirrors Claude Code's own `architecture_overview` + `explain_tool` prompt pattern, and produces grounded dependency graphs traced from actual imports rather than guessed from naming conventions.

During generation, a **live agent trace panel** shows each investigation step in real time — making the reasoning process visible and building trust in why each concept appears.

The result is rendered as an interactive concept map: 6–8 concepts, each with a non-obvious insight and the naive alternative that was rejected. Arrows show conceptual prerequisites (understand A before B). Click any card to expand its insight. Scroll to zoom, drag to pan.

### Diagram View

On-demand Mermaid diagrams from indexed metadata (no re-reading files):
- **Architecture** — module-level import relationships
- **Class hierarchy** — `base_classes` payload field → inheritance tree
- **Call graph** — `calls` payload field → function dependency graph

Diagrams stream progressively via SSE (progress events while the LLM generates, then the final Mermaid source). Rendered client-side, fullscreen-capable.

### Session History

Every conversation is persisted to `localStorage` per repo. The sidebar shows recent sessions with auto-generated titles; sessions are reloadable and renameable. Up to 10 sessions per repo.

### Pipeline Provenance Bar

After every RAG answer, a compact bar shows the exact path the query took:
```
HyDE → +2 expansions → hybrid search → cohere re-ranked → 8 sources → generated · gemma · ✓ high
```

Every badge is interactive — hover shows the full explanation of each stage.

### Query Classification

Queries are classified before retrieval: `implementation` / `architecture` / `debugging` / `comparison`. The classification adjusts the system prompt and retrieval strategy. Shown in the UI as a subtle mode tag on each response.

### README Generator

For any indexed repository, Cartographer can generate a complete, structured README on demand. Triggered via a hover-reveal document icon on sidebar repo items. The LLM is grounded in the repo map (AST metadata — class names, file layout, entry points) rather than raw source, keeping the prompt compact while producing accurate output. Results are cached to disk; the Regenerate button forces a fresh generation.

Quality improves when the repo is indexed with contextual retrieval enabled (`CONTEXTUAL_TOP_N > 0`) — the quality tip is shown inline in the view.

### Contextual Retrieval Re-indexing

The ⟳ button in the sidebar triggers a re-index with LLM-generated chunk descriptions. Repos indexed this way show a `✦` badge. The improvement is most noticeable for generic function names and repos with large shared utility layers.

---

## Tech Stack

| Layer | Technology | Detail |
|-------|-----------|--------|
| Backend | FastAPI + uvicorn | Async ASGI, 20+ endpoints, SSE streaming throughout |
| Frontend | React + Vite | Component-based UI, localStorage sessions, SSE token streaming |
| Vector DB | Qdrant Cloud | Native hybrid search (dense + sparse), free 1 GB tier |
| Embeddings (default) | Nomic `nomic-embed-text-v1.5` | 768-dim, via Nomic API (zero local RAM) |
| Embeddings (optional) | Voyage `voyage-code-3` | 1024-dim, code-optimised, 200M tokens/month free |
| Code parsing | tree-sitter | Multi-language AST — Python, JS, TS, Go, Rust, Java |
| Reranker (primary) | Cohere `rerank-v3.5` | Cross-encoder, API, 1000 calls/month free |
| Reranker (fallback) | `ms-marco-MiniLM-L-6-v2` | Local cross-encoder, baked into Docker image |
| LLM generation | Cerebras → Groq → Gemini → OpenRouter → Anthropic | Automatic cascade, all free tiers |
| Agent | MCP via FastMCP | JSON-RPC tool discovery + calling; compatible with Claude Code |
| Diagrams | Mermaid.js | Lazy-loaded, client-side rendering |
| Graph visualization | D3.js (force-directed) | Architecture diagram; Explore view uses topological layout |
| Analytics | Vercel Analytics | Page views and visitor tracking |
| Backend hosting | HuggingFace Spaces (Docker) | Free CPU tier, port 7860 |
| Frontend hosting | Vercel | Global CDN, instant deploys |
| CI/CD | GitHub Actions | Parallel deploy jobs: HF Spaces + Vercel on every push to `main` |

---

## Deployment Architecture

```
GitHub push to main
  │
  ├─ GitHub Actions: deploy-backend
  │    git subtree split --prefix=cartographer -b hf-deploy
  │    git push --force → huggingface.co/spaces/umanggarg/cartographer
  │    HF detects Dockerfile → builds image → runs container on port 7860
  │    (cross-encoder model pre-downloaded into image layer — no cold-start delay)
  │
  └─ GitHub Actions: deploy-frontend (parallel)
       npm ci → npx vercel --prod
       Vite builds dist/ with VITE_API_URL baked in as a literal string
       Vercel uploads to 100+ global CDN edge nodes
```

The backend (`FastAPI`) and frontend (`React`) are deployed independently. They communicate over HTTPS — the frontend calls `VITE_API_URL` (the HF Space URL) for all API requests.

**Why split deployment?** React is a static site (just files) — serving it from a container wastes resources and loses CDN benefits. Vercel's edge network delivers assets from the nearest node to every user. The FastAPI backend needs a runtime, which HuggingFace provides free via Docker.

---

## Setup

```bash
git clone https://github.com/umanggarg28/Cartographer
cd Cartographer

# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
uvicorn backend.main:app --reload
# → http://localhost:8000 · docs at /docs

# Frontend (separate terminal)
cd ui && npm install && npm run dev
# → http://localhost:5173
```

### Required Environment Variables

```bash
# Vector DB (required)
QDRANT_URL=          # Qdrant Cloud cluster URL
QDRANT_API_KEY=      # Qdrant Cloud API key
QDRANT_COLLECTION=   # e.g. cartographer_nomic

# Embeddings (one required)
NOMIC_API_KEY=       # Default — free at atlas.nomic.ai
VOYAGE_API_KEY=      # Optional upgrade — free at voyageai.com (set EMBEDDING_MODEL=voyage-code-3)

# LLM (at least one required)
CEREBRAS_API_KEY=    # Fastest — free at cloud.cerebras.ai (1M tok/day)
GROQ_API_KEY=        # free at console.groq.com
GEMINI_API_KEY=      # free at aistudio.google.com

# Optional
COHERE_API_KEY=      # Reranking — free 1000 calls/month at cohere.com
GITHUB_TOKEN=        # Raises GitHub rate limit 60 → 5000 req/hr
FRONTEND_URL=        # Your Vercel URL — required for CORS in production
```

---

## What This Demonstrates

Building Cartographer required implementing and integrating:

**Retrieval engineering** — AST chunking, embedding model selection, hybrid dense+sparse indexing, HyDE, query expansion, RRF fusion, cross-encoder reranking, contextual retrieval

**Agentic systems** — ReAct loop design, MCP protocol, tool schema definition, working memory, parallel async tool execution, streaming thought traces

**Production backend** — FastAPI async architecture with domain-separated routers (`ingestion`, `query`, `agent`, `diagrams`, `mcp_routes`), SSE streaming, multi-provider LLM fallback, rate limit handling, Docker containerisation, environment-based config

**Frontend engineering** — Real-time SSE token streaming, localStorage session persistence, D3 force-directed graphs, Mermaid diagram rendering, component-level state management

**Deployment & CI/CD** — GitHub Actions parallel jobs, `git subtree split` for monorepo HF deployment, Vercel edge CDN, Docker layer caching

Every component is implemented from scratch in readable, documented Python and React — no framework magic hiding the internals.

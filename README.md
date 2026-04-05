---
title: Cartographer
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Cartographer

A production-grade Retrieval-Augmented Generation (RAG) system that indexes any GitHub repository and answers questions about code — built entirely from scratch, without LangChain or LlamaIndex.

Every layer of the stack is implemented manually so the internals are visible and learnable. The UI exposes the AI reasoning pipeline — retrieval traces, agent tool calls, confidence scores, and 2D semantic maps — making abstract RAG concepts tangible.

---

## What This Is (and Why It's Built This Way)

Most RAG tutorials show you `from langchain import RAGChain` and call it a day. That teaches you nothing about *why* things work or break.

This project takes the opposite approach. Every concept — chunking, embedding, hybrid search, reranking, HyDE, contextual retrieval, agentic tool use, UMAP projection — is implemented as a discrete, readable module. You can read the code and understand exactly what is happening at each step.

**Dual purpose:**
- **Tool**: Ask questions about any GitHub codebase and get accurate, cited answers
- **Learning platform**: The UI makes the AI pipeline visible. You see which chunks were retrieved, how they were ranked, what the agent reasoned, and where your query landed in semantic space

---

## Architecture Overview

```
cartographer/
├── ingestion/
│   ├── repo_fetcher.py      # Download repo from GitHub API as zip
│   ├── file_filter.py       # Filter indexable files by extension
│   ├── code_chunker.py      # AST chunking (Python) + sliding window fallback
│   ├── embedder.py          # Voyage AI / Nomic API embedding client
│   └── qdrant_store.py      # Qdrant Cloud upsert, search, list, delete
├── retrieval/
│   └── retrieval.py         # Hybrid search + HyDE + query expansion + Cohere rerank
├── backend/
│   ├── main.py              # FastAPI app, 20+ endpoints, rate limiting, CORS
│   ├── config.py            # All env vars in one place
│   ├── mcp_server.py        # MCP server (search_code, get_file_chunk tools)
│   ├── mcp_client.py        # MCP client (used by agent to call tools)
│   ├── models/schemas.py    # Pydantic models for all API shapes
│   └── services/
│       ├── ingestion_service.py  # Orchestrates full ingestion pipeline
│       ├── generation.py         # LLM generation with provider cascade
│       ├── agent.py              # Agentic RAG via MCP tool calls
│       ├── diagram_service.py    # Mermaid + tour diagram generation
│       ├── graph_service.py      # Call graph / dependency graph
│       └── map_service.py        # 2D semantic map via UMAP/PCA
├── ui/src/
│   ├── App.jsx              # Main app, routing, state
│   ├── api.js               # All backend API calls
│   ├── index.css            # Design system (dark violet, LiteLLM-inspired)
│   └── components/
│       ├── Sidebar.jsx      # Repo ingest, mode selection, session list
│       ├── Message.jsx      # Chat messages, agent traces, pipeline provenance
│       ├── DiagramView.jsx  # Mermaid diagram renderer
│       └── ExploreView.jsx  # Semantic map + concept tour
├── Dockerfile               # HuggingFace Spaces deployment
├── PLAN.md                  # Build plan and phase tracking
└── LEARN.md                 # Learning guide
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI + uvicorn | Async, fast, excellent OpenAPI docs |
| Frontend | React + Vite | Fast dev server, minimal boilerplate |
| Vector DB | Qdrant Cloud (free, 1GB) | Native hybrid search + sparse vectors |
| Embeddings | Voyage AI `voyage-code-3` (1024-dim) | Code-optimised, 200M tokens/month free |
| Embedding fallback | Nomic `nomic-embed-text-v1.5` (768-dim) | General purpose, still excellent |
| LLM (RAG) | Groq `llama-3.3-70b-versatile` → Gemini → OpenRouter → Anthropic | Free cascade |
| LLM (Agent) | Gemini `gemini-2.0-flash` via MCP | Low latency, generous free tier |
| Reranker | Cohere Rerank `rerank-english-v3.0` | Cross-attention scoring, far better than bi-encoders |
| Reranker fallback | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local model, no API needed |
| Code parsing | tree-sitter | AST-level chunking for Python |
| Backend hosting | HuggingFace Spaces (Docker) | Free, Docker support |
| Frontend hosting | Vercel | Free, instant deploys |

Everything on this stack is free tier. See the [Free Tier Budget](#free-tier-budget) section.

---

## Ingestion Pipeline

The ingestion pipeline transforms a raw GitHub URL into a searchable vector index. Here is every step.

### Step 1 — Repo Fetching (`ingestion/repo_fetcher.py`)

Parses the GitHub URL (handles both `github.com/owner/repo` and full `https://` URLs), then downloads the repository as a zip archive from the GitHub API.

The zip is streamed into memory — nothing is written to disk. This keeps the deployment clean (no temp file cleanup, works in read-only containers).

GitHub rate limits:
- Unauthenticated: 60 requests/hour
- With `GITHUB_TOKEN`: 5,000 requests/hour

Setting a token is strongly recommended for any serious usage.

### Step 2 — File Filtering (`ingestion/file_filter.py`)

Not all files in a repository are worth indexing. The filter keeps:

- Code: `.py`, `.js`, `.ts`, `.tsx`, `.go`, `.rs`, `.java`, `.cpp`, `.c`, `.h`
- Config/docs: `.md`, `.yaml`, `.yml`, `.json`, `.toml`

It excludes:
- Package directories: `node_modules/`, `vendor/`, `.git/`
- Build artifacts: `dist/`, `build/`, `__pycache__/`, `*.pyc`
- Lock files: `package-lock.json`, `yarn.lock`, `poetry.lock`
- Test files (configurable — test code is often useful to index)
- Files larger than 100KB

The 100KB cap is deliberate. Large files (auto-generated, minified JS, data dumps) produce low-quality chunks that pollute the index.

### Step 3 — AST Chunking (`ingestion/code_chunker.py`)

This is where the project diverges most from naive implementations.

**The problem with sliding windows:** A typical approach splits code into 1000-character windows with 200-character overlap. This works for prose. For code it produces garbage: half a function body in one chunk, the other half in the next. Neither chunk is independently meaningful.

**The AST approach for Python:** We use `tree-sitter` to parse the abstract syntax tree. Every function definition and class definition becomes its own chunk, extracted at its exact boundaries. Each chunk carries metadata extracted from the AST:

```python
{
  "text": "def scaled_dot_product_attention(q, k, v, mask=None):...",
  "name": "scaled_dot_product_attention",
  "chunk_type": "function",
  "start_line": 42,
  "end_line": 61,
  "docstring": "Compute scaled dot-product attention.",
  "calls": ["torch.matmul", "torch.softmax", "F.dropout"],
  "imports": ["torch", "torch.nn.functional"],
  "base_classes": []
}
```

The `calls`, `imports`, and `base_classes` fields are not used for retrieval — they feed the diagram and graph services later.

**Other languages:** Sliding window with 1000-char size and 200-char overlap. We preserve line boundaries so chunks don't break mid-statement.

### Step 4 — Contextual Retrieval (`backend/services/ingestion_service.py`)

This is an optional enhancement, triggered by the re-index button (⟳) in the UI.

**The problem:** Embedding raw code captures syntax and token patterns, but not meaning. A function named `forward()` in a transformer model and a `forward()` in a queue implementation will have similar embeddings because the code structure is similar — even though they are conceptually unrelated.

**The fix:** Before embedding, for each chunk, we ask Groq to generate a 1-2 sentence description of what the chunk does in the context of the full file:

```
This is the forward pass of the multi-head attention layer in a decoder-only
transformer. It applies scaled dot-product attention with causal masking,
then projects back to the model dimension.
```

We prepend this description to the chunk text before embedding. The embedding now captures meaning, not just syntax.

This technique is from Anthropic's research on contextual retrieval. Their benchmarks show 35–49% improvement in retrieval accuracy. The improvement is especially large for generic function names (`forward`, `process`, `run`, `update`) that appear in many codebases.

Rate-limit handling: contextual retrieval makes one Groq call per chunk. A large repo might have 2,000+ chunks. We process all of them but skip gracefully on 429 errors — those chunks fall back to raw code embedding.

### Step 5 — Embedding (`ingestion/embedder.py`)

**Primary model: Voyage AI `voyage-code-3`**

Voyage AI's code embedding model is trained specifically on code. It understands that `self.attention = MultiHeadAttention(d_model, num_heads)` is semantically related to "how is attention initialised?" — even though the surface text barely overlaps. General-purpose embedding models handle this worse because they're trained on natural language.

Dimensions: 1024. Free tier: 200M tokens/month (more than enough for any project).

**Fallback model: Nomic `nomic-embed-text-v1.5`**

768-dim, general purpose, still very capable. Used if Voyage API is unavailable or keys are missing.

**Task types:** Both models support `search_document` (for indexing) and `search_query` (for queries). This is important — the models use slightly different encodings depending on task type. Always use the right one.

**Deduplication via SHA-256:** Before sending a chunk for embedding, we hash its text. If that hash already exists in our index (e.g., a utility function copied between repos), we skip the embedding API call and reuse the existing vector. On large repos with shared code, this can skip 10–30% of embedding calls.

**Batching:** 256 texts per API call, with retry on 429/503.

### Step 6 — Storing in Qdrant (`ingestion/qdrant_store.py`)

Each chunk becomes a "point" in Qdrant Cloud with:

**Stable ID:** MD5 of `repo::filepath::start_line`. This means re-indexing the same file updates the existing point rather than creating duplicates.

**Two vector types per point:**

```python
vectors = {
    "code": [0.023, -0.41, ...],    # dense 1024-dim semantic embedding
    "bm25": SparseVector(...)        # sparse keyword vector
}
```

The sparse BM25 vector is computed from token frequencies. It enables exact keyword matching — if the user asks about `MultiHeadAttention`, the BM25 vector will surface chunks containing that exact string, even if the semantic embedding misses it.

**Payload fields:** `text`, `repo`, `filepath`, `language`, `chunk_type`, `name`, `start_line`, `end_line`, `calls`, `imports`, `base_classes`, `text_hash`

Batch size 50 per upsert to avoid Qdrant Cloud write timeouts.

---

## Retrieval Pipeline

The retrieval pipeline runs on every query. It has five stages, each independently improving the quality of what gets passed to the LLM.

### Stage 1 — HyDE: Hypothetical Document Embeddings

**The core insight:** When a user asks "how does backpropagation work?", we embed that question and search for code semantically close to it. But natural language questions and code live in different parts of embedding space — the question is phrased in English prose; the answer is Python code. The semantic gap reduces recall.

**The fix:** We ask Groq to generate a *hypothetical code snippet* that would answer the question:

```python
# Hypothetical answer to: "how does backpropagation work?"
def backward(self, grad_output):
    grad_input = grad_output * self.sigmoid_output * (1 - self.sigmoid_output)
    return grad_input
```

We embed this hypothetical code and use it for vector search. Now we're searching code-space-to-code-space. The hypothetical code doesn't have to be correct or runnable — it just needs to be semantically close to the real answer.

Measured improvement: ~20% better recall on technical questions.

### Stage 2 — Query Expansion

In parallel with HyDE, we ask Groq to rephrase the original query two ways:

```
Original:  "how does attention work?"
Variant 1: "how is attention computed?"
Variant 2: "what is the attention mechanism implementation?"
```

All three variants are searched. Results from all three are merged using **Reciprocal Rank Fusion (RRF)**.

**RRF formula:** For each candidate chunk, its score is the sum of `1 / (rank_in_result_list + 60)` across all result lists it appears in. Chunks that appear highly ranked in multiple result lists get high scores; chunks that appear in only one list are penalised by rank.

Why 60? It's a smoothing constant. Without it, a chunk ranked #1 in one list but #50 in all others would dominate. The 60 floors the rank-to-score mapping and rewards consistent appearance over a single fluke top ranking.

### Stage 3 — Hybrid Search (Qdrant Native)

For each query variant (3 total), Qdrant runs two searches in a single network round trip:

1. **Dense vector search**: cosine similarity on the 1024-dim embedding. Finds semantically related code even if the exact words don't match.
2. **Sparse BM25 search**: keyword matching via sparse vectors. Finds exact identifiers — class names, function names, variable names — that semantic search might miss.

Qdrant fuses these server-side with RRF. We fetch `top_k × 3` candidates (default: 24) per variant to give the reranker sufficient room to work.

Filters (repo name, language) are applied at the index level as pre-filters — much faster than filtering the result set post-retrieval.

### Stage 4 — Cohere Rerank

After hybrid search, we have up to 24 candidate chunks. The reranker selects the best 8.

**Why reranking is necessary:** Embedding models encode query and chunk *independently*. Each gets its own vector, and we compare them with cosine similarity. This is fast but lossy — the model never sees both together.

Cohere's reranker is a cross-encoder. It reads `(query, chunk)` as a single input and produces a relevance score from their joint representation. It can see:

- The query asks about backpropagation
- This chunk calls `loss.backward()` and sets `optimizer.zero_grad()`
- This is directly relevant — top result

This cross-attention scoring is far more accurate than bi-encoder cosine similarity. The tradeoff is speed — cross-encoders are too slow to score thousands of candidates, which is why we use hybrid search to narrow to 24 first, then rerank.

Fallback: if Cohere API is unavailable, we use `cross-encoder/ms-marco-MiniLM-L-6-v2` locally. It's smaller and less accurate but produces the same scoring mechanic.

### Stage 5 — Generation

The top 8 reranked chunks are formatted as numbered sources:

```
[1] repo: my-repo | file: models/transformer.py | lines 42-61
def scaled_dot_product_attention(q, k, v, mask=None):
    ...

[2] repo: my-repo | file: models/attention.py | lines 12-30
...
```

These go to Groq `llama-3.3-70b-versatile` with instructions to cite by source number (`[1]`, `[2]`, etc.) in the answer. The response streams token-by-token via Server-Sent Events (SSE) to the frontend.

**Provider cascade:** If Groq is rate-limited or unavailable, we try Gemini, then OpenRouter, then Anthropic. The cascade is transparent — the user sees no interruption.

**Confidence grading:** After generation, the answer is graded on two dimensions:
- **Faithfulness**: does the answer stay within what the retrieved chunks say?
- **Confidence**: high / medium / low, shown in the pipeline provenance bar in the UI

---

## Agent Mode

Plain RAG is a single-shot system: search once, answer from whatever is retrieved. Agent mode enables multi-step reasoning — search, read, search again with new information, read more deeply.

### Architecture: MCP (Model Context Protocol)

This project uses MCP — the same protocol that Claude Code uses to connect to external tools. There are two components:

**`backend/mcp_server.py`** — A FastMCP server mounted at `/mcp`. It exposes two tools:

```python
@mcp.tool()
def search_code(query: str, repo: str, language: str, mode: str) -> list[dict]:
    """Run the full retrieval pipeline (hybrid search + HyDE + rerank) and return chunks."""

@mcp.tool()
def get_file_chunk(repo: str, filepath: str, start_line: int, end_line: int) -> str:
    """Read a specific range of lines from a file in the indexed repo."""
```

**`backend/mcp_client.py`** — An HTTP client that connects to the MCP server. The agent uses this to call tools.

**`backend/services/agent.py`** — Gemini 2.0 Flash serves as the agent brain. It sees the question and available tools, decides what to call, and iterates until it has enough information.

### The Agent Loop

```
User: "How does the training loop work? What optimizer and scheduler are used?"

Turn 1:  search_code("training loop optimizer scheduler")
         → finds train.py, optimizer setup, learning rate scheduler

Turn 2:  get_file_chunk("my-repo", "train.py", 1, 80)
         → reads the full training file header

Turn 3:  search_code("learning rate warmup cosine annealing")
         → finds the scheduler configuration

Turn 4:  get_file_chunk("my-repo", "config/training.yaml", 1, 50)
         → reads hyperparameter config

Turn 5:  Synthesizes final answer from all evidence
```

The agent can follow import chains, read related config files, and build a fuller picture than any single search would provide. It runs up to 8 tool-call iterations by default.

**Why Gemini for the agent?** The agent needs a model that reliably generates structured tool-call JSON. Gemini 2.0 Flash does this well, has a generous free tier (1,500 req/day), and has low latency — important when each iteration adds user-perceived delay.

---

## Diagrams (`backend/services/diagram_service.py`)

Four diagram types, all generated from the chunk metadata already stored in Qdrant — no re-reading of files needed.

| Diagram | Format | Source Data |
|---------|--------|-------------|
| Architecture | Mermaid flowchart | Module-level import relationships |
| Class Hierarchy | Mermaid flowchart | `base_classes` payload field |
| Call Graph | Mermaid flowchart | `calls` payload field |
| Codebase Tour | JSON (not Mermaid) | LLM-generated reading guide |

The **Codebase Tour** is the most useful for onboarding. It generates a structured reading guide: entry point → core concepts → advanced topics, with each step explaining what to read and why. This is rendered as an interactive card sequence in the UI's Explore view.

Diagrams are cached in memory. Re-indexing (⟳) invalidates the cache.

---

## Semantic Map (`backend/services/map_service.py`)

This is the most visually striking feature and also a genuine learning tool about how embeddings work.

**What it shows:** Every indexed chunk projected onto a 2D plane. Chunks that are semantically similar cluster together. You can see at a glance that "attention is all you need" — all the attention-related functions cluster in one region, all the training utilities in another.

**How it works:**

1. Fetch all chunk embeddings for a repo from Qdrant (with their vectors)
2. Project from 1024 dimensions → 2 dimensions using UMAP (`umap-learn`)
3. If UMAP is unavailable, fall back to PCA
4. K-means clustering in 2D space (2–8 clusters based on corpus size)
5. Label each cluster with the most common directory prefix

**Why UMAP over PCA:** UMAP preserves local neighbourhood structure — chunks that were close in 1024-dim space stay close in 2D. PCA preserves global variance, which tends to spread everything out and loses the local clustering.

**Query overlay:** UMAP supports `transform()` — projecting new points onto an existing layout. After a query, we embed the question and project it onto the existing 2D map. The dot showing where your question lands tells you which semantic region of the codebase the system searched. This makes the retrieval step visible.

---

## Full Quality Improvement Stack

Every layer in this stack independently improves the quality of the final answer. The layers compound.

| Layer | Technique | Why It Helps |
|-------|-----------|--------------|
| Chunking | AST at function/class boundaries | Semantic units, never arbitrary mid-function splits |
| Chunking | Contextual retrieval (re-index) | LLM descriptions embed meaning, not just syntax |
| Embedding | Voyage AI `voyage-code-3` | Code-optimised vectors, trained on code corpora |
| Embedding | Deduplication via SHA-256 | No redundant API calls, faster re-indexing |
| Search | Hybrid BM25 + semantic | Exact identifiers (BM25) + conceptual queries (dense) |
| Search | HyDE | Hypothetical code lives closer to real code in embedding space |
| Search | Query expansion + RRF | Multiple phrasings catch different chunks |
| Reranking | Cohere Rerank | Cross-attention sees (query, chunk) together — more accurate |
| Generation | Provider cascade | Always finds a working free LLM |
| Generation | Conditional temperature | Technical queries → low temperature; creative → higher |
| Generation | Faithfulness grading | Confidence score exposed to user |
| Agent | Multi-step MCP tool use | Search → read → search → read builds deeper understanding |

---

## UI Design

Premium dark violet aesthetic, inspired by litellm.ai.

**Design system:**
- Background: `#05021A` (deep violet-black)
- Primary accent: `#7C3AED` violet; secondary: `#C4B5FD` lavender
- Cards: 4-layer shadow + inset white glow `rgba(255,255,255,0.18) 0px 0px 20px 0px inset`
- Typography: Space Grotesk (UI text) + JetBrains Mono (code)
- Dot grid background pattern
- Ambient blob animation on empty chat state

**Pipeline visibility features:**

Every AI system has internal state that is usually hidden from users. This UI makes it visible:

- **Pipeline provenance bar**: `hybrid search → re-ranked → 8 sources → generated → high confidence` — shows the path every answer took through the system
- **Agent trace panel**: each tool call is expandable, showing what was searched and what was returned
- **Contextual retrieval badge**: a `✦` icon next to repos that have been re-indexed with LLM-generated descriptions
- **Semantic map**: UMAP projection of all chunks with query overlay
- **MCP server panel**: live counts of registered tools, resources, and prompts

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values.

```bash
# Embeddings — pick one (Voyage is preferred for code)
VOYAGE_API_KEY=         # https://dash.voyageai.com — free, 200M tokens/month
NOMIC_API_KEY=          # https://atlas.nomic.ai — fallback, 768-dim

# Reranking — optional but strongly recommended
COHERE_API_KEY=         # https://dashboard.cohere.com — free, 1000 reranks/month

# LLMs — at least one required; cascade tries them in order
GROQ_API_KEY=           # https://console.groq.com — free, 14,400 req/day
GEMINI_API_KEY=         # https://aistudio.google.com — free, 1,500 req/day
OPENROUTER_API_KEY=     # https://openrouter.ai — free tier
ANTHROPIC_API_KEY=      # Optional paid fallback

# Vector DB — required
QDRANT_URL=             # https://cloud.qdrant.io — free cluster, 1GB
QDRANT_API_KEY=         # From Qdrant Cloud dashboard
QDRANT_COLLECTION=      # e.g. github_repos_voyage
                        # Change this when switching embedding models — dimensions must match

# GitHub — optional but recommended
GITHUB_TOKEN=           # Raises rate limit: 60 → 5000 req/hr

# Deployment
FRONTEND_URL=           # Your Vercel URL — required for CORS

# Tuning — optional, defaults shown
EMBEDDING_DIM=1024      # Must match model: voyage-code-3=1024, nomic=768
USE_HYDE=true           # Enable HyDE query enhancement
EXPAND_QUERIES=true     # Enable query expansion
TOP_K=8                 # Chunks returned to LLM after reranking
```

> **Note on `QDRANT_COLLECTION`:** The collection name encodes your embedding model choice. If you switch from Voyage (1024-dim) to Nomic (768-dim), you must use a different collection name. Qdrant will error if you try to upsert 768-dim vectors into a 1024-dim collection.

---

## Setup

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cartographer
cd cartographer

# 2. Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install backend dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Start the backend
uvicorn backend.main:app --reload
# API runs at http://localhost:8000
# OpenAPI docs at http://localhost:8000/docs

# 6. Start the frontend (separate terminal)
cd ui
npm install
npm run dev
# UI runs at http://localhost:5173
```

### Verify the Setup

After starting both servers:

1. Open `http://localhost:5173`
2. Enter a GitHub URL in the sidebar (e.g., `https://github.com/karpathy/micrograd`)
3. Click Ingest and wait for the progress bar to complete
4. Ask a question: "How does the backward pass work?"

You should see the pipeline provenance bar fill in: hybrid search → reranked → sources → generated → confidence.

---

## Deployment

### Backend on HuggingFace Spaces (Docker)

The `Dockerfile` in the repo root builds the FastAPI app for HF Spaces.

Key details:
- HF Spaces requires port **7860** — the Dockerfile exposes this
- The local cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is downloaded into the Docker image at build time so there is no cold-start delay when Cohere is unavailable
- All Python dependencies are installed in the image

Steps:
1. Create a new Space at `huggingface.co/new-space`, choose **Docker** SDK
2. Push your repo to the Space's git remote
3. In Space Settings → Variables, add all env vars from your `.env`
4. Set `FRONTEND_URL` to your Vercel deployment URL

### Frontend on Vercel

1. Connect your GitHub repo to Vercel
2. Set the root directory to `ui/`
3. Add environment variable: `VITE_API_URL=https://your-space.hf.space`
4. Deploy

After deploying, go back to your HF Space and set `FRONTEND_URL` to your Vercel URL. This is needed for the CORS policy in `backend/main.py`.

---

## Free Tier Budget

This entire stack runs within free tiers. Here is the math.

| Service | Free Limit | Typical Usage | Notes |
|---------|-----------|--------------|-------|
| Voyage AI | 200M tokens/month | ~50K per repo ingest | 4,000 repo ingests/month |
| Nomic API | ~1,000 req/day | Fallback only | Rarely needed |
| Cohere Rerank | 1,000 calls/month | 1 per RAG query | ~33 queries/day |
| Groq | 14,400 req/day | ~2 per RAG query | ~7,200 RAG queries/day |
| Gemini | 1,500 req/day | ~5 per agent query | ~300 agent queries/day |
| Qdrant Cloud | 1GB storage | ~5MB per 100 repos | Fits ~20,000 repos |
| HF Spaces | Free (CPU) | Backend always-on | Sleeps after 48hr inactivity |
| Vercel | Free | Frontend | No limits for static |

The tightest constraint is Cohere Rerank at 1,000 calls/month. If you hit this, the fallback cross-encoder kicks in automatically — quality drops slightly but the system keeps working.

---

## What You Learn Building This

This project is intentionally a complete tour of modern RAG engineering. Building it from scratch teaches:

1. **RAG fundamentals** — the chunk → embed → store → retrieve → generate pipeline end to end
2. **Hybrid search** — why combining dense vectors (semantic) with sparse BM25 (keyword) outperforms either alone
3. **Reranking** — the difference between bi-encoders (fast, parallel, less accurate) and cross-encoders (slower, joint attention, much more accurate)
4. **HyDE** — how embedding a hypothetical answer instead of the raw question improves semantic search recall
5. **Query expansion** — using LLMs to improve search rather than just to answer
6. **Contextual retrieval** — Anthropic's technique for prepending LLM-generated descriptions before embedding
7. **Agentic RAG** — when single-shot RAG is insufficient and how multi-step tool use overcomes it
8. **MCP** — Model Context Protocol, the standard for connecting LLMs to external tools (the same protocol Claude Code uses)
9. **SSE streaming** — server-sent events for real-time progress bars and token-by-token LLM output
10. **AST parsing** — using tree-sitter to understand code structure rather than treating code as plain text
11. **UMAP/PCA** — dimensionality reduction for visualising high-dimensional embedding spaces
12. **API vs local inference** — when to call an API, when to run a model locally, and the RAM/latency/cost tradeoffs
13. **Free-tier deployment** — HF Spaces Docker + Vercel, keeping everything within free limits without sacrificing capability

---

## Key Design Decisions (and Why)

**No LangChain or LlamaIndex.** These are excellent libraries for production use. They are bad for learning because they hide the interesting parts. Every module in this project is 100–300 lines of readable Python with no framework magic.

**No local LLMs.** Running LLaMA locally requires 16GB+ RAM and a fast GPU. Using API-based LLMs (Groq, Gemini) means the project works on any machine and deploys to free CPU hosting. The free tier limits are generous enough for personal use and demos.

**Qdrant over Pinecone/Weaviate.** Qdrant has native hybrid search (BM25 + dense in one query). Pinecone requires a separate BM25 index and a client-side merge. Qdrant's approach is simpler and faster.

**Voyage AI over OpenAI embeddings.** OpenAI `text-embedding-3-large` costs money. Voyage AI is free (200M tokens/month) and their `voyage-code-3` model is trained specifically on code, which matters for this use case.

**Gemini for the agent, Groq for generation.** Groq is fastest for single-turn generation (the RAG answer). Gemini is better at structured tool-call JSON and has better reasoning over multiple turns — ideal for the agent loop.

**MCP for the agent architecture.** Using MCP instead of a custom tool-calling framework means the agent tooling is compatible with Claude Code, Cursor, and any other MCP-aware client. It is a genuine open standard.

---

## Project Status

See `PLAN.md` for the full build plan and phase completion status.

See `LEARN.md` for a guided reading order through the codebase if you want to understand it concept by concept.
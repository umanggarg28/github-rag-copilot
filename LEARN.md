# GitHub RAG Copilot — Learning Guide

This document grows with the project. Each section is added when the
corresponding feature is built. Read it alongside the code and `notes/` entries.

---

# Table of Contents

1. [What is RAG on Code?](#1-what-is-rag-on-code) ✓
2. [AST-Based Chunking](#2-ast-based-chunking) ✓
3. [Code Embeddings](#3-code-embeddings) ✓
4. [Qdrant — A Hosted Vector Database](#4-qdrant) ✓
5. [Native Hybrid Search in Qdrant](#5-native-hybrid-search) ✓
6. [Generation for Code Queries](#6-generation-for-code-queries) ✓
7. [Agentic RAG and the ReAct Loop](#7-agentic-rag-and-the-react-loop) ✓
8. [MCP — Model Context Protocol](#8-mcp--model-context-protocol) ✓
9. [Multi-Provider LLM with Fallback](#9-multi-provider-llm-with-fallback) ✓
10. [Live Deployment](#10-live-deployment) ← _coming in Phase 4_
11. [Claude Code Features](#11-claude-code-features) ✓
12. [Re-ranking](#12-re-ranking) ✓
13. [Structured JSON Output (json_mode)](#13-structured-json-output) ✓
13. [Prompt Caching](#13-prompt-caching) ✓
14. [Contextual Retrieval](#14-contextual-retrieval) ✓
15. [Accuracy: How Real Tools Read Code](#15-accuracy-how-real-tools-read-code) ✓
    - 11a. CLAUDE.md
    - 11b. Slash Commands
    - 11c. Hooks
    - 11d. Subagents

---

# 1. What is RAG on Code?

## The core idea

RAG stands for Retrieval-Augmented Generation. The pipeline has two phases:

**Ingestion** (done once, or when a repo is added):
```
Source → extract text → chunk → embed → store in vector DB
```

**Query** (done on every question):
```
Question → embed → search vector DB → retrieve relevant chunks → LLM → answer
```

For this project:
```
Ingestion:  GitHub repo → files → code chunks → embed → Qdrant Cloud
Query:      Question → embed → Qdrant hybrid search → retrieved chunks → Groq/Claude → answer with citations
```

## Why does RAG exist?

LLMs are trained on a static snapshot of the internet up to a cutoff date.
They cannot know about:
- Private repositories
- Repos that changed after their training cutoff
- Your specific codebase

Without RAG, asking "how does authentication work in this repo?" forces the
LLM to either guess (hallucinate) or say "I don't know." With RAG, the
relevant code is retrieved and placed directly in the prompt — the LLM reads
your actual code and answers from it.

## What makes code RAG different from document RAG?

Three things change when the source is code instead of documents:

**1. What you index**

A research paper contains only relevant content. A GitHub repo contains:
- Source code you care about (`.py`, `.ts`, `.go`, etc.)
- Auto-generated files you don't care about (`package-lock.json`, `*.lock`)
- Dependency directories with thousands of files (`node_modules/`, `.venv/`)
- Build output (`dist/`, `__pycache__/`)

Indexing everything would drown relevant code in noise. You need a
`file_filter.py` with explicit rules about what to include.

**2. How you chunk**

A research paper can be split anywhere — even mid-sentence, the surrounding
context still makes sense to an LLM. Code is structured differently:

```python
def embed_text(self, text: str) -> list[float]:
    """Embed a single text string into a vector."""
    tokens = self.tokenizer.encode(text)
    return self.model(tokens).pooled_output.tolist()
```

Split mid-function and you lose either:
- The signature (what it accepts, what it returns) — can't answer "what does this take?"
- The body (what it actually does) — can't answer "how does this work?"

Solution: **AST-based chunking** — parse the code into its syntax tree and
split at natural boundaries (functions, classes). Each chunk is a complete,
self-contained unit of code.

**3. What metadata matters**

For documents: `source` (paper name), `page` (page number)

For code: `repo`, `filepath`, `language`, `function_name`, `class_name`,
`start_line`, `end_line`

This makes citations meaningful:
```
Code: torch/nn/functional.py — scaled_dot_product_attention() — lines 4823–4891
```

And enables powerful filters you can't do with documents:
- "Only search in test files"
- "Only search in the `auth/` directory"
- "Only search in Python files"

---

# 2. AST-Based Chunking

## Why not just split on character count?

The simplest approach to chunking text: split every 1000 characters. Works for documents.
For code it fails immediately:

```python
# Split at character 1000 — you get this abomination:
"...    return output

def forw"   # half a function
"ard(self, x):
    x = self.norm(x)..."  # second half
```

An LLM given the second chunk can't answer "what does `forward` accept?" — the signature is cut off.

## AST to the rescue

AST stands for **Abstract Syntax Tree** — the structured representation of code that the language's parser builds. Every function, class, and method is a node in the tree with a precise start and end line.

```
Module
├── ClassDef: Transformer (lines 1–120)
│   ├── FunctionDef: __init__ (lines 4–42)
│   ├── FunctionDef: forward (lines 44–78)
│   └── FunctionDef: generate (lines 80–120)
└── FunctionDef: train (lines 122–200)
```

We use **tree-sitter** — a fast, multi-language parser — to walk this tree and extract each function and class as its own chunk:

```python
# ingestion/chunker.py
parser = Parser(Language(tree_sitter_python.language()))
tree   = parser.parse(source_code.encode())

def _walk(node):
    if node.type in ("function_definition", "class_definition"):
        yield Chunk(
            text       = source_code[node.start_byte : node.end_byte],
            start_line = node.start_point[0] + 1,
            end_line   = node.end_point[0] + 1,
            name       = _get_name(node),
        )
    for child in node.children:
        yield from _walk(child)
```

The result: every chunk is a **complete, self-contained unit** of code — never mid-function, never orphaned.

## What about long functions?

Some functions exceed the embedding model's token limit (~8192 tokens). When that happens, we fall back to a line-window approach: overlap adjacent windows so context bleeds across boundaries. But AST chunking is tried first.

## Why does this matter for retrieval quality?

When you ask "how does `generate` work?", the retrieved chunk is the entire `generate` function — signature, body, return value. The LLM can read it and answer accurately. Without AST chunking, you'd retrieve arbitrary windows that might not contain the signature at all.

See `ingestion/chunker.py` for the full implementation.

---

# 3. Code Embeddings

## What is an embedding?

An embedding is a function that maps text to a fixed-size vector of floats:

```
"def forward(self, x): ..."  →  [0.12, -0.44, 0.93, 0.01, ...]  (768 floats)
```

Semantically similar inputs produce vectors that are close in this 768-dimensional space.
"Cosine similarity" measures how close two vectors are — 1.0 = identical direction, 0.0 = unrelated.

## Why a code-specific model?

General text models (trained on Wikipedia, books, news) learn that "forward" relates to movement.
Code models know that `forward()` in a neural network context relates to `__call__`, `backward()`, and gradient computation.

We use **`nomic-embed-code`** — a 768-dim model fine-tuned on code. It understands:
- Function names and their relationship to bodies
- Import dependencies
- Variable naming patterns specific to ML frameworks

```python
# ingestion/embedder.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-code")
vector = model.encode("def forward(self, x): ...")  # → list[float] of length 768
```

## Query vs document embeddings

Many embedding models use different prefixes for queries vs documents:

```python
# Document (chunk being indexed):
"search_document: def backward(self):\n    ..."

# Query (question being asked):
"search_query: how does the backward pass work?"
```

This asymmetry teaches the model that a short question and a long function body can be "related" even though they look nothing alike syntactically.

## Batch embedding is 10x faster

Embedding one text at a time is slow. `encode_batch()` processes many texts in one GPU pass:

```python
# Slow:
vectors = [model.encode(chunk.text) for chunk in chunks]

# Fast (10x+):
vectors = model.encode_batch([chunk.text for chunk in chunks])
```

See `ingestion/embedder.py` for the implementation.

---

# 4. Qdrant — A Hosted Vector Database

## Why not just use a list?

After embedding 10,000 code chunks, each as a 768-float vector, finding the closest one to a query requires computing cosine similarity against all 10,000. That's 10,000 × 768 multiplications for every search — about 7.68 million operations. Slow for large repos.

Qdrant uses **HNSW** (Hierarchical Navigable Small World) — a graph data structure that enables approximate nearest-neighbour search in O(log N) time instead of O(N). It pre-builds a graph where each vector is connected to its neighbours. Searching traverses the graph instead of scanning everything.

## Why Qdrant specifically?

- **Free hosted tier** (Qdrant Cloud): 1GB free, enough for hundreds of repos
- **Native hybrid search**: stores dense + sparse vectors per point, fuses them server-side
- **No separate infrastructure**: just an API key, no Docker, no self-hosting

## Collections, Points, Payloads

Qdrant organises data as:

```
Collection ("code_chunks")
└── Point
    ├── id:      "abc123"                        ← unique string or UUID
    ├── vectors:
    │   ├── "code":  [0.12, -0.44, ...]          ← dense (768 floats)
    │   └── "bm25":  {42314: 1.0, 8821: 2.0}     ← sparse (dict of index→value)
    └── payload: {                                ← arbitrary JSON
            "repo": "karpathy/micrograd",
            "filepath": "engine.py",
            "start_line": 45,
            "end_line": 72,
            "chunk_type": "function",
            "name": "backward",
            "text": "def backward(self): ..."
        }
```

The `payload` is filterable — you can search only within `{"repo": "karpathy/micrograd"}` or only `{"language": "python"}` without a full scan.

## Hash collision fix

Sparse vectors require **unique indices**. When we hash token strings to integer positions using `MD5 % 2^20`, two different tokens can collide to the same index. Qdrant rejects vectors with duplicate indices with a 422 error.

The fix: use a dict and **sum values** for colliding indices:

```python
# Wrong: appends duplicates
indices = [hash(tok) % 2**20 for tok in tokens]  # may have duplicates!

# Right: accumulate into a dict first
index_map: dict[int, float] = {}
for token, count in token_counts.items():
    idx = int(md5(token).hexdigest()[:8], 16) % (2 ** 20)
    index_map[idx] = index_map.get(idx, 0.0) + float(count)
```

See `ingestion/qdrant_store.py:_text_to_sparse()`.

---

# 5. Native Hybrid Search in Qdrant

## What is hybrid search?

A single query can be interpreted two ways:
- **Semantic** ("what does this *mean*?") — find code that's conceptually related
- **Keyword** ("does this *exact string* appear?") — find code with this identifier

A pure semantic search misses exact function names. A pure keyword search misses
related-but-differently-named code. Hybrid search runs both and fuses the results.

## How it works in Qdrant

Every chunk is stored with **two** vectors:

```
Dense vector (768 floats):  [0.12, -0.44, 0.93, ...]   ← semantic meaning
Sparse vector (indices+values): {42314: 1.0, 8821: 2.0}  ← BM25 term frequencies
```

When you search, Qdrant runs both in one request and fuses results with **RRF**
(Reciprocal Rank Fusion):

```python
results = client.query_points(
    collection_name=self.collection,
    prefetch=[
        Prefetch(query=dense_vector,  using="code", limit=top_k * 2),  # semantic
        Prefetch(query=sparse_vector, using="bm25", limit=top_k * 2),  # keyword
    ],
    query=FusionQuery(fusion=Fusion.RRF),   # fuse on server
    limit=top_k,
)
```

The `Prefetch` fetches `top_k * 2` from each system before fusion — a result
ranked 6th semantically and 6th by keyword would rank 1st by RRF, but would be
missed if you only fetched `top_k`.

## What is RRF (Reciprocal Rank Fusion)?

RRF converts raw scores (which are in incompatible units — cosine similarity vs
BM25) into ranks, then combines them:

```
RRF score = 1/(60 + rank_semantic) + 1/(60 + rank_keyword)
```

The constant 60 dampens the effect of very high ranks — position 1 isn't 60x
better than position 2. The final ranking puts highest-RRF-score first.

**Example:**
- Result A: rank 3 semantic, rank 1 keyword → 1/63 + 1/61 ≈ 0.032
- Result B: rank 1 semantic, rank 8 keyword → 1/61 + 1/68 ≈ 0.031

Both ranked highly in at least one system — RRF surfaces them both.

## Sparse vectors: how BM25 terms become vectors

BM25 is a keyword ranking function. We convert text to a sparse vector by:

```python
tokens = re.findall(r"[a-zA-Z_]\w*", text.lower())
token_counts = Counter(tokens)           # {"embed": 2, "text": 1, ...}
indices = [abs(hash(token)) % 2**20 for token in token_counts]  # integer IDs
values  = [float(count) for count in token_counts.values()]
```

Qdrant applies IDF weighting (how rare is this term across all documents?)
at query time — the BM25 "magic" happens on the server.

Why 2^20 dimensions? Sparse vectors can have any number of dimensions. We use
2^20 = 1 million possible positions to reduce hash collisions without
wasting memory (only non-zero positions are stored).

See `ingestion/qdrant_store.py:_text_to_sparse()` for the full implementation.

---

# 6. Generation for Code Queries

## The full RAG pipeline

```
Question
   ↓  (classify: technical or creative?)
   ↓  (embed question → dense + sparse vectors)
   ↓  Qdrant hybrid search → top-K chunks
   ↓  format_context() → numbered source list
   ↓  LLM (Groq / Anthropic)
Answer with source citations
```

## Conditional LLM parameters

Not all code questions need the same answer style:

| Query type | Example | Temperature | Max tokens |
|------------|---------|-------------|------------|
| Technical  | "trace the backward pass of ReLU" | 0.1 (precise) | 1024 |
| Creative   | "explain intuitively what a Value node is" | 0.7 (expressive) | 1536 |

We detect query type with weighted keyword signals:

```python
_CREATIVE_SIGNALS  = {"explain": 1, "intuitively": 3, "analogy": 3, "eli5": 3, ...}
_TECHNICAL_SIGNALS = {"implement": 2, "trace": 2, "formula": 3, "algorithm": 2, ...}

creative_score  = sum(w for s, w in _CREATIVE_SIGNALS.items()  if s in question)
technical_score = sum(w for s, w in _TECHNICAL_SIGNALS.items() if s in question)
# Technical wins ties — better to be precise than creatively wrong
return "creative" if creative_score > technical_score else "technical"
```

**Why weighted signals instead of a classifier?**
A classifier needs training data. Weighted keyword matching is instant,
interpretable ("why did it choose creative? because 'intuitively' scored 3"),
and tunable without retraining.

## System prompt structure

The LLM is given one of two system prompts:

```
Base: "You are a code assistant. Answer from context only. Cite sources by number."

Technical adds: "Be precise. Show exact signatures and return values."
Creative adds:  "Explain clearly. Use analogies where they help."
```

This framing changes how the model responds without changing the model itself —
the same 70B Llama can write a textbook explanation or a concise technical doc.

## Why two LLM providers?

- **Groq** (primary): `llama-3.3-70b-versatile`, free tier, very fast (~100 tok/s)
- **Anthropic** (fallback): `claude-haiku-4-5`, if no Groq key is set

In `GenerationService.__init__()`, we check which key is available and set
`self.provider`. All subsequent calls go through the same `answer()`/`stream()`
interface — the router doesn't care which provider runs under the hood.

## Streaming with Server-Sent Events (SSE)

For the `/query/stream` endpoint, we use SSE to push tokens as they arrive:

```
Browser → GET /query/stream?question=...
Server  → text/event-stream
          data: The\n\n
          data: Value\n\n
          data:  class\n\n
          ...
          data: [DONE]\n\n
```

Each `data: ...\n\n` is one SSE event. The browser's `EventSource` API splits
on the double newline and fires `onmessage` per event.

**Important:** literal newlines inside a token would break the SSE format
(the parser would treat it as an event boundary). We escape them:
```python
safe_token = token.replace("\n", "\\n")
yield f"data: {safe_token}\n\n"
```

The frontend then unescapes `\\n` back to `\n` before rendering.

See `backend/main.py:query_stream()` and `backend/services/generation.py`.

## FastAPI architecture

```
main.py                  ← lifespan (startup), routes, CORS, SSE
  ↓ Depends()
backend/services/
  ingestion_service.py   ← pipeline orchestrator (fetch → filter → chunk → embed → store)
  generation.py          ← LLM wrapper, query classifier, streaming
retrieval/retrieval.py   ← Qdrant hybrid/semantic/keyword search
backend/models/schemas.py ← Pydantic request/response models (auto-docs at /docs)
```

**Lifespan** (vs old @on_event):
```python
@asynccontextmanager
async def lifespan(app):
    # STARTUP: load models once
    _ingestion_service = IngestionService()   # loads 600MB embedding model
    _retrieval_service = RetrievalService()
    _generation_service = GenerationService()
    yield
    # SHUTDOWN: cleanup if needed
```

Models load once and are shared via `Depends()` — not reloaded per request.

---

# 7. Agentic RAG and the ReAct Loop

## What's wrong with plain RAG?

Standard RAG runs exactly one search:

```
Question → embed → search → retrieve 5 chunks → LLM → answer
```

This works if the answer is in those 5 chunks. But complex questions require reasoning across multiple searches:

> "How does the attention mechanism in nanochat connect to the training loop?"

To answer this, you need to:
1. Find the attention implementation
2. Find the training loop
3. Understand how they interact

A single search might find attention OR the training loop, but not both at the level of detail needed to explain their connection.

## The ReAct loop

**ReAct** (Reason + Act) is a pattern where the LLM alternates between:
- **Thinking** — deciding what information it needs
- **Acting** — calling a tool to get that information
- **Observing** — reading the result and updating its understanding

```
Question: "How does nanochat's attention connect to training?"

Iteration 1:
  Think: "I need to find the attention implementation"
  Act:   search_code("attention mechanism implementation")
  Observe: Found MultiHeadAttention class in model.py lines 45–82

Iteration 2:
  Think: "Now I need the training loop"
  Act:   search_code("training loop optimizer step")
  Observe: Found train() in train.py lines 110–190

Iteration 3:
  Think: "I need to see how train() calls the model"
  Act:   get_file_chunk("train.py", start=110, end=190)
  Observe: model(x) called inside train loop, gradients computed via loss.backward()

Iteration 4:
  Think: "I have enough. The attention is inside the model which is called in train loop."
  Answer: (full explanation with citations)
```

Each iteration narrows the search until the agent has enough evidence.

## Why this is better

| Plain RAG | Agentic RAG |
|-----------|-------------|
| 1 fixed search | Up to 8 adaptive searches |
| Retrieves what embedding says is relevant | Retrieves what the LLM decides it needs |
| Can miss multi-hop dependencies | Follows the chain: attention → model → training |
| Fast (1 LLM call) | Slower but more accurate |

The tradeoff: agent mode uses 3–9 LLM calls per question. With free-tier providers, this burns through daily quotas faster.

## Tools available to the agent

The agent has three tools exposed via MCP:

| Tool | What it does |
|------|--------------|
| `search_code` | Semantic + keyword search, returns matching chunks |
| `get_file_chunk` | Fetch specific lines from a file (for context expansion) |
| `find_callers` | Find all places that call a given function |

These are defined in `backend/mcp_server.py` and discovered dynamically at runtime — the agent doesn't hardcode them.

## Streaming agent progress

Unlike plain RAG (which streams tokens), the agent also streams its **reasoning steps** as SSE events:

```
event: tool_call    {"tool": "search_code", "input": {"query": "attention"}}
event: tool_result  {"tool": "search_code", "output": "Found: model.py:45..."}
event: tool_call    {"tool": "get_file_chunk", ...}
event: tool_result  ...
data: The attention mechanism...   ← final answer tokens
data: [DONE]
```

This lets the UI show "Searching... Found in model.py... Searching again..." while the agent reasons — the user sees progress instead of a loading spinner.

See `backend/services/agent.py` for the ReAct loop, `backend/main.py:agent_stream` for SSE serialisation.

---

# 8. MCP — Model Context Protocol

## What is MCP?

MCP (Model Context Protocol) is an open standard for connecting LLMs to tools. Think of it as HTTP but for AI tools: a client/server protocol where:
- The **server** exposes tools, resources, and prompts
- The **client** discovers and calls them

Before MCP, every agent hardcoded its tools:
```python
# Old way — tools are baked into the agent
tools = [
    {"name": "search_code", "description": "...", "parameters": {...}},
    {"name": "get_file_chunk", ...},
]
```

With MCP, tools are discovered at runtime:
```python
# New way — agent asks "what tools do you have?"
mcp_tools = await mcp_client.list_tools()   # → [Tool(name="search_code", ...), ...]
```

## Why MCP matters

**Without MCP:** Add a new tool → edit the agent file, redeploy.

**With MCP:**
1. Add the tool to `mcp_server.py` with the `@mcp.tool()` decorator
2. Restart the server
3. The agent automatically discovers it on the next call — no agent code changes needed

This is exactly how Claude Code works: it connects to MCP servers (filesystem, git, bash) and uses whatever tools they expose, without any of those tools being hardcoded.

## Our MCP setup

```
backend/
  mcp_server.py   ← FastMCP server — defines tools with @mcp.tool()
  mcp_client.py   ← MCPClient — connects to the server, lists/calls tools
  services/
    agent.py      ← AgentService — uses MCPClient to call tools via protocol
```

The MCP server is mounted as a sub-app inside FastAPI:

```python
# main.py
app.mount("/mcp", mcp.streamable_http_app())
```

So one process speaks two protocols:
- **REST API** — `POST /ingest`, `GET /query/stream`, etc.
- **MCP** — `POST /mcp` (JSON-RPC 2.0 over HTTP)

## Tool definition vs tool call

Defining a tool in the server:
```python
# mcp_server.py
@mcp.tool()
def search_code(query: str, repo: str | None = None, top_k: int = 5) -> str:
    """Search for code by semantic meaning and keywords."""
    results = _retrieval_service.search(query=query, repo_filter=repo, top_k=top_k)
    return format_results(results)
```

The LLM sees a JSON schema derived from the function signature. When it decides to call the tool, it generates:
```json
{"name": "search_code", "arguments": {"query": "attention mechanism", "top_k": 5}}
```

The agent routes this through MCPClient → MCP server → `search_code()` → returns result.

## MCP prompts

MCP also supports **prompt templates** — pre-written instructions the user can insert with a slash command. In the UI, typing `/` opens an autocomplete showing available prompts:

```
/analyze_repo  — full architecture overview of the indexed repo
/explain_function — deep-dive into a specific function
/trace_call    — follow a function call through the codebase
```

These are defined in `mcp_server.py` with `@mcp.prompt()`.

---

# 9. Multi-Provider LLM with Fallback

## The problem with a single provider

Free LLM tiers have limits:
- **Groq** (Llama 3.3 70B): ~14,400 tokens/day, resets midnight UTC
- **Gemini** (gemini-2.0-flash): 1,500 requests/day AND 15 requests/minute via AI Studio free tier
- **Anthropic** (Claude Haiku): paid per token, no free tier

Agent mode is token-hungry — up to 9 LLM calls per question (8 reasoning iterations + 1 final streaming answer). At ~2000 tokens/call, one session can burn through 18,000 tokens.

## Priority order

We use different priority orders for the two services:

| Service | Priority | Why |
|---------|----------|-----|
| `generation.py` | Groq → Gemini → Anthropic | Groq is fastest, no tool calls so no hermes bug |
| `agent.py` | Gemini → Groq → Anthropic | Groq's Llama 3.3 emits broken `<function=name{...}>` format for tool calls; Gemini uses proper JSON |

## The hermes bug

Groq's Llama 3.3 model intermittently generates tool calls in "hermes format":
```
<function=search_code{"query": "attention"}></function>
```

Instead of the proper OpenAI JSON format that Groq's own API expects. Groq then rejects its own output with a 400 error. This is a known Groq/Llama issue.

Fix: use Gemini first for the agent (Gemini always outputs correct JSON), and only fall back to Groq if Gemini is unavailable.

## Runtime fallback

Both services implement `_try_fallback()` — called when the current provider throws a quota/rate-limit error:

```python
def _try_fallback(self) -> bool:
    """Switch to the next provider in the priority chain. Returns True if switched."""
    if self._provider == "gemini" and settings.groq_api_key:
        # Re-init client to Groq
        ...
        return True
    if self._provider in ("gemini", "groq") and settings.anthropic_api_key:
        # Re-init client to Anthropic
        ...
        return True
    return False
```

`_is_exhausted()` detects the error strings that indicate a quota/rate-limit rather than a real API error:
```python
def _is_exhausted(e: Exception) -> bool:
    msg = str(e).lower()
    return any(kw in msg for kw in (
        "credit", "billing", "quota", "rate_limit", "rate limit",
        "resource_exhausted", "429", "daily limit",
    ))
```

## The Gemini OpenAI-compatible endpoint

Gemini offers an OpenAI-compatible API — same request/response format as OpenAI/Groq. This means Gemini reuses all the Groq code paths with zero changes except the client init:

```python
from openai import OpenAI

client = OpenAI(
    api_key=settings.gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# Now use exactly like OpenAI/Groq
response = client.chat.completions.create(model="gemini-2.0-flash", ...)
```

Get a free Gemini key at: [aistudio.google.com](https://aistudio.google.com)

---

# 10. Live Deployment

← _coming in Phase 4_

---

# 11. Claude Code Features

This section explains how to use Claude Code effectively while building this
project — it's a meta-layer that makes you faster and more consistent.

## 11a. CLAUDE.md

`CLAUDE.md` is a file at the project root that Claude Code reads at the start
of every session. It's a permanent briefing document.

**Without CLAUDE.md:** Claude rediscovers the project structure on every
session by reading files and asking questions — costing time and tokens.

**With CLAUDE.md:** Claude immediately knows the architecture, conventions,
environment variables, and key design decisions. It can start working without
a warm-up phase.

**What to put in CLAUDE.md:**
```markdown
- Directory structure (quick mental map)
- How to run the project
- Environment variables required
- Coding conventions (e.g. "no LangChain", "write docstrings")
- Key decisions and WHY they were made
- Available slash commands
```

**What NOT to put in CLAUDE.md:**
- Detailed implementation docs (put those in code comments)
- Anything that changes frequently (stale info wastes tokens)
- Things Claude can infer from reading the code

**Token cost:** Claude reads CLAUDE.md on every message. Keep it concise —
every unnecessary line is paid for in tokens across hundreds of messages.

Open `CLAUDE.md` at the project root to see this project's example.

## 11b. Slash Commands

Slash commands are custom prompts stored in `.claude/commands/*.md`.
Invoke them with `/command-name [args]`.

```
.claude/commands/
    ingest-repo.md    →  /ingest-repo https://github.com/owner/repo
    search-code.md    →  /search-code "how does auth work"
    add-to-notes.md   →  /add-to-notes
```

Each file is a markdown prompt. `$ARGUMENTS` is replaced with whatever
you type after the command name.

**Why slash commands?**

Instead of typing:
> "Run the ingestion pipeline for this GitHub URL, filter files by language,
> chunk with AST, embed, upsert to Qdrant, then print a summary of files
> indexed, languages detected, total chunks stored, and any files skipped."

You type: `/ingest-repo https://github.com/owner/repo`

The command encodes the full instruction once. Every subsequent invocation
is consistent — you don't accidentally forget to ask for the summary, or
phrase the instruction slightly differently each time.

**When to create a slash command:**
- Any operation you do more than twice
- Multi-step workflows that need to be consistent
- Instructions that are long or easy to get wrong

## 11c. Hooks

Hooks are shell commands that run automatically in response to Claude Code
events. They're defined in `.claude/settings.json`.

**Hook events:**
- `PreToolUse` — before Claude calls a tool (e.g. before writing a file)
- `PostToolUse` — after Claude calls a tool (e.g. after editing a file)
- `Stop` — when Claude finishes responding

**Example: auto-lint after every Python file edit**
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "ruff check $CLAUDE_FILE_PATH --fix --quiet"
      }]
    }]
  }
}
```

Now every file Claude edits is automatically linted and auto-fixed. You
never commit code with style errors, and you don't have to remember to run
the linter.

**Example: remind Claude to update notes after a commit**
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "echo '$CLAUDE_TOOL_INPUT' | grep -q 'git commit' && echo 'REMINDER: run /add-to-notes' || true"
      }]
    }]
  }
}
```

**Why hooks matter:**
They enforce quality gates without relying on Claude (or you) to remember.
A lint hook means the gate is always applied — it's not optional, it doesn't
get skipped when you're in a hurry. Think of them as `pre-commit` hooks but
triggered by Claude's actions rather than git commands.

## 11d. Subagents

Subagents are Claude instances spawned to handle independent tasks in parallel
or in isolation. In Claude Code, you use the `Agent` tool.

**Why subagents?**

Two reasons:

1. **Parallelism** — independent tasks run simultaneously instead of sequentially.
   Ingesting three repos takes 2 minutes instead of 6.

2. **Context isolation** — a subagent does its work in a separate context window.
   If you ask a subagent to explore a large repo, it reads thousands of lines
   without filling the main context. Only the summary comes back.

**Subagent types:**
- `general-purpose` — full tool access, can read/write/run commands
- `Explore` — read-only, optimised for fast codebase exploration
- `Plan` — architecture and design, returns structured plans

**Patterns in this project:**

*Parallel ingestion:*
```
Main agent spawns:
  ├── Subagent A: ingest pytorch/pytorch
  ├── Subagent B: ingest huggingface/transformers
  └── Subagent C: ingest openai/openai-python
All three run simultaneously → done in ~2 min instead of ~6 min
```

*Pre-PR review:*
```
Before committing, spawn an expert subagent:
  "Review the ingestion pipeline for correctness, edge cases, and clarity.
   Rate it and list specific issues."
Main agent gets back a structured review without reading all the files again.
```

*Repo exploration before indexing:*
```
Before ingesting an unfamiliar repo, spawn an Explore subagent:
  "Read the repo structure and README. What are the main modules?
   What languages are used? What should be filtered out?"
Main agent uses the summary to configure ingestion — no need to read everything.
```

---

_Section 10 (Live Deployment) will be added in Phase 4._

---

# 12. Re-ranking

## Why retrieval alone isn't enough

Hybrid search (BM25 + semantic) is excellent at **recall** — it finds most
relevant chunks somewhere in the top-20 results. But its **precision** is weak:
it ranks results by scores computed *independently* for the query and each chunk.

The embedding model encodes the query into one vector and each chunk into another.
Cosine similarity measures how close those vectors are in space. This is fast —
but it misses interactions between specific words in the query and specific words
in the chunk.

Example:
```
Query:     "how does the optimizer update weights?"
Chunk A:   "optimizer.step() — updates all parameters using gradient descent"
Chunk B:   "step() — moves the simulation forward by one time unit"
```
Both chunks contain "step()" and have similar embedding dimensions. The cosine
similarity for both might be nearly identical. The retriever can't distinguish them.

## The cross-encoder solution

A **cross-encoder** reads the query and the chunk **together** in one forward pass:

```
Input:  "[CLS] how does the optimizer update weights? [SEP] optimizer.step()
         — updates all parameters using gradient descent [SEP]"
Output: relevance score = 8.7
```

Because it sees both at once, it can attend to specific query words (optimizer,
update, weights) against specific chunk words (optimizer, parameters, gradient)
and produce a far more accurate relevance score.

The tradeoff: it must process every (query, chunk) pair separately — you can't
pre-compute chunk representations. This makes it too slow for full-index search,
but fast enough for re-ranking 20 candidates (~100ms on CPU).

## Two-stage retrieval

This is the standard production RAG pattern:

```
Stage 1 — Recall (fast, coarse)
  Query → Qdrant hybrid search → top 18 candidates
  (BM25 + semantic, O(log N), all computed offline)

Stage 2 — Precision (slower, precise)
  (query, candidate_1) → cross-encoder → score 6.2
  (query, candidate_2) → cross-encoder → score 9.1
  ...
  (query, candidate_18) → cross-encoder → score 3.4
  → sort by score → return top 6

Final: 6 precisely relevant chunks → LLM → answer
```

The wider candidate pool in Stage 1 (top_k × 3 = 18) matters: re-ranking can
only surface results that were retrieved. If the 7th-most-relevant chunk by
cosine similarity is actually the best answer, the re-ranker can promote it —
but only if it's in the candidate pool.

## The model: ms-marco-MiniLM-L-6-v2

```python
from sentence_transformers.cross_encoder import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([(query, chunk_text) for chunk_text in candidates])
```

- **80MB** — small, runs on CPU, no GPU needed
- **Trained on MS MARCO** — 120M human-judged (query, passage) relevance pairs
- **Part of sentence-transformers** — no new dependency (we already use it for embeddings)
- **Lazy loaded** — downloads once on first search, stays in RAM after

## Bi-encoder vs cross-encoder

| | Bi-encoder (our embedder) | Cross-encoder (re-ranker) |
|---|---|---|
| How | Encodes query + doc separately | Encodes query + doc together |
| Speed | Fast — docs pre-computed offline | Slow — must run per (query, doc) pair |
| Use case | Full-index search (Stage 1) | Re-ranking small candidate set (Stage 2) |
| Accuracy | Good | Much better |

Both models live in `sentence-transformers`. The embedder is a bi-encoder
(produces independent vectors). The re-ranker is a cross-encoder (produces
a relevance score for each pair).

## Where to find it

- `retrieval/retrieval.py:Reranker` — cross-encoder wrapper with lazy loading
- `retrieval/retrieval.py:RetrievalService.search()` — two-stage pipeline
- `backend/main.py:lifespan()` — shared `Reranker` instance created at startup

---

# 16. Model-Based Grading

## The problem: silent hallucination

A RAG answer can sound confident while being wrong. The LLM reads retrieved
code chunks and answers — but it can:
- Make a claim slightly beyond what the sources actually say
- Conflate two similar functions from different files
- Answer from training memory rather than the retrieved context

Without grading, you have no signal. Every answer looks equally authoritative.

## What model-based grading is

After generating the answer, a second LLM call reads the question, the retrieved
sources, and the full answer, then returns a structured verdict:

```json
{"confidence": "high", "faithful": true, "note": "All claims confirmed in sources"}
```

The grader asks: *"Can every claim in this answer be traced back to one of the
retrieved sources?"* If yes → high. If mostly yes → medium. If the answer
contains claims not in the sources → low.

This is called **faithfulness evaluation** in the RAG literature — the same
concept tested in frameworks like RAGAS (RAG Assessment).

## Code-based vs model-based grading

The Anthropic course distinguishes two grading approaches:

| | Code-based grading | Model-based grading |
|---|---|---|
| How | Rule checks (does answer cite a source?) | LLM reads and evaluates |
| Strengths | Fast, deterministic, free | Handles nuance, catches subtle hallucinations |
| Weaknesses | Misses semantic errors | Uses LLM calls, slight latency |
| Example | Check answer contains "Source N" | Check answer is supported by source text |

We use model-based because code-based checks (citation format, length) can't
detect a citation that's present but used incorrectly.

## Free-tier implementation

The grader is designed to stay within free-tier limits:
- Context truncated to 2000 chars (enough to spot hallucinations)
- Output capped at 80 tokens
- Uses the same provider already active (Groq/Gemini/OpenRouter)
- Fires after streaming completes — doesn't affect the streaming UX

```python
# generation.py — grade_answer()
raw = self.generate(_GRADE_SYSTEM, _GRADE_PROMPT, temperature=0.0, json_mode=True)
# json_mode forces clean JSON — no parsing fragility
grade = json.loads(raw)
# → {"confidence": "high", "faithful": True, "note": "..."}
```

## Streaming flow with grading

```
POST /query/stream
  ↓
event: meta  → {sources, query_type}
data: token  → "The forward..."     ← user sees answer streaming
data: token  → " pass computes..."
...
event: grade → {"confidence": "high", "note": "..."}   ← ~300ms after last token
data: [DONE]
```

The `event: grade` named event is handled separately by the frontend — it doesn't
mix with the answer token stream. The confidence badge appears under the answer
after grading completes.

## Where to find it

- `backend/services/generation.py:grade_answer()` — grading logic
- `backend/main.py:token_stream()` — buffers answer, fires grader, emits grade event
- `ui/src/api.js:streamQuery()` — `onGrade` callback handles the grade event
- `ui/src/components/Message.jsx:ConfidenceBadge` — renders the badge

---

# 12. Structured JSON Output

## The problem with asking nicely

Most diagram and tour generation calls expect the LLM to return a JSON object.
The naive approach is to include "Return ONLY valid JSON" in the system prompt and hope.
This fails intermittently because LLMs often wrap responses in markdown fences:

```
Here is the JSON you requested:

```json
{
  "nodes": [...]
}
```
```

Our cleanup code (`strip("`"), check for "json" prefix, regex fallback`) had to handle
these cases manually — fragile and easy to get wrong.

## json_mode: a formal contract

OpenAI-compatible providers (Groq, Gemini, OpenRouter) support `response_format`:

```python
client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[...],
    response_format={"type": "json_object"},  # ← the magic parameter
)
```

This doesn't just hint — it **forces** the model's output to be a syntactically valid
JSON object. The model's sampler is constrained to only generate tokens that form valid
JSON. You cannot get markdown fences, preamble text, or a partial object.

## How we use it

`GenerationService.generate()` accepts `json_mode=True`:

```python
# Before — fragile
raw = gen.generate(system, prompt, temperature=0.0)
cleaned = raw.strip().strip("`")
if cleaned.startswith("json"):
    cleaned = cleaned[4:].strip()
tour = json.loads(cleaned)  # still might fail

# After — reliable
raw = gen.generate(system, prompt, temperature=0.0, json_mode=True)
tour = json.loads(raw)  # always valid JSON from OpenAI-compat providers
```

Under the hood, `generate()` passes `response_format={"type": "json_object"}` to
`_groq_complete()`. For Anthropic (which has no equivalent), the system prompt
instruction ("Return ONLY valid JSON") remains, and the `_parse_json()` cleanup
helper handles any edge cases.

## Where to find it

- `backend/services/generation.py:generate()` — accepts `json_mode` param
- `backend/services/generation.py:_groq_complete()` — passes `response_format`
- `backend/services/diagram_service.py:_parse_json()` — shared cleanup helper
- All 4 `gen.generate()` calls in `diagram_service.py` now pass `json_mode=True`

---

# 13. Prompt Caching

## What is prompt caching?

Every LLM call has two cost components:
- **Input tokens**: the text you send in (system prompt + context + question)
- **Output tokens**: the text the model generates back

Input tokens are the expensive part for large contexts. If you call the same
system prompt 100 times, you pay to process those same tokens 100 times.

Prompt caching lets you tell the provider: "compile the KV state for this text
and reuse it." The compiled state is stored on the provider's side for a short
time (typically 5 minutes). Subsequent calls with the same text at the same
position get a **cache hit** — the tokens are not re-processed.

Anthropic's pricing:
- **Cache write**: 25% cheaper than normal input tokens
- **Cache read**: 90% cheaper than normal input tokens (!)

## How KV cache works

Transformers generate each output token by attending over all previous tokens.
This requires computing Key (K) and Value (V) matrices for every input token —
the "KV cache." Computing this is the expensive part of inference.

Prompt caching pre-computes and stores these K/V matrices for a fixed prefix.
On the next call, if the same prefix appears, the stored K/V matrices are loaded
instead of recomputed. The model continues generation from there.

```
Normal call:
  Input: [system: 200 tokens][context: 1000 tokens][question: 20 tokens]
  Cost:  1220 tokens × $X per token = full price

Cached call (second time, same system + context):
  Input: [CACHED: 1200 tokens][question: 20 tokens]
  Cost:  20 tokens × $X (new) + 1200 tokens × $0.1X (cache read) = 90% cheaper
```

## Our implementation

In `_anthropic_complete()` and `_anthropic_stream()`, the system prompt is
passed as a content block with `cache_control`:

```python
system_block = [
    {
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}   # ← tell Anthropic to cache this
    }
]
client.messages.create(
    system=system_block,
    messages=[...],
    ...
)
```

The `cache_control` marker means "cache everything up to and including this block."
On the second call with the same system prompt, Anthropic returns the cached KV state.

## When it activates

Caching only activates when:
1. The cached content is ≥ 1024 tokens (for Claude 3.5+)
2. The same content appears at the same position across calls
3. Within the cache TTL (5 minutes for ephemeral)

Our system prompts are short (~50 tokens), so they won't trigger the minimum.
The real benefit appears when we add large code context to a user message and
mark that cacheable — the same file content processed for chunk 1 is reused for
chunks 2, 3, 4 in the same file (see Contextual Retrieval below).

## Where to find it

`backend/services/generation.py:_anthropic_complete()` — `system_block` with `cache_control`

---

# 14. Contextual Retrieval

## The core problem with raw code chunks

When you embed a function like:

```python
def _relu(x):
    return max(0.0, x)
```

The embedding vector captures: "mathematical comparison, scalar, return statement."
It does NOT capture: "neural network activation function used in forward pass."

When a user asks "how does the network add non-linearity?", the semantic search
finds text about "non-linearity" — but `_relu` only contains `max` and `0.0`.
It might not rank highly, even though it's the exact answer.

## Anthropic's Contextual Retrieval solution

Anthropic's September 2024 paper introduced Contextual Retrieval:
before embedding each chunk, prepend a short LLM-generated context sentence.

```
Before: def _relu(x): return max(0.0, x)

After:
"_relu is the ReLU activation function applied in every layer of the neural
network to introduce non-linearity after the linear transformation.

def _relu(x): return max(0.0, x)"
```

Now the embedding captures "neural network activation, non-linearity, layer" —
much better match for the user's query.

Their finding: this reduces retrieval failure rates by 49% combined with BM25.

## How we implement it (free-tier friendly)

The naive implementation calls an LLM once per chunk. For a 500-chunk repo,
that's 500 LLM calls — this would hit Groq's free-tier rate limit (30 req/min)
in less than a minute.

Our implementation:
1. **Rank chunks by importance** — classes, entry-point files, and files with
   inheritance score highest. These chunks appear in the most answers, so
   improving their embeddings has the most impact.
2. **Only contextualise top 50** — stays within rate limits, takes ~2 minutes.
3. **Graceful fallback** — any chunk that fails keeps its original text.
4. **Only on force re-ingest** — skip on first-time indexing to keep it fast.

```python
# ingestion_service.py — simplified
if force and self._gen is not None:
    chunks = _add_context(chunks, file_dicts, self._gen)
```

## If using Anthropic (with prompt caching)

Anthropic's implementation uses prompt caching to further reduce cost ~50x:
- Group chunks by file
- Mark the full file content as `cache_control: ephemeral`
- For chunks 2, 3, 4 in the same file — the file content is a cache hit
- You only pay full price for the short per-chunk instruction

```python
# The file content is cached after the first chunk in each file.
# Subsequent chunks in the same file get a 90% discount on the document tokens.
messages=[{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": f"<document>\n{file_content}\n</document>",
            "cache_control": {"type": "ephemeral"},   # ← cached after first call
        },
        {
            "type": "text",
            "text": f"<chunk>\n{chunk_text}\n</chunk>\nDescribe this chunk.",
        },
    ],
}]
```

## Where to find it

- `backend/services/ingestion_service.py:_add_context()` — the full implementation
- `backend/services/ingestion_service.py:_chunk_importance()` — ranking function
- Called in `IngestionService.ingest()` step 5b, only on `force=True`

---

# 15. Accuracy: How Real Tools Read Code

## The core accuracy challenge

LLMs are trained to be helpful, which means they'll produce plausible-sounding
output even when they don't have enough information. For a code exploration tool,
this is dangerous: a confidently wrong explanation is worse than "I don't know."

## Three accuracy levels in this project

**Level 1 — RAG chat (most accurate)**
- Retrieves actual chunk text from Qdrant
- Passes real code to the LLM
- LLM reads the code and answers from it
- Accuracy depends on retrieval quality (whether the right chunks are found)

**Level 2 — Architecture/Class diagrams (grounded structure, LLM descriptions)**
- Node/edge structure comes from real AST data: actual imports, actual `class Foo(Bar):` declarations
- Node descriptions are LLM-generated from function names and file paths
- Structure is 100% accurate; descriptions may hallucinate

**Level 3 — Explore tour (most risky)**
- Before this session: only function/class names were sent to the LLM
- The LLM had to guess what `forward()` does from the name alone
- After this session: top 35 chunks include actual source code
- LLM reads real code → descriptions reflect what the code actually does

## How Claude Code and Cursor do it

Tools like Claude Code, Cursor, and GitHub Copilot don't use embeddings for everything.
They use a layered approach:
1. **File tree** — indexed first, gives a map of the codebase
2. **Keyword search** — fast exact-match for identifiers
3. **Semantic search** — for conceptual queries
4. **Full file reads** — for files deemed relevant, they read the whole file
5. **Edit context** — currently open files are always in context

The key insight: they read actual code, not summaries of code. Our tour generation
now does this for the top 35 most important chunks.

## The remaining accuracy gap

Even with real code in the tour prompt, the LLM only sees 35 chunks out of
potentially 500+. For large repos, the remaining 465 chunks are still described
by name only. Future improvement: use the RAG retrieval pipeline to fetch the
most relevant chunks for each concept the LLM is trying to describe.

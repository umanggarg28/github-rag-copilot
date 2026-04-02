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
16. [Agent Tool Architecture — Making the Agent Smarter](#16-agent-tool-architecture--making-the-agent-smarter) ✓
17. [Streaming Agent Thought Steps — Making Reasoning Visible](#17-streaming-agent-thought-steps--making-reasoning-visible) ✓
18. [Reaching 9.5/10 — System Prompt Engineering + Tool Completeness](#18-reaching-9510--system-prompt-engineering--tool-completeness) ✓
19. [Dead Code Hygiene — Keeping the Codebase Clean](#19-dead-code-hygiene--keeping-the-codebase-clean) ✓
20. [Replacing a Third-Party Library with Custom SVG](#20-replacing-a-third-party-library-with-custom-svg) ✓
21. [API Request Size Limits — The 413 Error](#21-api-request-size-limits--the-413-error) ✓
22. [Agent Working Memory — The Note/Recall Pattern](#22-agent-working-memory--the-noterecall-pattern) ✓
23. [Call Chain Tracing — Walking the Execution Graph](#23-call-chain-tracing--walking-the-execution-graph) ✓
24. [Persistent Repo Map — Agent Cross-Session Knowledge](#24-persistent-repo-map--agent-cross-session-knowledge) ✓
25. [Planning Before Acting — Structured Agent Reasoning](#25-planning-before-acting--structured-agent-reasoning) ✓
26. [Parallel Tool Execution — asyncio.gather in the ReAct Loop](#26-parallel-tool-execution--asynciogather-in-the-react-loop) ✓
27. [Streaming Diagram Generation — SSE Progress for Long LLM Calls](#27-streaming-diagram-generation--sse-progress-for-long-llm-calls) ✓
28. [Docker CI/CD Pipeline — GitHub Actions + HuggingFace Spaces + Vercel](#28-docker-cicd-pipeline--github-actions--huggingface-spaces--vercel) ✓

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

---

# 16. Agent Tool Architecture — Making the Agent Smarter

## The problem with vector search alone

Our original agent had three tools: `search_code`, `get_file_chunk`, `find_callers`.

`search_code` uses embeddings — it embeds your query and finds the nearest code chunks
in vector space. This works well for conceptual questions ("how does backpropagation work?")
but has two blind spots:

1. **Exact name lookups**: If you ask "what does `_add_context` do?", the embedding of
   `"_add_context"` may not land close to the function's actual definition vector.
   The function could be missing from the top-10 results despite being *exactly* what we want.

2. **File-level understanding**: Vector search returns *chunks* (individual functions).
   It never returns the whole file. So the agent can't see imports, module docstrings,
   or how all the functions in a file relate to each other.

These are not retrieval quality problems — they're fundamental limitations of the
chunked-embedding approach.

## The solution: two new tools

### `search_symbol(symbol_name, repo=None)`

This is a **structural lookup** — pure Qdrant filter, zero vector math:

```
Qdrant: WHERE name = 'backward' AND repo = 'karpathy/micrograd'
```

It's the equivalent of `grep -r "def backward"` but fast and over the cloud index.

Why does this matter? When the agent has already found a caller:

```
# search_code result says: "backward() is called in engine.py:45"
# Agent now knows the name. Use search_symbol to jump to the definition:
search_symbol("backward", repo="karpathy/micrograd")
# → Returns the exact chunk where backward() is defined, every time.
```

This is how real dev tools work. Cursor and VS Code don't embed "backward" to find
its definition — they use an AST index and do an exact name lookup. We're doing the
same thing now with our Qdrant payload index.

### `read_file(repo, filepath)`

Reads the **entire source file** via GitHub API — same as `get_file_chunk` but no
line range needed. The agent gets:
- All imports (critical for understanding dependencies)
- Module-level docstring
- Every function/class in order
- The full context of how things relate

When to use it:

| Question type | Right tool |
|---|---|
| "What does `backward` do?" | `search_symbol` or `search_code` |
| "What does engine.py import?" | `read_file` |
| "How do all these classes relate?" | `read_file` |
| "What happens on line 140?" | `get_file_chunk` (you know the line) |

The warning: `read_file` on a 2000-line file could use 20k+ tokens of context.
The tool warns the agent when a file is >500 lines so it can choose `get_file_chunk` instead.

## Why the agent gets them for free

Here's the elegant part of the MCP architecture:

```python
# In agent.py — called once when a question arrives:
tools = await self.mcp.list_tools()
# Returns: search_code, find_callers, get_file_chunk, search_symbol, read_file
```

**We added tools to `mcp_server.py`. The agent called zero new code.**

The agent uses `mcp.list_tools()` to discover tools at runtime. Every `@mcp.tool()`
decorator in `mcp_server.py` is automatically published via the MCP protocol.
The agent doesn't know or care how many tools exist — it reads their descriptions
and decides which ones to call.

This is **plug-and-play tooling**. The agent is a general-purpose ReAct loop.
The tools are interchangeable plugins. Want to add a `run_tests` tool? Add it to
`mcp_server.py` with a `@mcp.tool()` decorator and the agent has it immediately.

This is exactly how Claude Code works: you configure MCP servers in settings,
and Claude discovers their tools at startup. Our agent does the same thing —
it just happens to connect to its own MCP server (the one we built).

## The impact

Before: `search_code` → "find similar chunks in vector space"  
After: Agent can now do what senior engineers do:
1. Search for a concept (vector search)
2. Jump to a specific definition (exact name lookup)
3. Read the whole file for full context (file read)
4. Trace all callers (structural graph lookup)

Concrete example — before vs after for "explain how autograd works in micrograd":

**Before (3 search_code calls):**
```
search_code("autograd backward pass") → 6 chunks, maybe missing the Value class def
search_code("Value class") → finds it, but no imports visible
search_code("gradient accumulation") → 6 more chunks
Answer: pretty good but missing some context
```

**After (mixed tool strategy):**
```
search_code("autograd backward pass") → finds backward() chunk, get its file
search_symbol("Value") → exact definition immediately  
read_file("karpathy/micrograd", "micrograd/engine.py") → full file: all methods, imports
find_callers("backward") → see how it's invoked in the training loop
Answer: complete picture, every detail
```

## Technical implementation

`search_symbol` uses a new `find_symbol()` method on `QdrantStore`:

```python
# qdrant_store.py
def find_symbol(self, symbol_name, repo=None):
    conditions = [FieldCondition(key="name", match=MatchValue(value=symbol_name))]
    if repo:
        conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo)))
    # scroll() = paginated full-scan with filter — no vectors needed
    ...
```

`read_file` is a GitHub API call — same pattern as `get_file_chunk` but no slicing:
```python
url = f"https://api.github.com/repos/{owner}/{name}/contents/{filepath}"
headers = {"Accept": "application/vnd.github.v3.raw"}
# Returns the raw file text, numbered line by line
```

Both tools have the same path-traversal protection as `get_file_chunk`: they
reject `../` in paths to prevent an LLM (or a prompt injection attack) from
trying to read files outside the intended repo.

## What's still missing (honest assessment)

Even with these tools, there are quality gaps vs. commercial tools:

| Feature | Ours | Cursor/Claude Code |
|---|---|---|
| Exact symbol lookup | ✅ `search_symbol` | ✅ LSP-based |
| Full file read | ✅ `read_file` | ✅ |
| Type inference | ❌ | ✅ LSP |
| Cross-file call graph | Partial (`find_callers`) | ✅ full LSP |
| Edit current file | ❌ | ✅ |
| Evals / offline scoring | ❌ (next) | ✅ |

The big remaining unlock is an **evals framework**: a set of 50-100 gold
question-answer pairs per repo that we can run offline to score quality.
Right now we judge quality by feel. With evals, we'd know if a change to the
system prompt or retrieval parameters actually improves answers.

---

# 17. Streaming Agent Thought Steps — Making Reasoning Visible

## What the user sees

When you ask a question in agent mode, instead of a blank spinner for 10 seconds,
you see a live timeline:

```
● Agent  ↻
  💭 "I need to find the backward pass implementation first"
  🔍 search_code          "backpropagation gradient computation"     ↻
     Found Value.backward in engine.py, lines 45–67 ...
  ◎ search_symbol         "backward"                                 ✓
     Found 1 definition of 'backward': ...
  📖 read_file            "micrograd/engine.py"                      ✓
     # micrograd/engine.py  (72 lines total) ...
  ⬡ find_callers          "backward"                                 ✓
     Found 3 caller(s): train(), evaluate(), ...
```

Each step collapses when the next one starts. The full trace is expandable after
the answer arrives. The thought bubble shows WHY the agent chose each tool.

## How it works end to end

### 1. Backend: the agent is an async generator

`agent.stream()` in `backend/services/agent.py` uses Python's `yield` keyword
to produce events as they happen:

```python
async def stream(self, question, repo, messages_history):
    # ...
    for iteration in range(MAX_ITERATIONS):
        step = await asyncio.to_thread(self._call_llm, messages, mcp_tools)

        thought = _extract_thought(step["assistant_message"])
        if thought:
            yield {"type": "thought", "text": thought}   # 💭 bubble

        for tc in step["tool_calls"]:
            yield {"type": "tool_call",   "tool": tc["name"], "input": tc["input"]}
            result = await mcp.call_tool(tc["name"], tc["input"])
            yield {"type": "tool_result", "tool": tc["name"], "output": result}

        if step["done"]:
            async for token in self._stream_final_answer(messages):
                yield {"type": "token", "text": token}  # answer tokens
            yield {"type": "sources", ...}
            yield {"type": "done", ...}
            return
```

`yield` suspends the function and delivers the event immediately — the caller
gets each event the instant it's produced, not all at once at the end.

### 2. Backend: SSE transport

FastAPI's `StreamingResponse` wraps the async generator and sends each event
as a Server-Sent Events (SSE) frame:

```
event: thought
data: {"text": "I need to find the backward pass first"}

event: tool_call
data: {"tool": "search_code", "input": {"query": "backpropagation"}}

event: tool_result
data: {"tool": "search_code", "output": "Found Value.backward ..."}

event: message
data: According to the source code...

event: done
data: {"iterations": 4}
```

SSE is just HTTP with `Content-Type: text/event-stream`. The connection stays
open and the server pushes frames. The browser's `fetch` API can read them
chunk-by-chunk via `response.body.getReader()`.

### 3. Frontend: api.js parses the SSE stream

```javascript
// api.js
const reader = response.body.getReader();
while (true) {
    const { value, done } = await reader.read();
    const text = new TextDecoder().decode(value);

    for (const part of text.split("\n\n")) {
        // each part is one SSE event
        if (eventType === "thought")      onThought(text);
        if (eventType === "tool_call")    onToolCall(tool, input);
        if (eventType === "tool_result")  onToolResult(tool, output);
        if (eventType === "token")        onToken(data);
        if (eventType === "done")         onDone(iterations);
    }
}
```

The callbacks (`onThought`, `onToolCall`, etc.) are passed in from `App.jsx`.

### 4. Frontend: App.jsx builds message state

Each callback mutates the in-progress assistant message:

```javascript
onThought:    (text)         => append { type: "thought", text } to toolCalls array
onToolCall:   (tool, input)  => append { tool, input, output: "" } to toolCalls
onToolResult: (tool, output) → find last matching tool call, fill in its output
onToken:      (text)         → append to msg.content (the answer text)
onDone:       (iterations)   → mark msg.streaming = false, save iteration count
```

React re-renders on every state update — so each event triggers a re-render and
the UI updates live.

### 5. Frontend: Message.jsx renders the trace

`ToolCallTrace` maps `msg.toolCalls` to a vertical timeline:
- `{ type: "thought", text }` → `AgentThought` bubble (italic, dimmed)
- `{ tool, input, output }` → `AgentStep` with icon + query label + collapsible output

```jsx
{steps.map((step, i) =>
  step.type === "thought"
    ? <AgentThought text={step.text} />
    : <AgentStep
        step={step}
        icon={toolIcon[step.tool]}      // SVG icon per tool
        isLast={i === lastToolIdx}      // last step = still animating
        streaming={streaming}
      />
)}
```

`isLast && streaming` = the step is currently executing → show spinner, expand output.
Once the next step starts, the previous one collapses automatically.

## The formatStepQuery function

Each tool has a different input shape. Raw JSON in the step header is unreadable:
```
search_symbol   {"symbol_name":"backward","repo":"karpathy/micrograd"}  ← bad
search_symbol   backward                                                 ← good
```

`formatStepQuery(tool, input)` extracts the right field per tool:
```javascript
case "search_code":    return input.query;
case "search_symbol":  return input.symbol_name;
case "find_callers":   return input.function_name;
case "read_file":      return input.filepath;
case "get_file_chunk": return `${input.filepath} (L${input.start_line}–${input.end_line})`;
```

This is a pure function at module level — no state, no hooks. It's called inside
`AgentStep` where both `tool` and `input` are available from `step.tool` and `step.input`.

## What this teaches about async architectures

This feature connects several concepts:

1. **Async generators**: Python `yield` in an `async def` → events flow out as they happen
2. **SSE (Server-Sent Events)**: Persistent HTTP connection for server→client push
3. **React state as a stream**: Each callback call is a state mutation that triggers re-render
4. **Optimistic UI**: The trace shows steps BEFORE they complete — spinner → result

The key insight: the agent's reasoning was always happening. This feature just
makes the invisible visible. No new AI capability was added — just observability.

---

# 18. Reaching 9.5/10 — System Prompt Engineering + Tool Completeness

## The three levers

Going from 7.5 → 9.5 required three things working together:

1. **Right tools** — the agent needs tools that match how a human engineer actually navigates code
2. **Right system prompt** — the agent needs to know *when* to use each tool, not just *that* they exist  
3. **Conversation memory** — already wired up; the agent had it all along

## `list_files` — the missing orientation step

Before `list_files`, the agent was dropped into a repo blind. It would guess file paths
like `src/models.py` or `lib/engine.py` and often get 404s. A human engineer's first
move when entering an unfamiliar codebase is: look at the directory tree.

```python
@mcp.tool()
def list_files(repo: str, path: str = "") -> str:
    # GitHub API: GET /repos/{owner}/{name}/contents/{path}
    # Returns a JSON array of {name, type, size} entries
    # Directories are marked with / so the agent can drill down
```

GitHub's contents API returns the directory listing as a JSON array when given a directory path,
or a single file object when given a file path. The tool handles both cases.

Now the agent's first move on any complex question is:
```
list_files("karpathy/micrograd") →
  micrograd/
  test/
  README.md  (8KB)
  setup.py   (1KB)
```
Then `list_files("karpathy/micrograd", "micrograd")` →
```
  __init__.py  (200B)
  engine.py    (4KB)
  nn.py        (2KB)
```

Two tool calls and the agent knows exactly which files to read. No guessing.

## System prompt engineering — the biggest quality lever

Here's an uncomfortable truth: **the system prompt is more important than the tools**.

You can give the agent 10 perfect tools, but if the system prompt says "use search_code
for everything", it will use search_code for everything — including cases where
search_symbol would be 10x more reliable.

### What the old prompt did wrong

```
When answering questions about code:
1. Start by calling search_code to find relevant code
2. If the initial results don't fully answer the question, search again with a different query
3. Use get_file_chunk to see more context around a result
...
```

Problems:
- Listed tools as an afterthought — no guidance on *when* to use each one
- No planning step — agent dives in immediately, often searching the wrong thing first
- Didn't mention `search_symbol` or `read_file` at all (they were added later, prompt never updated)
- Stopping criteria were vague ("stop searching when no new info")

### What the new prompt does right

**Tool selection guide** — each tool has an explicit trigger condition:
```
search_symbol(symbol_name) → Use when you KNOW the exact name.
  You found a caller that references `Value` → search_symbol("Value")

read_file(repo, filepath) → Use when you need imports, structure, or full context.
  "What does engine.py import?" → read_file (not search_code)
```

The agent now has a decision tree: "do I know the exact name? → search_symbol. 
Do I need the full file? → read_file. Am I exploring? → list_files."

**Mandatory planning step** — before any tools:
```
Before using any tools, write a 1-2 sentence plan:
"To answer this I need to: (1) find X, (2) trace how it connects to Y, (3) check Z."
```

This is the biggest single change. An agent that plans first makes fewer wasted tool calls.
It's the same reason senior engineers sketch a plan before coding — it prevents going
down wrong paths for 30 minutes before realizing the approach is wrong.

**Explicit strategy sequence** — ORIENT → FIND → READ → CONNECT → ANSWER:
```
1. ORIENT   — list_files to understand structure
2. FIND     — search_code for concepts, search_symbol for known names  
3. READ     — read_file for full context, get_file_chunk for targeted lines
4. CONNECT  — find_callers to trace how pieces relate
5. ANSWER   — cite every file path and line number
```

**History awareness**:
```
Check conversation history first — don't re-search things you already found this session
```

Without this instruction, the agent re-searches the same files every turn even when
they're already in the conversation history from 2 messages ago.

## Conversation memory — already there

The infrastructure was built from the start:

```python
# App.jsx — builds history from completed messages
const history = completedMsgs.slice(-10).map(m => ({ role: m.role, content: m.content }));

# agent.py — prepends history to messages
def _build_initial_messages(self, question, repo_filter, history=None):
    messages = [{"role": h["role"], "content": h["content"]} for h in (history or [])]
    messages.append({"role": "user", "content": question})
    return messages
```

History = last 10 messages (5 exchanges) as plain `{role, content}` pairs. The agent
doesn't re-attach retrieved code chunks — it can re-search if needed. This is intentional:
attaching full tool outputs to every history message would balloon the context to 50k+ tokens.

## The compound effect

None of these changes alone gets to 9.5. They work together:

| Without list_files | With list_files |
|---|---|
| Agent guesses file paths | Agent browses structure first |
| 2-3 failed tool calls per question | Straight to the right file |

| Old system prompt | New system prompt |
|---|---|
| Always starts with search_code | Chooses the right tool for the context |
| No planning step | Plans before any tool call |
| Ignores conversation history | Checks history before re-searching |
| search_symbol never used | Used immediately when name is known |

| Without memory | With memory |
|---|---|
| "What does backward() do?" (re-searches) | Already in context from Q2 |
| Can't answer "what about the other method?" | Knows which method from prior turn |

Combined: the agent now behaves like a senior engineer pairing with you — it knows
the codebase from prior turns, orients itself before diving in, picks the right tool
for each step, and plans before acting.

---

# 19. Dead Code Hygiene — Keeping the Codebase Clean

## Why it matters

Every unused file is a liability: it confuses newcomers, wastes build time, and can
silently break when touched by accident. When a feature is replaced or removed, the
old code must go with it — not be left as "maybe useful later."

## What we removed

Three React components were built early as prototypes but never wired into the UI:
- `SemanticMap.jsx` — 2D scatter plot of embeddings via UMAP
- `CodeGraph.jsx` — call-graph view built on top of React Flow
- `SequenceDiagram.jsx` — sequence diagram renderer

They were never imported, never rendered. Dead weight.

On the backend, `graph_service.py` and `map_service.py` served `/graph` and `/map`
endpoints that were only ever called by those deleted components. Removing the services
meant also removing their imports, their globals in `main.py`, their lifespan
initialisation, their dependency injectors, and their endpoint handlers.

**Key lesson**: deleting one file often means deleting 5 things — the file, its import,
its global, its route, and the calls to its methods. Trace the dependency graph fully
before declaring cleanup done.

## The checklist after any dead-code removal

```
1. Delete the file
2. Remove its import from every file that imported it
3. Remove globals / instances that held the object
4. Remove the routes or endpoints it backed
5. Remove calls to its methods from other handlers
6. Run the app — confirm no NameError or ImportError
```

---

# 20. Replacing a Third-Party Library with Custom SVG

## The problem with React Flow

React Flow is a full layout engine for node-based editors. It's powerful for drag-and-drop
workflow builders, but it adds friction when all you need is a static diagram:

- Forces its own DOM structure — every node gets wrapped in React Flow wrappers
- Injects its own CSS (`@xyflow/react/dist/style.css`) with handles, selection rings, and control buttons
- Calculates layout in its own coordinate system, making it hard to match visual design from elsewhere
- Adds 200KB+ to the bundle for features (drag, connect, selection) we never used

The `GraphDiagram` component (Architecture and Class Hierarchy diagrams) used React Flow
while `ExploreView` used a hand-crafted SVG canvas. The visual mismatch was obvious:
React Flow's mechanical look vs. Explore's refined custom rendering.

## The custom SVG canvas pattern

ExploreView established a pattern we can reuse everywhere:

```
┌─────────────────────────────────────────────────┐  ← ec-container (flex column)
│  [legend bar]                                   │
│  ┌───────────────────────────────────────────┐  │  ← ec-canvas (position: relative, overflow: hidden)
│  │  <svg> — arrows layer (position: absolute)│  │
│  │  <div node> (position: absolute, left, top)│  │  ← ec-card
│  │  <div node> ...                           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

Key pieces:
1. **Canvas div** with `position: relative` — all children absolute-positioned within it
2. **SVG overlay** stretched to 100%×100% for drawing bezier arrows between nodes
3. **Pan**: `onMouseDown/Move/Up` track a `pan` ref and apply `translate(x, y)` via CSS transform
4. **Zoom**: non-passive `wheel` event (must be added via `addEventListener` for `passive: false`)
   applies `scale(z)` on top of the translate
5. **Nodes**: plain `div`s styled with `left: node.x` / `top: node.y` matching computed layout
6. **Edges**: SVG `<path>` elements using cubic bezier curves

## BFS topological layout

To position nodes without a library:

```
1. Build adjacency list from edges
2. BFS from root nodes (in-degree 0) → assign each node a column depth
3. Split columns that exceed MAX_PER_COL rows vertically (wrap into next column)
4. Compute each column's height = n * CARD_H + (n-1) * ROW_GAP
5. Center each column against the TALLEST column's height:
   startY = (maxColH - colH) / 2 + TOP_PADDING   ← CRITICAL: always ≥ 0
6. Space columns evenly with COL_GAP between them
```

### The negative-y bug

A common mistake: centering around 0 with `startY = -totalH / 2 + offset`.
For a single-node column with CARD_H=178 and offset=60:
```
startY = -178/2 + 60 = -89 + 60 = -29px   ← node renders ABOVE the canvas
```

Fix: measure the tallest column first (`maxColH`), then offset every column
relative to that. The tallest column always starts at `TOP_PADDING`; shorter
columns start lower. All y values are therefore non-negative.

## Bezier arrow formula

```js
const tension = Math.abs(x2 - x1) * 0.5;
const d = `M ${x1} ${y1} C ${x1+tension} ${y1}, ${x2-tension} ${y2}, ${x2} ${y2}`;
```

`tension` scales with the horizontal distance so short hops get gentle curves
and long cross-column hops get pronounced S-curves. This is the same formula
D3 force graphs use for curved links.

## Why remove the library entirely

The best fix for a library causing visual inconsistency is to remove it and own
the rendering. This gives:
- Full control over every pixel
- Shared CSS with the rest of the app (no override battles)
- Smaller bundle (removed `@xyflow/react`)
- One rendering pipeline instead of two diverging approaches

---

# 21. API Request Size Limits — The 413 Error

## What happened

After adding Contextual Retrieval (section 14), each chunk's text grew from ~500
characters to ~8000 characters (the original chunk + the LLM-generated context
paragraph prepended to it). Batching 96 of these through the Nomic API:

```
96 chunks × ~8000 chars × ~1 byte/char ≈ 768KB per batch
```

That sounds fine — but HTTP JSON encoding adds overhead, and Nomic's free-tier
gateway has a ~10MB request body limit. With large repositories some batches
exceeded this limit, returning HTTP 413 Request Entity Too Large.

## The fix: two levers

**Lever 1 — smaller batches**: `_BATCH_SIZE = 96` → `_BATCH_SIZE = 32`

Reduces the worst-case batch body from ~768KB to ~256KB. Three times fewer
chunks per API call, but the extra round trips are cheap compared to the cost
of a failed 413 and a forced re-index.

**Lever 2 — truncate long texts**: `_MAX_CHARS = 8000` per text before sending

Embedding models have a token limit (~8192 tokens ≈ 6000–8000 chars). Text
beyond this limit is ignored by the model anyway — the embedding is computed
only from the first N tokens. Truncating explicitly:
- Prevents 413 errors on contextually-enriched chunks
- Makes the request body predictable regardless of chunk content
- Has negligible effect on retrieval quality (the key signal is at the top)

**Rule of thumb**: `batch_size × max_chars_per_text` should stay well under the
API's body limit with a 2× safety margin.

---

# 22. Agent Working Memory — The Note/Recall Pattern

## The problem

An LLM agent running a ReAct loop has no memory within a session except what
fits in its context window. As the loop runs, the context fills with tool outputs,
and early discoveries can scroll out of the effective window for models with
shorter contexts.

Worse: the agent might search for the same function twice in different iterations
because it "forgot" finding it three tool calls ago.

## Claude Code's approach

Claude Code uses a `TodoWrite` tool (persistent task list) and a Memory tool
(cross-session notes). The key insight is that **memory should be an explicit
tool call** — the agent actively decides what to remember, not a passive side-effect.

## Our implementation

```python
# In mcp_server.py

_session_notes: dict[str, str] = {}   # module-level — survives across tool calls

@mcp.tool()
def note(key: str, value: str) -> str:
    """Save a key fact to working memory for this session."""
    _session_notes[key] = value
    return f"✓ Noted: {key} = {value}"

@mcp.tool()
def recall_notes() -> str:
    """Retrieve all facts saved with note() during this session."""
    ...
```

In `agent.py`, at the start of every `stream()` call:
```python
from backend.mcp_server import clear_notes
clear_notes()   # Fresh memory for each new question
```

## System prompt instruction

The system prompt tells the agent:
```
WORKING MEMORY — USE IT
After every discovery, call note(key, value) immediately.
Call recall_notes() as step 1 before any search, and again before final answer.
```

Without the explicit instruction, the agent ignores the tools even though they
exist. **LLMs only use tools they are told to use in the system prompt.**

## What to note vs what not to note

Good candidates for `note()`:
- File paths where key classes/functions were found
- Module responsibilities ("auth.py handles JWT validation")
- Key relationships ("UserService calls AuthService.verify()")
- Dead ends ("search_code for 'batch_norm' returned no results")

Not worth noting:
- Raw code snippets (too verbose, already in tool output)
- Things that are obvious from the question itself

## Comparison

| Without note/recall | With note/recall |
|---|---|
| Re-searches the same function | Knows file from earlier note |
| Loses thread after 8+ tool calls | recall_notes() restores context |
| Final answer misses early findings | All notes consulted before answering |

---

# 23. Call Chain Tracing — Walking the Execution Graph

## What it enables

"What is the execution path when I call `train()`?" requires following a chain:
`train → _run_epoch → _forward_pass → _compute_loss → ...`

Standard semantic search finds individual functions but can't reconstruct the
execution path. `trace_calls` walks it explicitly.

## How it works

The ingestion pipeline already stores a `calls` field in each chunk's payload —
a list of function names that the function calls. This is parsed from the AST
at index time.

`trace_calls` does a BFS over this graph:

```python
def trace_calls(repo: str, symbol_name: str, max_depth: int = 3) -> str:
    visited = set()
    queue = [(symbol_name, 0, None)]   # (name, depth, parent)
    while queue:
        name, depth, parent = queue.pop(0)
        if name in visited or depth > min(max_depth, 5):
            continue
        visited.add(name)
        chunk = _store.find_symbol(repo, name)   # Qdrant lookup by name
        if chunk:
            # record in output tree
            for callee in chunk.payload.get("calls", [])[:6]:  # fan-out cap
                queue.append((callee, depth+1, name))
```

Output is a markdown call tree:
```
trace_calls: train
  train  [src/trainer.py:45]
    _run_epoch  [src/trainer.py:88]
      _forward_pass  [src/model.py:122]
        _compute_loss  [src/model.py:156]
```

## Design decisions

- **max_depth hard-capped at 5**: call graphs can be deep and circular. A cap of 5
  gives 3–4 hops of useful context without risking infinite loops.
- **fan-out capped at 6**: functions that call 20+ others (e.g. `__init__`) would
  explode the tree. Cap to the 6 most-called functions per level.
- **Cycle detection via `visited` set**: function A calling B calling A is common
  in recursive code. Skip already-visited nodes.
- **Graceful missing symbols**: if `find_symbol` returns None (function not indexed),
  note it and continue — don't crash the trace.

## Complementary to find_callers

| Tool | Direction | Question answered |
|---|---|---|
| `find_callers` | Up (callers) | "Who calls this function?" |
| `trace_calls` | Down (callees) | "What does this function call?" |

Together they let the agent answer "explain the full data flow through this system."

---

# 24. Persistent Repo Map — Agent Cross-Session Knowledge

## The cold-start problem

Every agent session starts blind. Even if you've asked 20 questions about
`karpathy/micrograd`, the next question costs 2-4 tool-call turns just to
re-orient: list_files → search for entry point → find GPT class → now answer.

Claude Code solves this with CLAUDE.md: a persistent file the agent reads before
every session. We replicate this pattern with `RepoMapService`.

## What a repo map contains

Built by scanning Qdrant chunk metadata — no LLM calls, no text vectors:

```json
{
  "repo": "karpathy/micrograd",
  "total_chunks": 45,
  "total_files":  8,
  "entry_files":  ["micrograd/engine.py"],
  "key_classes":  ["Value", "Neuron", "Layer", "MLP"],
  "files": {
    "micrograd/engine.py": {
      "classes":   ["Value"],
      "functions": ["__add__", "__mul__", "__pow__", "backward"]
    }
  }
}
```

Saved to `backend/repo_maps/{owner}_{name}.json`. Zero cost to load.

## Injecting it into agent context

The map is prepended to the user's first message before any tool calls:

```
╔══ REPO MAP: karpathy/micrograd (45 chunks, 8 files) ══╗
  Entry files : engine.py
  Key classes : Value, Neuron, Layer, MLP
  Key files   :
    engine.py        classes: Value  |  fns: __add__, __mul__, backward +2
    nn.py            classes: Neuron, Layer, MLP  |  fns: __call__, parameters
╚══ Skip list_files — use this map to target searches directly ══╝
```

The agent reads this before planning and can skip `list_files` entirely —
going straight to `search_symbol("Value")` or `trace_calls(repo, "backward")`.

## Token cost

~300 tokens per session. Equivalent to one `search_code` call.
The saved tool calls (list_files + orient step) cost 2-3x more.
Net negative token usage for any question longer than trivial.

## Invalidation

After re-ingestion, `RepoMapService.invalidate(repo)` deletes the cached JSON.
Next agent call rebuilds it from the fresh Qdrant data.

## Key pattern: inject static context before dynamic search

This is a general technique:
1. Build a cheap static summary from data you already have
2. Inject it as upfront context
3. Let the agent use it to avoid expensive discovery calls

Used by: CLAUDE.md, Claude Code's memory tool, GitHub Copilot's workspace index,
Cursor's codebase index.

---

# 25. Planning Before Acting — Structured Agent Reasoning

## The problem with reactive agents

Without explicit planning, agents jump straight to the first search that occurs
to them. This leads to:
- Redundant searches (finds "backward" 3 times with different queries)
- Missed coverage (searches "forward pass" but forgets to check "loss function")
- Opaque reasoning (users can't see what the agent is trying to do)

## The solution: structured `<plan>` blocks

The system prompt requires the agent to write a plan before its first tool call:

```
<plan>
Goal: find how the training loop calls the backward pass
Search 1: trace_calls(repo, "train", depth=3) for execution path
Search 2: search_symbol("backward") for the implementation
</plan>
```

This appears as the agent's first "thought bubble" in the UI.

## Why this works

1. **Forces deliberate thinking** — writing the plan slows the agent down just enough
   to consider multiple angles before committing to the first search
2. **Prevents backtracking** — agent that planned "search both forward AND backward"
   in step 1 doesn't need to come back for a second pass later
3. **Groups parallel searches** — the plan naturally identifies independent searches
   that can run concurrently (see section 26)
4. **Visible to users** — the thought bubble shows "here's what I'm going to do"
   before anything executes, building trust

## Implementation

Pure system prompt change — no code changes required. The agent emits the plan as
its `content` field before `tool_calls` in the first iteration. The existing
`_extract_thought()` captures it and the UI renders it as a thought bubble.

## Lesson: prompting is architecture

Adding a planning step is purely a prompting decision, but it changes the agent's
effective architecture from reactive (stimulus → response) to deliberate
(stimulus → plan → execute). Many agent "failures" are actually planning failures.

---

# 26. Parallel Tool Execution — asyncio.gather in the ReAct Loop

## The serial bottleneck

Original agent loop: tool calls happen one at a time.
- LLM returns `[search_code("forward"), search_code("backward")]`
- Agent calls search 1 → waits 500ms
- Agent calls search 2 → waits 500ms
- Total: 1000ms for two independent searches

Both searches hit the same Qdrant service and have no data dependency.
Running them serially wastes 500ms of wall-clock time per extra tool call.

## asyncio.gather for parallel IO

```python
# Before: serial
for tc in step["tool_calls"]:
    result = await self.mcp.call_tool(tc["name"], tc["input"])

# After: parallel
async def _run_tool(tc):
    try:
        return await self.mcp.call_tool(tc["name"], tc["input"])
    except Exception as e:
        return f"Tool error: {e}"

results = await asyncio.gather(*[_run_tool(tc) for tc in new_calls])
```

`asyncio.gather` runs all coroutines concurrently on the single event loop.
Since MCP tool calls are async HTTP requests (no CPU blocking), they interleave
efficiently — the total time is roughly `max(individual_times)` not `sum(times)`.

## UI ordering: emit tool_call first, results after

The tricky part is UX: we want the UI to show "these tools are running" before
any results arrive. Solution: separate the `yield` calls:

```python
# 1. Emit all tool_call events upfront (UI shows them immediately)
for tc in new_calls:
    yield {"type": "tool_call", "tool": tc["name"], "input": tc["input"]}

# 2. Execute all in parallel
results = await asyncio.gather(*[_run_tool(tc) for tc in new_calls])

# 3. Emit tool_result events after all complete
for tc, result in zip(new_calls, results):
    yield {"type": "tool_result", "tool": tc["name"], "output": result[:500]}
```

The user sees all tool_call events together, then all results together.
This correctly communicates "these ran in parallel."

## Deduplication before parallel execution

Duplicate calls (same tool + same args) are filtered out synchronously before
launching the parallel batch. The `seen_calls` set is checked in the main thread
(not inside the async task) to avoid race conditions:

```python
for tc in step["tool_calls"]:
    call_key = (tc["name"], tuple(sorted(tc["input"].items())))
    if call_key in seen_calls:
        # emit skip message, don't add to new_calls
    else:
        seen_calls.add(call_key)
        new_calls.append(tc)

results = await asyncio.gather(*[_run_tool(tc) for tc in new_calls])
```

## When models emit parallel calls

- **Gemini** (default): emits multiple `tool_calls` in one turn naturally
- **Groq**: `parallel_tool_calls=False` is set to prevent the hermes bug — single calls only
- **OpenRouter**: depends on the model; most support parallel tool calls

To encourage parallel calls, the system prompt says:
> "For multi-part questions, call 2-3 searches in a SINGLE turn — they run in parallel"

---

# 27. Streaming Diagram Generation — SSE Progress for Long LLM Calls

## The blank spinner problem

Generating a codebase tour takes 5-10 seconds (Qdrant scan + LLM call).
During this time the user sees a spinner with no feedback — "is it working?
did it hang? how long will this take?"

The solution: stream progress events over SSE, the same mechanism used for
the agent's tool-call trace.

## Generator-based progress

`DiagramService.build_tour_stream()` is a synchronous Python generator that
yields `(stage, progress, data)` dicts at key checkpoints:

```python
def build_tour_stream(self, repo: str):
    yield {"stage": "loading",    "progress": 0.10, "message": "Loading chunks…"}
    chunks = self._list_chunks(repo)

    yield {"stage": "analysing",  "progress": 0.35, "message": "Analysing chunks…"}
    # ... rank and build prompt ...

    yield {"stage": "generating", "progress": 0.55, "message": "Generating tour with AI…"}
    raw = self._gen.generate(...)   # ← the slow part

    yield {"stage": "parsing",    "progress": 0.90, "message": "Finalizing…"}
    # ... parse and validate ...

    yield {"stage": "done",       "progress": 1.00, **tour}
```

The LLM call (stage "generating") is where time is actually spent — the user
now sees progress jump from 35% to 55% when it starts, and to 90% when it finishes.

## Bridging sync generator → async SSE

The generator is synchronous but the FastAPI endpoint is async. We bridge them
with the same `asyncio.Queue + run_in_executor` pattern used by `_stream_final_answer`:

```python
@app.get("/repos/{owner}/{name}/tour/stream")
async def stream_tour(owner, name, diagram_svc):
    async def _event_stream():
        queue = asyncio.Queue()
        loop  = asyncio.get_running_loop()

        def _run():
            for event in diagram_svc.build_tour_stream(repo):
                loop.call_soon_threadsafe(queue.put_nowait, event)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(None, _run)   # run sync gen in thread pool

        while True:
            event = await queue.get()
            if event is None: break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
```

## Frontend progress bar

The client subscribes via fetch + ReadableStream (same pattern as agent streaming).
Progress events update a `loadStage` state that drives a CSS progress bar:

```jsx
<div style={{
  height: "100%",
  width: `${pct}%`,         // ← animated from 0 to 100
  background: "var(--accent)",
  transition: "width 0.4s ease",  // ← smooth animation between stages
}} />
```

The `transition: width 0.4s ease` makes the bar glide smoothly between the
discrete progress checkpoints (10% → 35% → 55% → 90% → 100%) rather than
jumping abruptly.

## Why generators, not callbacks

An alternative would be a progress callback:
```python
def build_tour(self, repo, progress_cb=None):
    progress_cb(0.1, "Loading…")
    ...
```

Generators are cleaner because:
- No callback parameter threaded through every helper method
- `yield` composes naturally with other generators
- The caller (SSE endpoint) pulls at its own pace
- Easier to test: just iterate the generator and inspect yields

## Token cost

Zero extra tokens. The LLM call is identical to the non-streaming version.

---

# 28. Docker CI/CD Pipeline — GitHub Actions + HuggingFace Spaces + Vercel

## What CI/CD means

**CI** (Continuous Integration): every push to `main` automatically runs checks
(lint, build, syntax) to catch errors before they land.

**CD** (Continuous Deployment): if those checks pass, the new code is automatically
deployed to production. No manual steps, no "remember to deploy" errors.

```
git push main
    ↓
GitHub Actions triggers
    ├── CI job: lint + build (catches errors)
    └── Deploy jobs (run in parallel if CI passes):
        ├── deploy-backend  → HuggingFace Spaces
        └── deploy-frontend → Vercel
```

## The deployment architecture

```
HuggingFace Spaces (free)          Vercel (free)
┌──────────────────────┐           ┌──────────────────────┐
│ Docker container     │           │ Static CDN           │
│ port 7860            │    ←──    │ React SPA            │
│ FastAPI + Qdrant     │  API req  │ VITE_API_URL=HF URL  │
│ Python 3.11          │           │ 100+ edge nodes      │
└──────────────────────┘           └──────────────────────┘
```

- **HF Spaces** runs your Docker container. It's a full server with persistent
  processes, suitable for a FastAPI backend.
- **Vercel** hosts static files (your compiled React app). It has no server
  process — just CDN nodes serving HTML/JS/CSS.
- They communicate at runtime: the browser makes `fetch()` calls from the Vercel
  URL to the HF Space URL.

## The monorepo problem — `git subtree split`

Our project lives inside a subdirectory of a larger repo:
```
deep-learning-from-scratch/
└── github-rag-copilot/    ← our project
    ├── Dockerfile          ← HF Spaces needs this at ROOT
    ├── backend/
    └── ui/
```

HuggingFace Spaces expects `Dockerfile` at the repository root. But we can't
push the entire monorepo — HF would see no Dockerfile at the root.

**`git subtree split`** solves this:

```bash
git subtree split --prefix=github-rag-copilot -b hf-deploy
```

This command rewrites git history to create a new branch where:
- `github-rag-copilot/Dockerfile` becomes `./Dockerfile`
- `github-rag-copilot/backend/main.py` becomes `./backend/main.py`
- Every other directory is removed

Then we push that branch as `main` to the HF Space's git endpoint:
```bash
git push https://user:$TOKEN@huggingface.co/spaces/user/space hf-deploy:main --force
```

`--force` is needed because subtree split produces different commit SHAs each
time (history is being rewritten). The HF Space repo is a pure deploy target,
so force-pushing it is safe.

## GitHub Actions concepts

### Secrets vs Variables

```yaml
${{ secrets.HF_TOKEN }}     # encrypted, never shown in logs
${{ vars.HF_USERNAME }}     # plain text, visible in logs
```

Use **secrets** for anything sensitive: tokens, API keys, passwords.
Use **vars** for non-sensitive config: usernames, space names, URLs.

### `fetch-depth: 0`

By default, `actions/checkout@v4` does a shallow clone (only the latest commit).
`git subtree split` needs to walk the full history to know which commits touched
the subdirectory. `fetch-depth: 0` fetches everything.

### Parallel jobs

```yaml
jobs:
  deploy-backend:   # starts immediately
    ...
  deploy-frontend:  # starts immediately, in parallel
    ...
```

Jobs in GitHub Actions run in parallel unless you add `needs: [job-name]`.
Since backend and frontend are independent, both deploy at the same time.
Total deploy time ≈ max(backend, frontend) instead of sum.

### `concurrency`

```yaml
concurrency:
  group: deploy-${{ github.ref }}
  cancel-in-progress: true
```

If you push twice in quick succession, this cancels the first deploy before
starting the second. Prevents two deploys racing to overwrite each other.

## Vercel CLI deployment

```bash
npx vercel --prod --token="$VERCEL_TOKEN" --yes
```

- `npx vercel` downloads and runs the Vercel CLI without global installation
- `--prod` promotes the deployment to the production URL (not a preview URL)
- `--token` authenticates via the secret stored in GitHub
- `--yes` skips interactive confirmation (required in non-TTY environments like CI)
- `VERCEL_ORG_ID` + `VERCEL_PROJECT_ID` env vars tell the CLI which project to deploy

## VITE_API_URL — baked at build time

Vite replaces `import.meta.env.VITE_API_URL` with a literal string at build time:
```js
// Source code:
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// After Vite build with VITE_API_URL=https://umang-github-rag-copilot.hf.space:
const BASE = "https://umang-github-rag-copilot.hf.space";
```

This means the production build always points to the right backend without any
runtime configuration. The tradeoff: changing the backend URL requires rebuilding
the frontend.

## One-time setup checklist

Before the GitHub Actions workflow will succeed, you need:

**HuggingFace Space:**
- [ ] Create a new Space with SDK = Docker at `https://huggingface.co/new-space`
- [ ] Set env vars in Space → Settings → Variables (not in Dockerfile):
      `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `GROQ_API_KEY`,
      `GEMINI_API_KEY`, `NOMIC_API_KEY`, `FRONTEND_URL` (Vercel URL)
- [ ] Generate a token at `https://huggingface.co/settings/tokens` with Write permission

**Vercel:**
- [ ] Create project at `https://vercel.com/new`, root dir = `github-rag-copilot/ui`
- [ ] Set `VITE_API_URL` = your HF Space URL in Vercel project env vars
- [ ] Get `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID` from `.vercel/project.json`
      after running `npx vercel link` locally once

**GitHub repo secrets/variables:**
- [ ] `HF_TOKEN` (secret), `VERCEL_TOKEN` (secret)
- [ ] `VERCEL_ORG_ID` (secret), `VERCEL_PROJECT_ID` (secret)
- [ ] `HF_USERNAME` (var), `HF_SPACE_NAME` (var), `HF_SPACE_URL` (var)
Only the delivery mechanism changes — progress events are pure metadata.

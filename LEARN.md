# Cartographer — How It Works (Your Personal Reference)

This document explains everything in the project from first principles — the AI
concepts, the software stack, every file, the architecture decisions, and how the
deployment pipeline works. After reading this you should be able to build it again
from scratch.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [System Architecture](#2-system-architecture)
3. [The Tech Stack](#3-the-tech-stack)
4. [AI/ML Concepts](#4-aiml-concepts)
   - Embeddings & Semantic Search
   - RAG (Retrieval-Augmented Generation)
   - AST Chunking
   - Hybrid Search
   - HyDE & Query Expansion
   - Re-ranking
   - Contextual Retrieval
   - Agent Workflow (ReAct)
   - Tool Use & Structured JSON Output
   - MCP (Model Context Protocol)
5. [Codebase Walkthrough](#5-codebase-walkthrough)
6. [Architecture Decisions](#6-architecture-decisions)
7. [Deployment & CI/CD](#7-deployment--cicd)
8. [SSE Streaming — Deep Dive](#8-sse-streaming--deep-dive)
9. [Python Async/Await — Deep Dive](#9-python-asyncawait--deep-dive)
10. [React Hooks — Deep Dive](#10-react-hooks--deep-dive)
11. [Dimensionality Reduction & UMAP](#11-dimensionality-reduction--umap)
12. [LLM Features: Prompt Caching, Structured Output, System Prompts](#12-llm-features-prompt-caching-structured-output-system-prompts)
13. [Production Patterns: Rate Limits, Retry, Multi-Provider Cascade](#13-production-patterns-rate-limits-retry-multi-provider-cascade)
14. [Agent Deep Dive: Working Memory, Parallel Execution, Streaming Thoughts](#14-agent-deep-dive-working-memory-parallel-execution-streaming-thoughts)
15. [Data Layer: Repo Map & Diagram Generation](#15-data-layer-repo-map--diagram-generation)
16. [Qdrant In Depth](#16-qdrant-in-depth)
17. [FastAPI Lifespan and Dependency Injection](#17-fastapi-lifespan-and-dependency-injection)
18. [AST Chunking and tree-sitter](#18-ast-chunking-and-tree-sitter)
19. [Prompt Templates](#19-prompt-templates)
20. [D3.js Force Simulation (Explore View)](#20-d3js-force-simulation-explore-view)
21. [Agent-Based Tour Generation — Why Agents Beat One-Shot](#21-agent-based-tour-generation--why-agents-beat-one-shot)
22. [Quality Patterns: Evaluator-Optimizer, Terse Docs, and Tool Descriptions](#22-quality-patterns-evaluator-optimizer-terse-docs-and-tool-descriptions)
23. [Quality Improvements from the Anthropic Cookbook](#chapter-23--quality-improvements-from-the-anthropic-cookbook)
24. [Full ReAct Pipeline: Agentic Phase 1 + Phase 2](#24-full-react-pipeline-agentic-phase-1--phase-2)
25. [Tour Quality: Signal-Checklist Stopping, Gaps, and Claude Code /init Study](#25-tour-quality-signal-checklist-stopping-gaps-and-claude-code-init-study)
26. [Tour Framing, Prompt Hygiene, Model Tiering, and Navigation Tools](#26-tour-framing-prompt-hygiene-model-tiering-and-navigation-tools)

---

## 1. What We Built

**Cartographer** is a RAG (Retrieval-Augmented Generation) system that lets you ask
natural-language questions about any public GitHub repository.

The problem it solves: LLMs like GPT-4 or Gemini don't know about your private code
or repos that changed after their training cutoff. Even for public repos they often
hallucinate specifics. Cartographer solves this by:

1. **Indexing** the repo — fetching every file, splitting code into functions/classes,
   embedding each chunk into a vector, storing in a cloud vector database.
2. **Retrieving** on every question — embedding the question, searching for the most
   semantically similar code chunks, and passing them as context to the LLM.
3. **Generating** an answer grounded in the actual source code, with citations.

Beyond basic Q&A it also has:
- An **agent mode** where the LLM autonomously calls tools (search, read file, trace
  call chains, draw diagrams) to investigate complex questions
- An **Explore view** that generates a concept map of the entire repo's architecture
- A **Diagram view** that draws class and flow diagrams with Mermaid

---

## 2. System Architecture

```
User's browser (React + Vite)
       │
       │  HTTPS (JSON / SSE)
       ▼
FastAPI backend  ←→  MCP server (same process, mounted at /mcp)
  │         │
  │         └──→  Qdrant Cloud  (vector DB, stores embeddings)
  │
  ├──→  Nomic API        (embeds text → 768-dim vector)
  ├──→  Groq / Gemini / Cerebras  (LLM generation)
  └──→  Cohere API       (optional: neural re-ranker)

Ingestion pipeline (runs on demand, same process):
  GitHub API → files → tree-sitter AST chunker → Nomic embedder → Qdrant
```

**Data flow for a question:**
```
Question
  → embed question (Nomic API)
  → hybrid search Qdrant (semantic + BM25)
  → re-rank top-N results (Cohere or local cross-encoder)
  → inject top chunks into prompt
  → stream LLM response (Groq/Gemini/Cerebras)
  → SSE tokens → React frontend
```

**Deployment split:**
- **Frontend** → Vercel (global CDN, static files)
- **Backend** → HuggingFace Spaces (Docker container, CPU free tier)
- **Vector DB** → Qdrant Cloud (managed, free 1 GB tier)

---

## 3. The Tech Stack

### Python & FastAPI

**FastAPI** is a Python web framework for building HTTP APIs. You define endpoints
with decorators:

```python
@app.post("/ingest")
async def ingest(request: IngestRequest) -> IngestResponse:
    ...
```

Key features we use:
- **Pydantic models** for request/response validation — if the request body doesn't
  match the schema, FastAPI auto-returns a 422 error. No manual validation needed.
- **async/await** — FastAPI is built on ASGI (Asyncio-based). A single process can
  handle many simultaneous requests because while one request awaits a network call
  (Qdrant, Nomic API), another request runs.
- **StreamingResponse** — for SSE streaming. Instead of waiting for the full LLM
  response, we yield tokens as they arrive. FastAPI streams each `data:` event to
  the browser.
- **Depends()** — dependency injection. Services (Qdrant, embedder) are created once
  at startup and injected into every route handler that needs them.
- **Lifespan** — `@asynccontextmanager` function that runs setup on startup and
  teardown on shutdown. We create all service objects here.

### React & Vite

**React** is a JavaScript library for building UIs. You write components — functions
that return JSX (HTML-like syntax) and re-render when their state changes.

**Vite** is the build tool. It:
- Runs a dev server with instant hot-reload on file change
- Bundles everything into optimised static files for production (`npm run build → dist/`)
- Handles env vars: anything prefixed `VITE_` in `.env` is baked into the JS bundle

```
npm run dev    → local dev server at localhost:5173
npm run build  → produces dist/ (uploaded to Vercel)
```

Key React patterns used in this project:
- **useState** — local state (current repo, messages, streaming flag)
- **useEffect** — side effects (fetch repos on mount, reset view when repo changes)
- **useRef** — mutable values that don't trigger re-renders (stopStream fn, scroll ref)
- **SSE streaming** — `fetch()` with streaming body + `ReadableStream.getReader()` to
  consume tokens as they arrive from the backend

### Qdrant

A **vector database** purpose-built for similarity search. Think of it as a database
where instead of `WHERE name = 'foo'` you query "give me the 10 rows most similar to
this vector."

Under the hood it uses **HNSW** (Hierarchical Navigable Small World) — a graph where
each vector is connected to its neighbours. Search traverses the graph in O(log N)
time instead of scanning all N vectors.

We use Qdrant Cloud (managed, free tier). No infrastructure to manage — just an API
key and a collection name.

---

## 4. AI/ML Concepts

### Embeddings & Semantic Search

An **embedding model** maps text to a fixed-size vector of floats:
```
"def backward(self): ..."  →  [0.12, -0.44, 0.93, ...]  (768 floats)
```

Semantically similar texts produce vectors with high **cosine similarity** (ranges
0–1, where 1 = identical direction, 0 = unrelated).

This enables *semantic* search: query "how does backpropagation work?" matches
"def backward(self)" even though no words overlap. The model learned that
"backpropagation" and "backward pass" are related concepts.

We use **`nomic-embed-code`** via the Nomic API (free tier). It's trained on code and
understands programming concepts, not just natural language.

**Why use an API instead of running locally?**
HuggingFace Spaces has a RAM limit (~16GB for free CPU). The embedding model is
~550MB. We avoid loading it by calling the Nomic API. Zero RAM cost, same quality.

### RAG (Retrieval-Augmented Generation)

RAG solves a fundamental LLM limitation: models have a training cutoff and can't see
your specific code. RAG gives the LLM a "reading list" — retrieved chunks of actual
source code — before asking it to answer.

**Why not just put the whole repo in the context?**
GPT-4 has a 128K token context. A medium-sized repo has 500K–5M tokens. It doesn't
fit. Even if it did, cost and latency would be prohibitive. RAG retrieves only the
*relevant* ~10 chunks (typically 5–20K tokens total) rather than everything.

**The two phases:**

*Ingestion* (once per repo):
```
GitHub API → download files → filter (no node_modules, no *.lock)
  → AST chunker → functions + classes as chunks
  → Nomic embed → 768-dim vector per chunk
  → Qdrant upsert (vector + sparse BM25 vector + payload metadata)
```

*Query* (every question):
```
Question → embed → hybrid search Qdrant → re-rank → top-10 chunks
  → build prompt: system_prompt + chunks + question
  → stream LLM → SSE tokens → browser
```

### AST Chunking

**Why not split on character count?** Split a function at character 1000 and you lose
either the signature (can't answer "what does this accept?") or the body (can't answer
"how does this work?"). Arbitrary windows break code.

**AST** (Abstract Syntax Tree) is the structured representation a language parser
builds. Every function, class, and method is a node with precise start/end lines.

We use **tree-sitter** (multi-language parser) to walk the AST and extract each
function/class as its own chunk. Every chunk is a complete, self-contained unit.

```python
# ingestion/code_chunker.py — simplified
tree = parser.parse(source_code.encode())
for node in walk(tree):
    if node.type in ("function_definition", "class_definition"):
        yield Chunk(text=source_code[node.start_byte:node.end_byte], ...)
```

Long functions that exceed the 8192-token embedding limit fall back to a
line-window approach with overlap.

### Hybrid Search

**Pure semantic search** has a weakness: it's fuzzy. Query "IndexError in tokenizer"
might not retrieve chunks where "IndexError" appears literally, because the model
compresses meaning into geometry and loses exact tokens.

**BM25** (keyword search) is the opposite: exact and fuzzy term matching, like a
search engine. Great for identifiers, error messages, function names.

**Hybrid search** combines both with **RRF** (Reciprocal Rank Fusion): take the
ranked lists from semantic and BM25, combine scores using `1/(k + rank)` for each,
re-sort. Consistently beats either alone.

Qdrant supports this natively: one request, two vector fields (`dense` + `sparse`),
one fused result. No client-side merging needed.

### HyDE & Query Expansion

Two techniques that improve retrieval quality *before* the search:

**HyDE** (Hypothetical Document Embeddings): Instead of embedding the question
("how does backpropagation work?"), ask the LLM to generate a short code snippet that
would *answer* the question, then embed that. A code snippet is much closer to what's
in the index than a natural-language question.

**Query Expansion**: Ask the LLM to generate 2–3 alternative phrasings of the
question, run all of them through Qdrant, merge results with RRF. Catches chunks that
only surface under one specific phrasing.

Both are enabled by default (`USE_HYDE=true`, `EXPAND_QUERIES=true`).

### Re-ranking

The first retrieval pass returns the top-N candidates fast (HNSW is approximate).
A **re-ranker** is a slower but more accurate model that re-scores each candidate
against the query and reorders them.

We use **Cohere rerank-v3.5** (API, free 1000 calls/month) as the primary, falling
back to a local **ms-marco cross-encoder** if Cohere isn't configured.

Cross-encoders: instead of embedding query and document *separately* and comparing
vectors, the model sees `[query] [SEP] [document]` jointly and scores their relevance
directly. More accurate than bi-encoder similarity because it can reason about
interactions between query tokens and document tokens.

### Contextual Retrieval

A technique from Anthropic: before embedding a chunk, prepend a one-sentence
description generated by an LLM explaining what the chunk does and where it sits in
the codebase. This description is embedded along with the code, making retrieval more
context-aware.

Example:
```
"This function implements the backward pass for the Value class,
computing gradients for the autograd engine."

def backward(self):
    ...
```

The re-indexing button in the sidebar triggers this. It's optional (and slow for large
repos) but measurably improves retrieval quality.

**The math behind why this works:** When you embed just `def backward(self): ...`, the
vector points toward "Python function, gradient computation". When you embed the
description + code, the vector gets pulled toward "autograd, backward pass, neural
network training" — a region closer to questions like "how does gradient flow?". The
Anthropic paper showed 35–49% retrieval improvement for this technique.

### Agent Workflow (ReAct)

Standard RAG is one shot: embed → retrieve → generate. This fails for complex
questions that require multiple steps of investigation (e.g. "trace how data flows
from the loader into the model's forward pass").

**ReAct** (Reason + Act) is the loop that enables agents:

```
THOUGHT:  "I need to find the data loader first, then trace calls into forward()"
ACTION:   search_code(query="data loader")
OBSERVATION: [returned chunks about DataLoader class]
THOUGHT:  "Found it. Now I need to see what calls forward()"
ACTION:   trace_calls(symbol_name="forward")
OBSERVATION: [call sites]
THOUGHT:  "I have enough to answer now."
ACTION:   (stop, synthesize answer)
```

The model alternates between thinking (reasoning about what to do) and acting (calling
a tool). Each observation updates its understanding. This continues until it decides
it has enough information to answer.

### Tool Use & Structured JSON Output

**Tool use** (also called "function calling") is a feature of modern LLMs where you
describe functions to the model in JSON schema format, and the model returns structured
JSON when it wants to call one instead of free text:

```json
// You send the LLM a tool schema:
{
  "name": "search_code",
  "description": "Search the indexed repository for code matching a query",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "repo": {"type": "string"}
    },
    "required": ["query", "repo"]
  }
}

// Model returns a structured tool call instead of prose:
{"tool": "search_code", "input": {"query": "backward pass", "repo": "karpathy/micrograd"}}
```

Your code executes the real function, passes the result back as an "observation", and
the model continues. This is fundamentally different from prompting the model to output
JSON — the model is given a dedicated "I want to call a tool" output channel separate
from its text output.

**json_mode** is a simpler related feature: some APIs (`response_format: json_object`)
force the model to output valid JSON. We use this for structured outputs like the
Explore view concept map — the LLM returns a structured JSON of nodes and edges, which
D3 renders as a force-directed graph.

Our agent has 10 tools: `search_code`, `search_symbol`, `read_file`, `list_files`,
`get_file_chunk`, `find_callers`, `trace_calls`, `note`, `recall_notes`, `draw_diagram`.

### MCP (Model Context Protocol)

**MCP** is a protocol (like HTTP, but for LLM tools) that standardises how AI models
discover and call tools. Instead of hard-coding tool schemas into each LLM call, you
run an **MCP server** that exposes tools via a JSON-RPC interface. Any MCP-compatible
client (Claude Code, our agent, Cursor) can connect and discover the tools dynamically.

In this project:
- **MCP server** runs inside the same FastAPI process, mounted at `/mcp`
- **MCP client** is the AgentService — it connects to `/mcp` at startup, lists
  available tools, and converts them to the format each LLM provider expects
- This means you can use the agent from Claude Code too (`/mcp-status` shows the tools)

Why this matters: the tools are defined once (in `mcp_server.py`) and work everywhere.

**Without MCP**, every client that wants to use your tools has to reimplement them
in its own format. With MCP, the server is the menu — any client reads the same menu:

```
Without MCP                            With MCP
───────────────────────────────        ──────────────────────────────────────
Claude Code  Agent   Cursor            Claude Code    Agent    Cursor
    │          │       │                    │            │        │
 writes     writes  writes                  └────────────┴────────┘
 its own    its own its own                              │
 search     search  search                     MCP Server at /mcp
 tool       tool    tool                       🔧 search_code
                                               🔧 read_file
(same tool, defined 3 times)                   🔧 recall_notes
                                        (defined once, used by everyone)
```

### How AgentService actually works

A common misconception: the AgentService is **not** the thinker — it's just a
coordinator. The LLM is the thinker. The AgentService is a while loop.

```
AgentService                      LLM (Gemma/Llama/etc.)
(dumb coordinator)                (the actual thinker)
──────────────────                ──────────────────────

1. Receives user question
   "how does backprop work?"
        │
        │  sends question +
        │  list of tools ─────────────────────────────►
        │
        │                          LLM decides:
        │                          "I should search first"
        │
        │◄────────────────────────  outputs a tool call:
        │                          {"tool": "search_code",
        │                           "args": {"query": "backprop"}}
        │
2. AgentService calls /mcp,
   runs the search, gets chunks
        │
        │  sends chunks back
        │  as a new message ───────────────────────────►
        │
        │                          LLM reads chunks,
        │                          has enough to answer
        │
        │◄────────────────────────  streams final answer
        │
3. AgentService streams
   it to the user
```

The LLM doesn't "run" tools — it outputs text that **looks like** a tool call.
The AgentService reads that output, notices it's a tool request (not a final answer),
executes it via MCP, and sends the result back. This loops until the LLM outputs
plain prose instead of another tool call.

Think of it like a person in a room communicating only by written notes:
- **LLM** = smart person in a locked room, can only pass notes under the door
- **AgentService** = assistant outside, reads notes, fetches whatever is asked for, slides results back
- **Tools** = things the assistant can actually go do (search Qdrant, read a file)

In code, the core loop is simply:

```python
while True:
    response = llm.generate(messages)

    if response.has_tool_call:
        result = mcp.call_tool(response.tool_name, response.args)
        messages.append(result)   # slide it back under the door
    else:
        return response.text      # LLM gave a final answer, done
```

---

## 5. Codebase Walkthrough

```
cartographer/
├── ingestion/              ← Pipeline: GitHub → chunks → Qdrant
│   ├── repo_fetcher.py     Calls GitHub API, downloads files, respects rate limits
│   ├── file_filter.py      Decides which files to index (no node_modules, no *.lock)
│   ├── code_chunker.py     tree-sitter AST → functions/classes as chunks
│   ├── embedder.py         Calls Nomic API (or Voyage) to get 768-dim vectors
│   └── qdrant_store.py     Upserts/searches/deletes from Qdrant Cloud
│
├── retrieval/
│   └── retrieval.py        RetrievalService: hybrid search + HyDE + query expansion
│                           Reranker: Cohere API or local cross-encoder fallback
│
├── backend/                ← FastAPI application
│   ├── main.py             Thin entry point: lifespan, CORS, router registration (~110 lines)
│   ├── dependencies.py     Service container + get_* providers + rate limiter + PostHog
│   ├── config.py           Pydantic Settings: reads env vars, validates at startup
│   ├── mcp_server.py       FastMCP server: defines tools the agent can call
│   ├── mcp_client.py       Connects to /mcp, wraps tools for LLM providers
│   ├── models/
│   │   └── schemas.py      All Pydantic request/response models
│   ├── routers/            ← Route handlers, split by domain
│   │   ├── ingestion.py    /ingest, /repos (CRUD)
│   │   ├── query.py        /search, /query, /query/stream (RAG)
│   │   ├── agent.py        /agent/stream, /agent/models (agentic RAG)
│   │   ├── diagrams.py     /repos/{owner}/{name}/tour|diagram
│   │   └── mcp_routes.py   /mcp-status, /mcp-prompt
│   └── services/
│       ├── ingestion_service.py  Orchestrates: fetch → chunk → embed → store
│       ├── generation.py         LLM generation: multi-provider, streaming, fallback
│       ├── agent.py              ReAct loop: thought → tool call → observe → repeat
│       ├── diagram_service.py    Prompts LLM to generate Mermaid diagrams
│       └── repo_map_service.py   Builds persistent "repo map" — summary of all files
│
├── ui/                     ← React frontend
│   ├── src/
│   │   ├── main.jsx        App entry point + Vercel Analytics
│   │   ├── App.jsx         Root component: state, SSE streaming, layout (800+ lines)
│   │   ├── api.js          All fetch() calls to the backend
│   │   ├── index.css       Design system (CSS variables + all component styles)
│   │   └── components/
│   │       ├── Sidebar.jsx       Repo list, session history, URL ingestion input
│   │       ├── Message.jsx       Chat bubble + agent trace (thoughts + tool steps)
│   │       ├── DiagramView.jsx   Mermaid diagram renderer + fullscreen
│   │       ├── ExploreView.jsx   Concept map graph (D3 force-directed)
│   │       ├── SourceCard.jsx    Code citation card (filepath + snippet)
│   │       └── MermaidBlock.jsx  Lazy-loaded Mermaid renderer
│   ├── public/
│   │   └── favicon.svg     Compass rose SVG (kite-diamond points)
│   └── index.html          HTML shell (Vite replaces <script> at build time)
│
├── Dockerfile              Backend container recipe for HuggingFace Spaces
├── requirements.txt        Python dependencies
├── CLAUDE.md               Instructions for Claude Code (you're reading LEARN.md)
├── PLAN.md                 Build phases and feature tracker
└── .github/workflows/
    ├── ci.yml              Run tests on every push
    └── deploy.yml          Deploy backend to HF Spaces + frontend to Vercel
```

### Key file deep-dives

---

#### Ingestion pipeline (`ingestion/`)

This is the one-time setup phase. You point it at a GitHub repo and it processes
everything into Qdrant so it can be searched later.

**`ingestion/repo_fetcher.py`** — Talks to the GitHub API to download the repo's
file tree and file contents. Handles rate limiting (60 req/hr unauthenticated,
5000/hr with a token). Doesn't clone the repo — fetches files via API so it works
in serverless environments with no disk space.

**`ingestion/file_filter.py`** — Decides what to skip. Ignores `node_modules/`,
`*.lock`, `*.min.js`, images, binaries, auto-generated files, etc. Without this,
you'd waste embeddings budget on files that contain no useful information (and
confuse retrieval with noise).

**`ingestion/code_chunker.py`** — The most important ingestion file. Uses
**tree-sitter** (a real language parser, not regex) to split code at meaningful
boundaries — functions, classes, methods — rather than fixed character windows.
This matters because a function is a coherent unit of meaning; a 500-character
window that cuts across two functions is not.

```
Without AST chunking        With AST chunking
──────────────────          ──────────────────
def forward(self):          chunk 1: full forward() function
    x = self.lin1(          chunk 2: full backward() function
──── cut here ────          chunk 3: full __init__() method
    x = self.relu(
    return self.lin         (each chunk = one semantic unit)
```

Each chunk also stores metadata: `file`, `start_line`, `end_line`, `name`,
`language`, `base_classes`, `methods` — used later for the diagram view.

**`ingestion/embedder.py`** — Calls the Nomic API (or Voyage) to convert each
chunk's text into a 768-dimensional vector. Also generates a **contextual
description** using an LLM before embedding ("this function is the backward pass
of the Value class in micrograd's autograd engine") — this pulls the vector toward
the right region of embedding space for retrieval.

**`ingestion/qdrant_store.py`** — Writes to and reads from Qdrant Cloud. Each
chunk is stored as a **point** with two vectors: a dense float vector (semantic)
and a sparse BM25 vector (keyword). `hybrid_search()` queries both at once and
gets back a single fused ranked list.

---

#### Retrieval (`retrieval/retrieval.py`)

Called on every user question. Takes a query string, returns the most relevant
chunks. This is where all the quality techniques live:

```
User question
     │
     ├─ HyDE: ask LLM to write a hypothetical code answer
     │        embed that instead of the raw question
     │
     ├─ Query expansion: generate 2-3 rephrased versions of the question
     │
     ├─ Search all variants through Qdrant hybrid search
     │
     ├─ Merge all result lists with RRF
     │
     └─ Rerank top-N with Cohere (or local cross-encoder)
          │
          └─ return top chunks to generation
```

Without any of these: you'd embed the raw question and return whatever Qdrant
finds. With all of them: you've dramatically increased the chance that the
genuinely relevant code surfaces, regardless of how the user phrased the question.

---

#### Backend (`backend/`)

**`backend/config.py`** — Reads all environment variables (API keys, URLs,
feature flags) using Pydantic Settings. If a required variable is missing,
the app refuses to start with a clear error rather than crashing mysteriously
later. One settings object is imported everywhere — no `os.getenv()` scattered
through the codebase.

**`backend/dependencies.py`** — The glue between lifespan and routers. Holds a
`services` container object whose attributes (`services.ingestion`,
`services.agent`, etc.) are set once at startup and read at request time by every
router via `Depends()`. Also owns the rate limiter, PostHog helpers, and all
`get_*` dependency provider functions. Routers import from here, never from
`main.py` — this prevents circular imports.

**`backend/main.py`** — Now intentionally thin (~110 lines). Two things it still owns:

1. **Lifespan**: services are expensive to initialise (model loads, connection
   pools). `lifespan()` creates them once at startup, sets them on the `services`
   container, and wires MCP. Not created per-request.

2. **SSE streaming** (in routers): instead of waiting for the full LLM response,
   the backend streams tokens as they arrive:
   ```python
   async def event_stream():
       async for token in service.stream(...):
           yield f"data: {json.dumps({'token': token})}\n\n"
   return StreamingResponse(event_stream(), media_type="text/event-stream")
   ```
   The frontend reads this token-by-token — that's why text appears word by word.

**`backend/routers/`** — Five files, each owning one domain. Adding a new route
means opening one focused file rather than searching a 1000-line `main.py`.

**`backend/mcp_server.py`** — Defines all the tools the agent can use, using the
`@mcp.tool()` decorator. Each tool is a Python function with a docstring and typed
arguments — FastMCP reads these to auto-generate the JSON schemas that get sent
to the LLM. Adding a new tool is just writing a new function here.

**`backend/mcp_client.py`** — The other side of MCP. Connects to `/mcp` at
startup, fetches the tool list, and converts them to the format each LLM provider
expects (OpenAI-style `tools` array for Groq/Gemini/Cerebras, Anthropic-style
`tools` for Claude). Bridges the protocol gap between MCP and each provider's API.

---

#### Services (`backend/services/`)

**`ingestion_service.py`** — Orchestrates the full ingestion pipeline. Calls
`repo_fetcher → file_filter → code_chunker → embedder → qdrant_store` in sequence.
Streams progress events back to the frontend (0%… 40%… 80%… done) via SSE so the
user sees a live progress bar instead of a spinner.

**`generation.py`** — Handles all LLM calls. Supports 5 providers (Gemini,
Cerebras, OpenRouter, Groq, Anthropic) with automatic fallback: if the primary
model hits its rate limit, it silently switches to the next one. Both streaming
(for chat responses) and non-streaming (for diagram generation, grading) modes.

**`agent.py`** — The ReAct loop (see the AgentService section above). Builds the
system prompt, sends the question + tool schemas to the LLM, parses the response
to detect tool calls vs final answers, executes tools via MCP, loops. Also collects
sources (which chunks were actually read) and streams structured progress events
so the frontend can show the "thought → tool → result" trace in real time.

**`diagram_service.py`** — Generates the Architecture and Class diagrams. The key
insight is that the graph **structure** (nodes and edges) comes from static analysis
of the AST data stored at ingest time — not from the LLM. The LLM's only job is to
write a one-sentence description for each node. This means the diagrams can't
hallucinate connections that don't exist in the code.

---

#### Frontend (`ui/src/`)

**`api.js`** — All network calls to the backend in one place. The most important
function is `streamQuery()` — it opens an SSE connection and calls callbacks as
events arrive (`onToken`, `onSources`, `onDone`, `onError`). Every other component
just calls these functions; none of them do raw `fetch()` calls themselves.

**`App.jsx`** — The React root. Owns all the shared state: which repo is selected,
the list of chat sessions, messages in the current session, whether the agent is
streaming. When the user submits a question, `handleSubmit` creates a new message
object and calls `streamQuery()` — as tokens arrive, it updates that message
in-place via React state, which is what produces the live streaming effect.

**`components/Sidebar.jsx`** — The left panel. Shows the repo list (click to
switch), session history (stored in localStorage), and the URL ingestion input.
Also handles the mobile drawer (slides in over the content on small screens).

**`components/Message.jsx`** — Renders one chat turn. For RAG answers it shows
the text bubble + source cards. For agent answers it shows the `ToolCallTrace` —
a vertical timeline of every thought and tool call the agent made. This is what
makes the agent's reasoning transparent rather than a black box.

**`components/DiagramView.jsx`** — The tab container for Explore / Architecture /
Class diagrams. Manages which tab is active, fetches diagram data via SSE (so the
loading progress bar works), caches results in localStorage so switching tabs
doesn't re-fetch, and handles the fullscreen toggle.

**`components/ExploreView.jsx`** and **`components/GraphDiagram.jsx`** — Both
render an interactive pan/zoom canvas of nodes and bezier-curve edges. ExploreView
renders the AI-generated concept tour (the "learning path" through the repo).
GraphDiagram renders the architecture and class diagrams. Both use the same
hand-written SVG approach — no third-party graph library — for full visual control.

---

## 6. Architecture Decisions

**Why Qdrant Cloud instead of Chroma or Pinecone?**
Qdrant is the only major vector DB with *native* hybrid search (dense + sparse in one
request). Chroma doesn't support sparse vectors. Pinecone requires a separate sparse
index. Qdrant also has a free 1 GB hosted tier, matching our zero-cost deployment goal.

**Why no LangChain or LlamaIndex?**
These frameworks abstract away the concepts this project is designed to teach. Using
them would mean `chain.run(query)` instead of understanding embeddings, retrieval, and
generation separately. Every component is built from scratch and visible in the code.

**Why MCP instead of calling tools directly?**
MCP means the tools are defined once and usable from any client — our own agent,
Claude Code, Cursor, anything that speaks the protocol. It also enables future tools
without changing the agent code: just add a new `@mcp.tool()` function.

**Why two-part deployment (Vercel + HuggingFace)?**
The React frontend is a static site (just files) — Vercel hosts static files cheaply
on a global CDN, with ~100ms load times worldwide. The Python backend needs a runtime
— HuggingFace Spaces provides free CPU Docker containers. Running both on HF would
mean serving static files from a container, losing Vercel's CDN advantage.

**Why SSE (Server-Sent Events) instead of WebSockets?**
SSE is one-way (server → client), which is all we need for streaming tokens. It's
simpler than WebSockets (just HTTP), works through proxies and CDNs without special
config, and browsers reconnect automatically if the connection drops.

**Why store sessions in localStorage, not the backend?**
Simplicity. No auth system needed — each user's sessions live in their browser. No
database table to maintain. Downside: sessions are per-browser (switching devices
loses history), which is acceptable for a learning/demo project.

---

## 7. Deployment & CI/CD

### Docker

A **Docker container** is a portable, isolated environment. You describe it in a
`Dockerfile` — a recipe that starts from a base image and layers instructions:

```dockerfile
FROM python:3.11-slim          # start from official Python image
RUN pip install -r requirements.txt  # install deps (cached as a layer)
COPY . .                       # copy source code
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Each instruction creates a **layer**. Docker caches layers — if `requirements.txt`
hasn't changed, `pip install` isn't re-run. This makes rebuilds fast.

**Why containers?** Reproducibility. The exact same image runs on your laptop, in CI,
and on HuggingFace. "It works on my machine" stops being a problem.

Key decision in our Dockerfile: **pre-download the re-ranker model during build**.
The cross-encoder (~80MB) is baked into the image layer. Cold starts are instant
instead of downloading it on first request.

### HuggingFace Spaces

HuggingFace Spaces is a hosting platform where each Space is a **git repository**.
When you push code:
1. HF detects the `Dockerfile` at the repo root
2. Builds a Docker image from it
3. Runs the container on their infrastructure
4. Proxies HTTPS traffic to port 7860 (hard requirement)

The URL format: `https://{username}-{space-name}.hf.space`

**Free tier limitations:**
- CPU only (no GPU) — fine for embeddings via API, but local embedding would be slow
- ~16GB RAM — reason we use the Nomic API instead of a local embedding model
- Container sleeps after 48h of inactivity — first request triggers a cold start

### Vercel

Vercel is a CDN-based hosting platform for static sites (no server runtime needed).

```
npm run build → dist/
  index.html
  assets/
    App-abc123.js   (React code, bundled + minified)
    index-def456.css
```

Vercel uploads `dist/` to 100+ global edge nodes. When a user in Tokyo visits
your app, they get assets from a Tokyo edge node, not from a US server.

**VITE_API_URL** is the critical env var. Vite replaces all `import.meta.env.VITE_API_URL`
references with the literal string at build time. The compiled JS bundle contains your
HuggingFace Space URL directly — no runtime config needed.

### GitHub Actions (CI/CD)

**CI/CD** = Continuous Integration / Continuous Deployment. Every push to `main`
triggers automated workflows.

**`ci.yml`** — runs tests and linting on every push. Catches broken code before deploy.

**`deploy.yml`** — runs on every push to `main`, two parallel jobs:

```
push to main
  ├── deploy-backend  → HuggingFace Spaces
  │     git push main → huggingface.co/spaces/umanggarg/cartographer
  │     (HF detects Dockerfile at repo root → rebuilds Docker image)
  │
  └── deploy-frontend → Vercel
        npm ci  (install deps)
        npx vercel --prod  (build + upload to CDN)
```

**Why direct push to HF (no subtree split)?**
The GitHub repo root IS the project — `Dockerfile` sits at the top level. HF Spaces
just needs a git repo with `Dockerfile` at the root, so we push `main` directly.
`git subtree split` would only be needed if the project lived inside a subdirectory
of a larger monorepo.

**Secrets vs Variables** — GitHub Actions has two kinds of stored config:
- **Secrets** (encrypted, never visible in logs): tokens and API keys
- **Variables** (plain text, visible): non-sensitive config like usernames

Set them at: GitHub repo → Settings → Secrets and variables → Actions

| Type | Name | What it is | Where to get it |
|------|------|-----------|-----------------|
| Secret | `HF_TOKEN` | HuggingFace access token | huggingface.co → Settings → Access Tokens → New token (Write scope) |
| Secret | `VERCEL_TOKEN` | Vercel personal access token | vercel.com → Account Settings → Tokens → Create |
| Secret | `VERCEL_ORG_ID` | Your Vercel team/org ID | Run `npx vercel link` inside `ui/` → reads from `.vercel/project.json` |
| Secret | `VERCEL_PROJECT_ID` | Your Vercel project ID | Same `.vercel/project.json` file |
| Variable | `HF_USERNAME` | Your HF username | e.g. `umanggarg` |
| Variable | `HF_SPACE_NAME` | Your HF Space name | e.g. `cartographer` |
| Variable | `HF_SPACE_URL` | Full HF Space URL | e.g. `https://umanggarg-cartographer.hf.space` |

**Why `npx vercel --prod` from the repo root (not `ui/`)?**
The Vercel project has its root directory configured as `ui/` in the Vercel dashboard.
Running the CLI from inside `ui/` would make Vercel look for `ui/ui` — a double-nested
path. Always run `vercel` from the repo root and let the dashboard setting handle the rest.

---

## 8. SSE Streaming — Deep Dive

### What SSE Is

**Server-Sent Events** is an HTTP-based protocol for the server to push a stream of
messages to the browser without the browser polling repeatedly.

The browser opens a single HTTP request, and the server keeps the connection open,
sending messages whenever it has something to say. The browser gets each message as
it arrives — this is how we show LLM tokens appearing word-by-word.

**Compared to WebSockets:** WebSockets are full-duplex (both sides can send at any
time). SSE is half-duplex — server sends, client only receives. For streaming LLM
output, we only ever need server → client, so SSE is simpler and fits on plain HTTP.

**Compared to polling:** The alternative is the browser asking "do you have tokens
yet?" every 500ms. This wastes bandwidth, adds latency, and hammers the server.
SSE inverts it: the server pushes whenever it's ready.

### The Wire Format

SSE is just a text protocol layered on HTTP. The response has:
```
Content-Type: text/event-stream
Cache-Control: no-cache
```

And the body is a stream of **events**, each separated by a blank line:
```
data: {"token": "The"}\n\n
data: {"token": " attention"}\n\n
data: {"token": " mechanism"}\n\n
data: {"type": "done"}\n\n
```

Rules:
- Each message starts with `data: ` (or `event: ` for named events)
- A blank line (`\n\n`) terminates each message
- The connection stays open until the server closes it or sends `data: [DONE]`
- Comments start with `:` — useful as keepalives: `: keep-alive\n\n`

You can also send named events:
```
event: thought\n
data: {"text": "I need to search for the loader..."}\n\n

event: token\n
data: {"text": "The"}\n\n
```

### Backend: Producing SSE in FastAPI

```python
from fastapi.responses import StreamingResponse
import json

async def event_stream():
    # This is an async generator — yields strings
    async for chunk in llm_client.stream(messages):
        token = chunk.choices[0].delta.content or ""
        if token:
            # The exact SSE format: data: <json>\n\n
            yield f"data: {json.dumps({'token': token})}\n\n"
    # Signal the client we're done
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.post("/query")
async def query(request: QueryRequest):
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering in prod
        }
    )
```

`StreamingResponse` with `media_type="text/event-stream"` is all FastAPI needs.
The generator `event_stream()` is lazy — FastAPI pulls one chunk at a time and
writes it to the socket. The connection stays open until the generator is exhausted.

**The `X-Accel-Buffering: no` header** is critical in production. Nginx (which
sits in front of many containers) buffers HTTP responses. Without this header,
nginx accumulates tokens and sends them in batches — you'd see the response appear
all at once instead of token-by-token.

### Frontend: Consuming SSE in React

The browser has a built-in `EventSource` API for SSE, but it only supports `GET`
requests and can't send headers. Since we need to `POST` a request body, we use
`fetch()` with manual stream reading:

```javascript
// In App.jsx handleSubmit()
const response = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, repo }),
    signal: abortController.signal,  // to cancel mid-stream
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // value is a Uint8Array — decode to string
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by \n\n
    const events = buffer.split("\n\n");
    // The last element might be incomplete — keep it in the buffer
    buffer = events.pop();

    for (const event of events) {
        if (!event.startsWith("data: ")) continue;
        const raw = event.slice("data: ".length);
        if (raw === "[DONE]") return;

        try {
            const parsed = JSON.parse(raw);
            if (parsed.token) {
                // Append to the current message in state
                setMessages(prev => {
                    const last = prev[prev.length - 1];
                    return [...prev.slice(0, -1), {
                        ...last,
                        text: last.text + parsed.token,
                    }];
                });
            }
        } catch (e) { /* malformed event, skip */ }
    }
}
```

**Why manual stream reading instead of `EventSource`?**
`EventSource` is simpler but `GET`-only. Our API needs `POST` to send the question
in the body. So we use `fetch()` with `response.body.getReader()` — a `ReadableStream`
that delivers `Uint8Array` chunks as they arrive from the network.

**The buffer trick:** Network chunks don't align with SSE events. A chunk might
contain half an event, or three events. The `buffer` variable accumulates chunks,
splits on `\n\n`, and holds the last (possibly incomplete) segment for the next
iteration.

**Cancellation via `AbortController`:** The stop button calls
`abortController.abort()`, which throws a `DOMException` at the `reader.read()` call.
We catch this and clean up — the backend SSE generator gets a `GeneratorExit` when
the client disconnects.

### Agent SSE: Multiple Event Types

The agent stream is more complex — it sends thoughts, tool calls, tool results,
and final tokens as separate event types, each with different structure:

```
data: {"type": "thought", "text": "I need to find the data loader..."}\n\n
data: {"type": "tool_call", "tool": "search_code", "input": {"query": "data loader"}}\n\n
data: {"type": "tool_result", "tool": "search_code", "result": "..."}\n\n
data: {"type": "token", "text": "The DataLoader class in..."}\n\n
data: {"type": "done"}\n\n
```

The frontend dispatches on `parsed.type` and routes each event to the right UI
component — thoughts into the collapsible `AgentThought`, tool calls into
`AgentStep`, tokens into the final answer text.

---

## 9. Python Async/Await — Deep Dive

### The Problem: Waiting Wastes Time

Traditional (synchronous) code waits at each I/O call:

```python
result = requests.get(url)  # blocks for 200ms while waiting for response
data = db.query(...)        # blocks for 50ms
# only THEN does anything else run
```

If you have 100 simultaneous HTTP requests on your server, and each one blocks for
200ms, you need 100 threads. Threads are expensive (~8MB each), so you hit limits fast.

### The Solution: Coroutines and the Event Loop

Python's `asyncio` solves this with **coroutines** — functions that can pause and
resume. When a coroutine hits an I/O wait, it suspends itself and lets other
coroutines run. One thread can handle thousands of concurrent requests.

```python
# Synchronous — blocks
def fetch_sync(url):
    return requests.get(url).text

# Asynchronous — suspends while waiting, yields control to event loop
async def fetch_async(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # suspends here
    return response.text
```

The `await` keyword says: "start this I/O operation, suspend me, run other things,
resume me when the result is ready."

### The Event Loop

The **event loop** is the scheduler that runs coroutines. It maintains a queue of
ready-to-run coroutines, runs one until it hits an `await`, picks the next ready
coroutine, and so on:

```
Event Loop tick:
  1. Run coroutine A until it hits `await client.get(...)`
  2. Register the network callback: "wake A when response arrives"
  3. Run coroutine B until it hits `await db.query(...)`
  4. Register DB callback: "wake B when result arrives"
  5. [network response arrives for A]
  6. Run coroutine A from where it paused
  ...
```

`asyncio.run(main())` starts the event loop and runs `main()` as the root coroutine.
In FastAPI, uvicorn starts the event loop when the server starts.

### async def, await, and awaitables

```python
# async def defines a coroutine function
# Calling it returns a coroutine object — it doesn't run yet
async def process(chunk):
    result = await embed(chunk)   # await a coroutine
    return result

# You must await a coroutine to run it and get its value
# Outside async context: asyncio.run(process(chunk))
# Inside async context: result = await process(chunk)
```

**What can you await?**
- Other coroutines (async def functions)
- `asyncio.Task` objects
- Any object with an `__await__` method
- Special objects like `asyncio.sleep(n)`, which pause without blocking

### asyncio.gather — Running Things in Parallel

`asyncio.gather(*coroutines)` runs multiple coroutines **concurrently** (not truly
parallel since Python has the GIL, but concurrent in the I/O sense):

```python
# Sequential — total time = 200ms + 300ms + 150ms = 650ms
result1 = await search_code(query1)
result2 = await search_code(query2)
result3 = await search_code(query3)

# Concurrent — total time ≈ max(200, 300, 150) = 300ms
result1, result2, result3 = await asyncio.gather(
    search_code(query1),
    search_code(query2),
    search_code(query3),
)
```

We use this in two places:
1. **Query expansion** — multiple query variants all hit Qdrant simultaneously
2. **Agent parallel tool calls** — when the agent needs data from several independent
   tools (e.g. search for `DataLoader` AND `forward()` at the same time)

The key insight: `gather` starts all coroutines nearly simultaneously. Each one
suspends when it hits its `await` (e.g. waiting for Qdrant to respond). The event
loop runs all of them interleaved — effectively overlapping their wait times.

### When NOT to use async

Async helps with **I/O-bound** work (network calls, database queries). It doesn't
help with **CPU-bound** work (matrix multiplication, image processing) — the GIL
means only one thread runs Python at a time anyway.

For CPU-bound work, use `asyncio.run_in_executor(None, cpu_bound_function)` to run it
in a thread pool, or `multiprocessing` for true parallelism.

The local cross-encoder re-ranker is CPU-bound (PyTorch inference). We run it in
an executor so it doesn't block the event loop:
```python
loop = asyncio.get_event_loop()
scores = await loop.run_in_executor(None, model.predict, pairs)
```

### async generators

An **async generator** is an async function that uses `yield`. FastAPI's
`StreamingResponse` consumes one:

```python
async def event_stream():
    async for chunk in llm.stream(messages):
        yield f"data: {json.dumps(chunk)}\n\n"
    # return implicitly closes the generator
```

FastAPI pulls values from the generator by calling `__anext__()` repeatedly. When the
generator is exhausted (or raises `StopAsyncIteration`), FastAPI closes the response.

---

## 10. React Hooks — Deep Dive

### The Mental Model

React re-renders a component (calls the function again) when its state changes. Hooks
are how you attach stateful behaviour to a function component without making it a
class.

**Key rule:** Hooks must be called unconditionally, in the same order, on every render.
No hooks inside `if` blocks or loops. React tracks them by position in the call stack.

### useState

```javascript
const [value, setValue] = useState(initialValue);
```

`value` is the current state. `setValue` is how you update it — calling it
schedules a re-render with the new value.

```javascript
// Wrong: mutating state directly — React won't see the change
messages.push(newMsg);

// Correct: replace with a new value
setMessages([...messages, newMsg]);

// Functional update: safe when new value depends on old value
setMessages(prev => [...prev, newMsg]);
```

**Functional updates** matter for streaming: if you call `setMessages` 100 times in
quick succession (once per token), using `prev =>` ensures each update builds on the
*latest* state, not the stale closure value from when the effect started.

### useEffect

```javascript
useEffect(() => {
    // runs after render
    fetchRepos();

    return () => {
        // cleanup runs before next effect OR on unmount
        subscription.unsubscribe();
    };
}, [dependency1, dependency2]);
```

**Dependency array controls when the effect re-runs:**
- `[]` — run once after first render (mount)
- `[repo]` — run when `repo` changes
- No array — run after every render (usually a bug)

**Common patterns in this project:**

```javascript
// Scroll to bottom when messages change
useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
}, [messages]);

// Reset state when user switches repos
useEffect(() => {
    setMessages([]);
    setCurrentSession(null);
}, [currentRepo]);

// Fetch initial repo list on mount
useEffect(() => {
    api.listRepos().then(setRepos);
}, []);
```

**The closure trap:** Inside a `useEffect`, you capture the values of variables from
the render. If `messages` changes but you don't include it in the dependency array,
the effect still sees the old `messages`. This is the most common React bug.

### useRef

```javascript
const ref = useRef(initialValue);
// ref.current is the mutable value
```

Unlike state, **changing `ref.current` does not cause a re-render**. Use it for:

1. **DOM references**: `<div ref={bottomRef}>` — `bottomRef.current` is the DOM node
2. **Mutable values that shouldn't trigger renders**: storing the AbortController or
   a streaming reader — updating these shouldn't re-render the UI
3. **Previous values**: `const prevRepo = useRef(currentRepo)` — accessible across
   renders without triggering them

```javascript
// In App.jsx — storing the stream abort function
const stopStreamRef = useRef(null);

// Inside handleSubmit:
const controller = new AbortController();
stopStreamRef.current = () => controller.abort();

// Stop button:
<button onClick={() => stopStreamRef.current?.()}>Stop</button>
```

If `stopStream` were in `useState`, clicking Stop would trigger a re-render, which
is wasted — we only care about calling the function.

### useCallback and useMemo

**useMemo** caches a computed value until dependencies change:
```javascript
const filteredRepos = useMemo(
    () => repos.filter(r => r.includes(search)),
    [repos, search]
);
```

**useCallback** caches a function reference until dependencies change:
```javascript
const handleSubmit = useCallback(
    (question) => { /* ... */ },
    [currentRepo, model]
);
```

These prevent unnecessary re-renders when passing functions as props to child
components. A new function reference on every render would cause child re-renders
even if the function logic didn't change. We use these sparingly — premature
optimization is the enemy.

### Component Lifecycle via Hooks

| Lifecycle event     | Hook equivalent                    |
|---------------------|------------------------------------|
| Mount               | `useEffect(() => {...}, [])`       |
| Update (specific)   | `useEffect(() => {...}, [dep])`    |
| Unmount             | `return () => cleanup()` in effect |
| Before render       | `useMemo` (pure computation)       |

### localStorage and Session Persistence

```javascript
// Save sessions
useEffect(() => {
    if (currentRepo) {
        localStorage.setItem(`sessions_${currentRepo}`, JSON.stringify(sessions));
    }
}, [sessions, currentRepo]);

// Load on mount
const savedSessions = JSON.parse(
    localStorage.getItem(`sessions_${repo}`) || "[]"
);
```

`localStorage` is synchronous (no async needed), persists across page refreshes,
and is scoped to the domain. Limit is ~5MB per origin. We store up to 10 sessions
per repo, with a max message history to stay within limits.

---

## 11. Dimensionality Reduction & UMAP

### Why We Need It

We have 768-dimensional vectors for every code chunk. Humans can't visualise 768
dimensions. The Explore view needs to show *which chunks are semantically similar*
as a 2D graph. Dimensionality reduction compresses 768 → 2 while preserving as much
structure as possible.

### PCA — Principal Component Analysis (baseline)

PCA finds the directions of maximum variance in the data and projects onto them.
It's linear, fast, and preserves global structure (clusters that are far apart
stay far apart).

The first two principal components give us an X-Y coordinate for each point, but
PCA optimises for *variance*, not for preserving *local neighbourhoods*. Points that
are similar in the original space may be far apart in the 2D projection.

### t-SNE (the older approach)

t-SNE (t-distributed Stochastic Neighbour Embedding) optimises a different objective:
preserve the **probability that points would be neighbours** in the original space.
It uses a heavy-tailed t-distribution in 2D to avoid points collapsing to the center.

t-SNE produces beautiful cluster visualisations, but:
- It's slow for large datasets (O(N²) naively)
- It's non-deterministic (different runs give different layouts)
- It doesn't preserve global structure — clusters that are far apart may be placed
  anywhere relative to each other in the 2D output
- You can't add new points to an existing projection

### UMAP — The Modern Approach

**UMAP** (Uniform Manifold Approximation and Projection) is the current state of the
art for dimensionality reduction. It's faster than t-SNE, preserves both local and
global structure, and is more theoretically grounded.

**Core idea:** UMAP models the high-dimensional data as a weighted graph. Each
point is connected to its k-nearest neighbours, with edge weights proportional to
how close they are. UMAP then finds a 2D layout that preserves this graph structure
as faithfully as possible.

**Key parameters:**
- `n_neighbors` — how many neighbours to consider for each point. Small value (2–5)
  focuses on fine local structure. Large value (50+) captures broader topology.
- `min_dist` — minimum distance between points in 2D. Small value (0.1) lets
  clusters pack tightly; large value (0.5) spreads points more evenly.
- `metric` — distance function in the original space. For embeddings, `cosine` is
  better than `euclidean` because cosine ignores magnitude (we care about direction).

```python
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
embeddings_2d = reducer.fit_transform(embeddings_768d)
# embeddings_2d.shape == (N, 2)
```

**Why UMAP over t-SNE for code?**
- 10x faster on large codebases
- Preserves global structure — similar modules appear in the same region of the plot
- `transform()` method lets you add new points without recomputing the whole projection
- Deterministic with `random_state`

### In the Explore View

The Explore view doesn't actually compute UMAP in real-time (that would be slow on
the free tier). Instead, it uses the LLM to generate a *conceptual* graph based on
chunk metadata (filenames, function names, `calls` fields, `imports` fields) — not
actual vector positions.

The LLM returns structured JSON:
```json
{
  "nodes": [
    {"id": "ingestion", "label": "Ingestion Pipeline", "description": "..."},
    {"id": "retrieval", "label": "Retrieval Service", "description": "..."}
  ],
  "edges": [
    {"from": "ingestion", "to": "retrieval", "label": "feeds"}
  ]
}
```

D3's force-directed layout (`d3.forceSimulation`) then positions these nodes — it
applies physics (repulsion between nodes, attraction along edges) until the graph
reaches equilibrium. It's not UMAP, but it's visually similar and much faster.

**When UMAP would be used:** If you wanted to show *embedding similarity* directly
(which chunks are most alike semantically), you'd compute UMAP on the stored vectors.
A future enhancement could render the actual embedding space with UMAP.

---

## 12. LLM Features: Prompt Caching, Structured Output, System Prompts

### Prompt Caching (Anthropic)

**The problem:** Every API call re-processes the entire prompt from scratch. For an
agent with a long system prompt and growing conversation history, this means re-reading
hundreds of tokens repeatedly — wasted compute (and money).

**Prompt caching** lets you "checkpoint" part of the prompt. Anthropic caches it on
their servers for a few minutes (exact TTL varies). Subsequent requests that start with
the same cached prefix skip the re-processing — dramatically faster and cheaper.

```python
# Anthropic API — cache_control marks what to cache
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": LONG_SYSTEM_CONTEXT,  # This gets cached
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": user_question  # This changes each call
            }
        ]
    }
]
```

**Best for:**
- Long system prompts that don't change (agent instructions, few-shot examples)
- Conversation history in a long chat (cache up to the latest turn)
- Large document ingested once and queried multiple times

**Cost:** Cached reads are ~10x cheaper than non-cached. Cache writes cost the same
as normal. Break-even is if the same prefix is used 2+ times.

**Where we'd add it:** The agent's system prompt is long (tool descriptions, behavior
guidelines) and identical across all agent calls. Caching it would reduce cost
significantly in production. We haven't added it yet (Anthropic isn't in the primary
provider cascade) but it's a straightforward addition.

### Structured JSON Output

LLMs can return free text, but structured data (JSON) is often what code needs.
Three approaches, from least to most reliable:

**1. Prompt engineering (worst):**
```python
prompt = "Return your answer as JSON like: {\"nodes\": [...], \"edges\": [...]}"
# Model may still return markdown, preamble, or invalid JSON
```

**2. json_mode / response_format (better):**
```python
# OpenAI-compatible APIs
response = client.chat.completions.create(
    model="...",
    messages=messages,
    response_format={"type": "json_object"},  # forces valid JSON output
)
```
The model is guaranteed to return valid JSON, but you still define the schema in the
prompt and it might not follow your structure perfectly.

**3. Tool calling as structured output (best):**
Define a "tool" with the exact schema you want. Ask the model to "call" it with the
structured data. Since tool arguments are JSON, this is guaranteed-valid AND
schema-constrained:
```python
tools = [{
    "name": "return_concept_map",
    "description": "Return the concept map",
    "input_schema": {
        "type": "object",
        "properties": {
            "nodes": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
            "edges": {"type": "array", "items": {"$ref": "#/$defs/Edge"}},
        }
    }
}]
# Force the model to call this tool
tool_choice = {"type": "tool", "name": "return_concept_map"}
```

We use json_mode for the Explore view concept map and query classification. Tool
calling as structured output is used in the agent loop.

### System Prompt Engineering

The system prompt is the instruction set that shapes the model's behavior. For an
agent, it's the most important part of the whole system. Poor system prompts cause
the agent to wander, call tools unnecessarily, or give up too early.

**Anatomy of a good agent system prompt:**

```
1. ROLE — Who the agent is and what it's optimizing for
   "You are a code investigator. Your goal is to answer questions about
   the indexed repository with precise, citation-backed answers."

2. AVAILABLE TOOLS — Explicit list of what each tool does and when to use it
   "- search_code(query): Use for conceptual questions about how things work
    - search_symbol(name): Use when you know the exact function/class name
    - read_file(path): Use to see a full file when you need all the context"

3. STRATEGY — How to approach investigation
   "Start with broad searches, narrow with specific ones. Use trace_calls
   to understand call chains. If you find the answer, stop — don't over-investigate."

4. OUTPUT FORMAT — What the final answer should look like
   "When you have enough information, provide a detailed answer with:
   - Code snippets from the tools
   - File citations in the format [filepath:line_range]"

5. GUARDRAILS — What NOT to do
   "Do not guess. If you can't find the answer, say so and explain what
   you searched. Do not call the same tool with the same input twice."
```

**The "note" pattern for working memory:** The system prompt must explicitly tell the
model to use `note` to record findings before switching to a new investigation. Without
this instruction, the model forgets earlier findings when its context grows long:
```
"When you find a key piece of information, call note(key, value) to store it.
Before synthesizing your final answer, call recall_notes() to retrieve all
stored findings."
```

**Temperature:** For code investigation, use a low temperature (0.1–0.3). You want
deterministic, literal answers — not creative ones. High temperature causes the agent
to "imagine" code that doesn't exist.

---

## 13. Production Patterns: Rate Limits, Retry, Multi-Provider Cascade

### Rate Limiting

Every API has rate limits. Anthropic, Groq, Cohere all limit requests per minute
(RPM) and tokens per minute (TPM). Exceeding them returns an HTTP 429 response.

**Exponential backoff with jitter:** When you hit a 429, wait and retry. Each retry
waits longer. Add random jitter to prevent all clients from retrying at the same time
(the "thundering herd" problem):

```python
import asyncio, random

async def call_with_retry(fn, *args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await fn(*args)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)  # exponential + jitter
            await asyncio.sleep(wait)
```

Retry delays: attempt 0 → ~1s, attempt 1 → ~2s, attempt 2 → ~4s.

**Headers to check:** APIs often send `Retry-After: 30` in the response header. Use
this exact value instead of guessing:
```python
retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
await asyncio.sleep(retry_after)
```

### API Request Size Limits (413 Errors)

HTTP 413 means "Request Entity Too Large" — you sent a body the server won't accept.
This happens when large repos produce many chunks with large metadata payloads.

Causes in this project:
1. **Embedding batch too large**: Nomic API accepts ~100 texts per call. If you send
   500 chunks at once, you get a 413 (or timeout).
2. **Context window overflow**: Packing too many retrieved chunks into the LLM prompt
   exceeds the model's token limit. The API returns a 400/413.
3. **Query with huge repo map**: The repo map (per-file summaries) can be very large
   for repos with 1000+ files. Sending the full map in a single request fails.

Fixes:
```python
# Batch embedding calls
EMBED_BATCH_SIZE = 50
for i in range(0, len(chunks), EMBED_BATCH_SIZE):
    batch = chunks[i:i + EMBED_BATCH_SIZE]
    vectors = await embedder.embed_batch([c.text for c in batch])

# Truncate chunks before embedding if they exceed token limit
MAX_CHUNK_TOKENS = 8000
if count_tokens(chunk.text) > MAX_CHUNK_TOKENS:
    chunk.text = truncate_to_tokens(chunk.text, MAX_CHUNK_TOKENS)

# Limit retrieved context to max tokens
MAX_CONTEXT_TOKENS = 20000
chunks = truncate_context(ranked_chunks, max_tokens=MAX_CONTEXT_TOKENS)
```

### Multi-Provider LLM Cascade

Relying on a single LLM API is fragile: APIs go down, hit rate limits, or have
context windows too small for some requests. The cascade pattern tries providers
in order, falling back to the next on any failure:

```python
PROVIDERS = [
    {"name": "cerebras",   "model": "llama-3.3-70b"},
    {"name": "groq",       "model": "llama-3.3-70b"},
    {"name": "gemini",     "model": "gemini-2.0-flash"},
    {"name": "openrouter", "model": "google/gemma-4-27b"},
    {"name": "anthropic",  "model": "claude-3-5-haiku-20241022"},
]

async def generate_with_fallback(messages, **kwargs):
    errors = []
    for provider in PROVIDERS:
        if not has_api_key(provider["name"]):
            continue
        try:
            return await call_provider(provider, messages, **kwargs)
        except (RateLimitError, APIError, TimeoutError) as e:
            errors.append(f"{provider['name']}: {e}")
            continue
    raise AllProvidersFailedError(errors)
```

**Provider selection considerations:**
- **Cerebras** is first because it's the fastest (2600 tok/s — near-instant responses).
  The model quality (llama-3.3-70b) is excellent for code.
- **Groq** is second — also fast (800 tok/s) with the same model class.
- **Gemini** is third — slower but has a large context window (1M tokens) — good
  fallback for requests that exceed smaller context limits.
- **Anthropic** is last — highest quality but most expensive, used only when everything
  else fails.

**Streaming with cascade:** Each provider has different streaming APIs. We abstract
over them:
```python
async def stream_tokens(provider, messages):
    if provider == "cerebras":
        async for chunk in cerebras_client.stream(messages):
            yield chunk.choices[0].delta.content or ""
    elif provider == "gemini":
        async for chunk in gemini_client.stream(messages):
            yield chunk.text
    # ... etc
```

The cascade is transparent to the user — the SSE stream starts after the first
provider responds. If Cerebras is down and Groq takes over, the user just sees
a slightly longer first-token delay.

---

## 14. Agent Deep Dive: Working Memory, Parallel Execution, Streaming Thoughts

### The Agent Loop in Detail

The agent in `backend/services/agent.py` runs a ReAct loop. Here's the full cycle:

```python
async def run(self, question: str, repo: str) -> AsyncIterator[AgentEvent]:
    messages = [{"role": "user", "content": question}]
    max_iterations = 10  # prevent infinite loops

    for i in range(max_iterations):
        # 1. Call LLM with tool schemas
        response = await self.llm.complete(
            system=AGENT_SYSTEM_PROMPT,
            messages=messages,
            tools=self.tool_schemas,  # from MCP server
        )

        # 2. Did the LLM produce a thought?
        if response.text:
            yield AgentEvent(type="thought", text=response.text)
            messages.append({"role": "assistant", "content": response.text})

        # 3. Did the LLM call a tool?
        if response.tool_calls:
            # Parallel execution if multiple tools called at once
            tool_results = await asyncio.gather(*[
                self.execute_tool(tc) for tc in response.tool_calls
            ])

            for tc, result in zip(response.tool_calls, tool_results):
                yield AgentEvent(type="tool_call", tool=tc.name, input=tc.input)
                yield AgentEvent(type="tool_result", tool=tc.name, result=result)
                # Add to conversation history so LLM sees the result
                messages.append({
                    "role": "tool",
                    "tool_use_id": tc.id,
                    "content": str(result),
                })

        else:
            # No tool call = LLM is done, return final answer
            yield AgentEvent(type="done", text=response.text)
            return

    yield AgentEvent(type="error", text="Max iterations reached")
```

### Working Memory: note and recall_notes

The agent's context window grows with each iteration — thoughts, tool calls, results.
For long investigations (10+ steps), older findings get pushed far back and the model
may "forget" them or repeat work.

**Working memory** (`note`/`recall_notes`) gives the agent a persistent key-value
store it can write to and read from at will:

```python
# Agent calls: note("loader_class", "DataLoader in ingestion/repo_fetcher.py:45")
# Agent calls: note("forward_fn", "Model.forward() in backend/model.py:112")

# Later, before answering:
# Agent calls: recall_notes()
# Returns: {"loader_class": "DataLoader...", "forward_fn": "Model.forward..."}
```

**Why not rely on conversation history?** Two reasons:
1. The model's attention degrades for content that appeared many turns ago
2. Key facts get buried in verbose tool results — a summary note is more salient

**Implementation in `mcp_server.py`:**
```python
_notes: dict[str, str] = {}  # module-level, cleared per session

@mcp.tool()
def note(key: str, value: str) -> str:
    """Store a fact in working memory"""
    _notes[key] = value
    return f"Stored: {key}"

@mcp.tool()
def recall_notes() -> str:
    """Retrieve all stored notes"""
    if not _notes:
        return "No notes stored."
    return "\n".join(f"{k}: {v}" for k, v in _notes.items())
```

The notes are per-session — cleared when a new agent session starts. This ensures
one user's investigation doesn't contaminate another's.

### Parallel Tool Execution

When the LLM's response contains multiple tool calls, we fire them concurrently:

```python
# Parallel — fires all tools at once
tool_results = await asyncio.gather(*[
    self.mcp_client.call_tool(tc.name, tc.input)
    for tc in response.tool_calls
])
```

This is most effective when the agent reasons: "I need to check both the loader
AND the model — let me search for both simultaneously." A well-prompted agent does
this naturally. Without `gather`, sequential calls to two independent tools would
take 2x the wall time.

**When the LLM decides to parallelize:** Some LLMs (Claude, GPT-4) can return
multiple tool calls in a single response when they recognise that the calls are
independent. Others always call tools one at a time. This is a model-level behaviour,
not something you control in the API.

### Streaming Thoughts to the UI

Each step of the agent loop emits `AgentEvent` objects. These are converted to SSE
events and streamed to the frontend in real time:

```python
# In the SSE generator
async for event in agent.run(question, repo):
    if event.type == "thought":
        yield f"data: {json.dumps({'type': 'thought', 'text': event.text})}\n\n"
    elif event.type == "tool_call":
        yield f"data: {json.dumps({'type': 'tool_call', 'tool': event.tool, 'input': event.input})}\n\n"
    elif event.type == "token":
        yield f"data: {json.dumps({'type': 'token', 'text': event.text})}\n\n"
    elif event.type == "done":
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
```

**Collapsible thought UI:** In `Message.jsx`, each thought is rendered as an
`AgentThought` component. It receives an `isActive` prop — true only for the most
recent thought while the agent is still running.

- `isActive=true` → shows the full thought text (streaming in real time)
- `isActive=false` → collapses to a 120-character preview, click to expand

This design choice keeps the UI focused on the most recent reasoning while preserving
access to the full investigation trail.

---

## 15. Data Layer: Repo Map & Diagram Generation

### The Repo Map

`backend/services/repo_map_service.py` builds a persistent summary of a repo's
structure. Instead of re-reading all files on every Explore request, the repo map
is computed once and stored (typically cached in memory or written to a lightweight
store).

**What's in a repo map entry:**
```python
{
    "filepath": "ingestion/code_chunker.py",
    "language": "python",
    "num_chunks": 12,
    "functions": ["parse_python", "parse_js", "chunk_file", "walk_ast"],
    "classes": ["CodeChunker"],
    "imports": ["tree_sitter", "pathlib", "typing"],
    "base_classes": [],
    "summary": "Tree-sitter based AST chunker for Python, JS, TS, Go, Rust"
}
```

This is derived entirely from chunk **metadata** (the payload stored in Qdrant) —
no file reading needed. When chunks are ingested, every function, class, import, and
call relationship is stored as payload. The repo map aggregates per file.

**Building the repo map:**
```python
async def build_repo_map(self, repo: str) -> list[FileEntry]:
    # Scroll through all chunks for this repo
    points = await self.qdrant.scroll(
        collection=COLLECTION,
        filter=Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))]),
        with_payload=True,
        limit=10000,
    )

    # Aggregate per file
    files: dict[str, FileEntry] = {}
    for point in points:
        fp = point.payload["filepath"]
        if fp not in files:
            files[fp] = FileEntry(filepath=fp, functions=set(), ...)
        files[fp].functions.update(point.payload.get("calls", []))
        # ...

    return list(files.values())
```

**Why not use GitHub API to read files?** Rate limits. A repo with 500 files would
need 500 API calls to read each file for analysis. The repo map uses data we already
have (Qdrant payloads), making it instant.

### Diagram Generation via SSE

Diagrams are generated by prompting the LLM with a subset of the repo map and
asking it to output Mermaid syntax. The interesting part is how we stream this.

**Two-phase SSE stream:**

```
Phase 1 (progress events):
  data: {"type": "progress", "message": "Analysing module relationships..."}\n\n
  data: {"type": "progress", "message": "Building call graph..."}\n\n

Phase 2 (the actual Mermaid source):
  data: {"type": "token", "text": "graph TD\n"}\n\n
  data: {"type": "token", "text": "  A[ingestion] --> B[Qdrant]"}\n\n
  data: {"type": "done"}\n\n
```

The frontend renders progress messages as a loading indicator, then switches to
accumulating Mermaid tokens once they start arriving. After `done`, it calls
`mermaid.render(diagramSource)` to produce the SVG.

**Mermaid.js** is lazy-loaded — we don't import it at page load (it's ~500KB).
Only when a diagram is requested does `MermaidBlock.jsx` do:
```javascript
const mermaid = await import("mermaid");
await mermaid.default.initialize({ theme: "dark" });
const { svg } = await mermaid.default.render("diagram", source);
diagramRef.current.innerHTML = svg;
```

Three diagram types available:
- **Architecture** — uses `imports` payload field to draw module dependency graph
- **Class hierarchy** — uses `base_classes` field to draw inheritance tree
- **Call graph** — uses `calls` field to draw function dependency graph

All three are generated from Qdrant payload data — no re-reading source files.

### Query Classification

Before retrieval, every question is classified:

```python
CATEGORIES = ["implementation", "architecture", "debugging", "comparison"]

async def classify_query(question: str) -> str:
    prompt = f"""Classify this code question into one of: {CATEGORIES}
Question: {question}
Return just the category word."""
    return await llm.complete(prompt)
```

The classification adjusts the system prompt and retrieval strategy:
- `implementation` → increase code-chunk weight in reranking
- `architecture` → use the repo map as additional context
- `debugging` → add error-handling examples to the prompt
- `comparison` → retrieve from multiple related functions

Classification is shown as a subtle tag in the UI on each response.

### Faithfulness Grading

After generation, a second LLM call scores the answer:

```python
async def grade_faithfulness(question: str, answer: str, sources: list[str]) -> str:
    prompt = f"""Given these source chunks:
{format_sources(sources)}

And this answer:
{answer}

Grade faithfulness: is every claim in the answer supported by the sources?
Return one word: high, medium, or low."""
    return await llm.complete(prompt, temperature=0.0)
```

This is a lightweight **LLM-as-judge** pattern. A separate model call evaluates
the output of the first — cheaper than human evaluation and scales automatically.

Shown in the pipeline provenance bar as `✓ high`, `~ medium`, or `✗ low`.

---

## How to Build This From Scratch

1. **Backend skeleton** — `fastapi`, `uvicorn`, one `POST /ingest` endpoint
2. **GitHub ingestion** — `repo_fetcher.py` using GitHub REST API
3. **File filtering** — explicit allowlist of extensions, blocklist of directories
4. **AST chunking** — install `tree-sitter`, walk function/class nodes
5. **Embeddings** — call Nomic API, get 768-dim vectors
6. **Qdrant** — create a free Cloud cluster, upsert chunks with `qdrant-client`
7. **Basic RAG query** — embed question, search Qdrant, prompt LLM, stream response
8. **React frontend** — Vite + React, `fetch()` to backend, render SSE tokens
9. **Hybrid search** — add sparse (BM25) vectors to Qdrant, enable hybrid mode
10. **Re-ranking** — add Cohere API call after retrieval
11. **Agent mode** — ReAct loop, define tools, wire through MCP
12. **Docker** — write `Dockerfile`, test locally with `docker build && docker run`
13. **HuggingFace Space** — push to HF git remote, verify container starts
14. **Vercel** — `npm create vite`, connect to Vercel, set `VITE_API_URL`
15. **GitHub Actions** — copy `deploy.yml`, set secrets, push to trigger

Each step is independently testable — you don't need the agent to test basic RAG, and
you don't need CI/CD to test the backend.

---

## 16. Qdrant In Depth

### Collection Setup

A Qdrant **collection** is the equivalent of a database table — it holds all your
points (chunks). Unlike a regular table, you must declare the vector configuration
upfront because Qdrant pre-allocates index structures at creation time.

```python
client.create_collection(
    collection_name="cartographer_nomic",
    vectors_config={
        # Dense vector field — 768-dim, cosine distance
        "code": VectorParams(size=768, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        # Sparse vector field — BM25, no fixed dimension (sparse by definition)
        "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=False))
    },
)
```

**Why cosine distance?** Cosine similarity measures the angle between two vectors,
ignoring their magnitude. Embedding models output L2-normalised vectors (unit length),
so magnitude carries no information — only direction matters. Cosine is the right
metric. Euclidean distance would treat longer vectors as farther away, which is wrong.

**Why `on_disk=False` for sparse?** Keeps the BM25 index in RAM for fast lookups.
The sparse index is small (just term frequencies, not float arrays) so RAM cost is low.

### Point Structure

Every stored item is a **point**:
```python
PointStruct(
    id="a3f2b1...",          # MD5 hash of "repo::filepath::start_line" — stable ID
    vector={
        "code": [0.12, -0.44, ...],   # 768 floats from Nomic
        "bm25": SparseVector(         # term frequencies for BM25
            indices=[142, 891, 2044],
            values=[0.33, 0.5, 0.17]
        ),
    },
    payload={                # all metadata — filterable and retrievable
        "text":       "def backward(self): ...",
        "repo":       "karpathy/micrograd",
        "filepath":   "micrograd/engine.py",
        "language":   "python",
        "chunk_type": "function",
        "name":       "backward",
        "start_line": 45,
        "end_line":   62,
        "calls":      ["topological_sort", "reversed"],
        "imports":    [],
        "base_classes": [],
    }
)
```

**Stable IDs via hashing:** `MD5("repo::filepath::start_line")` → same chunk always
gets the same ID. Re-ingesting a repo is safe — Qdrant's `upsert` overwrites existing
points rather than creating duplicates.

**BM25 sparse vector:** We compute term frequencies from the chunk text and pass
them as `SparseVector(indices, values)`. Qdrant stores these and uses them for BM25
keyword search. `indices` are vocabulary token IDs (hashed words), `values` are their
TF weights:
```python
def _text_to_sparse(text: str) -> SparseVector:
    tokens = text.lower().split()
    freq = {}
    for t in tokens:
        freq[hash(t) % 2**20] = freq.get(hash(t) % 2**20, 0) + 1
    total = sum(freq.values())
    return SparseVector(
        indices=list(freq.keys()),
        values=[v / total for v in freq.values()]  # normalised TF
    )
```

### Hybrid Search — The Actual API Call

```python
results = client.query_points(
    collection_name="cartographer_nomic",
    prefetch=[
        # First: get top-48 semantic candidates
        Prefetch(query=dense_vector, using="code", limit=48),
        # Second: get top-48 keyword candidates
        Prefetch(query=sparse_vector, using="bm25", limit=48),
    ],
    # Then: fuse both lists with RRF on the server
    query=FusionQuery(fusion=Fusion.RRF),
    limit=24,          # return top-24 fused results
    with_payload=True, # include all metadata in response
)
```

This is **one network round-trip**. Qdrant fetches from both indices internally and
returns the fused result. The `prefetch` list runs in parallel on the server side.

**Why prefetch 2× the final limit?** RRF rewards items that rank well in *both*
lists. A chunk ranked 6th semantically AND 6th by keyword would score highest by RRF,
but if we only prefetched the top 24 from each list, we'd miss it.

### Payload Indices

To filter by `repo` or `language` efficiently, we tell Qdrant to index those fields:
```python
client.create_payload_index(
    collection_name="cartographer_nomic",
    field_name="repo",
    field_schema=PayloadSchemaType.KEYWORD,
)
```

Without this, filtering scans every point. With it, Qdrant uses an inverted index —
`WHERE repo = 'karpathy/micrograd'` is O(1) lookup, not O(N) scan.

---

## 17. FastAPI Lifespan and Dependency Injection

### The Problem

Services like the Qdrant client, embedding model, and LLM client are expensive to
create: they open network connections, load models into RAM, and validate API keys.
You don't want to recreate them on every HTTP request — that would be 500ms of
overhead per call. You want to create them once and reuse them everywhere.

### Lifespan: Creating Services Once at Startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP (runs once before first request) ──
    embedder = Embedder()                    # loads embedding model / connects API
    qdrant   = QdrantStore()                 # opens Qdrant connection pool
    reranker = Reranker()                    # loads cross-encoder (80MB)
    gen      = GenerationService()           # sets up LLM client

    # Assign to module-level variables so routes can access them
    global _ingestion_service, _retrieval_service, _generation_service
    _ingestion_service  = IngestionService(store=qdrant, embedder=embedder, gen=gen)
    _retrieval_service  = RetrievalService(embedder=embedder, store=qdrant, gen=gen)
    _generation_service = gen

    yield  # ← server runs here, handling requests

    # ── SHUTDOWN (runs after last request) ──
    # cleanup connections, save state, etc.
```

`@asynccontextmanager` turns the function into a context manager. `yield` is the
pivot point: everything before is startup, everything after is shutdown. FastAPI
calls this once when the process starts.

**Key design:** all services share the same `qdrant` and `embedder` instances.
This prevents opening multiple connection pools and loading the embedding model twice.

### Depends(): Injecting Services into Routes

```python
# Step 1: define a getter function
def get_retrieval_service() -> RetrievalService:
    if _retrieval_service is None:
        raise RuntimeError("Service not initialised")
    return _retrieval_service

# Step 2: declare it as a dependency in the route
@app.post("/search")
async def search(
    request: SearchRequest,
    svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
):
    results = await svc.search(request.query, request.repo)
    return results
```

`Depends(get_retrieval_service)` tells FastAPI: "before calling `search()`, call
`get_retrieval_service()` and inject its return value as `svc`."

**Why not just use the global `_retrieval_service` directly?**
Two reasons:
1. `Depends` makes the dependency explicit and testable — in tests you can override
   it with a mock: `app.dependency_overrides[get_retrieval_service] = lambda: FakeSvc()`
2. If the service isn't initialised (missing env vars), `Depends` raises the error
   at the FastAPI layer before any business logic runs — clean error handling

**CORS** (Cross-Origin Resource Sharing): browsers block requests to a different
domain by default. The frontend at `vercel.app` calling the backend at `hf.space`
is a cross-origin request. `CORSMiddleware` adds `Access-Control-Allow-Origin` headers
to responses, telling the browser "yes, this other domain is allowed to read my responses."

---

## 18. AST Chunking and tree-sitter

### Python AST (stdlib)

Python has a built-in `ast` module — no installation needed. It parses Python source
into a tree of node objects:

```python
import ast

source = """
def train(model, data):
    loss = model(data)
    loss.backward()
"""

tree = ast.parse(source)
# tree is an ast.Module with a body list
# body[0] is an ast.FunctionDef for 'train'
# body[0].body contains the function's statements
```

We walk this tree to extract functions and classes:
```python
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        chunk_text = ast.get_source_segment(source, node)
        start_line = node.lineno
        end_line   = node.end_lineno
```

**Why AST and not regex?** Regex can find `def ` but it can't tell you where the
function ends — Python's indentation rules are context-sensitive. The AST parser
handles all edge cases: nested functions, decorators, multi-line signatures, etc.

### Call Extraction via AST Visitor

```python
class _CallExtractor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)   # self.embed() → "embed"
        elif isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)     # embed() → "embed"
        self.generic_visit(node)  # continue walking into sub-expressions
```

`ast.NodeVisitor` is the Visitor design pattern — you define what to do when you
encounter a specific node type, and the base class handles traversal. `generic_visit`
continues into child nodes (without it, nested calls would be missed).

The extracted `calls` list is stored in the chunk payload and used to build the
**call graph** in the Architecture diagram.

### Fallback: Character-Window Chunking

For non-Python files (JS, Go, Rust, Markdown), we don't have AST support built in.
Instead we use sliding windows:

```python
WINDOW_CHARS = 1500
OVERLAP_CHARS = 200

for start in range(0, len(text), WINDOW_CHARS - OVERLAP_CHARS):
    chunk = text[start : start + WINDOW_CHARS]
    yield Chunk(text=chunk, chunk_type="text", ...)
```

The 200-char overlap ensures that a function split across two windows appears fully
in at least one of them — at the cost of some duplication in the index.

---

## 19. Prompt Templates

The actual prompts used in the app shape every output. Here are the key ones:

### RAG System Prompt (generation.py)

```
You are a code assistant with access to retrieved source code snippets.
Answer questions based ONLY on the provided source code context.
If the context doesn't contain enough information, say so — do not hallucinate.

When referencing code, cite the source number like:
"According to Source 2 (src/model.py, lines 45–72)..."
Format code in markdown code blocks with the appropriate language tag.
```

Two variants: **technical** (precise, show signatures, be brief) and **creative**
(use analogies, explain accessibly). The `classify_query` function picks between
them based on signal words in the question:

- Creative signals: "explain", "intuitively", "why", "analogy", "simply" (high weight)
- Technical signals: "implement", "signature", "return type", "step by step" (high weight)

### Agent System Prompt (agent.py)

The agent system prompt is long and structured. Key sections:

**1. REPO MAP** — tells the agent to read the ╔══ REPO MAP ══╗ block before
anything else. The repo map is a condensed per-file summary injected into the
user message. This saves 1–2 tool calls for orientation.

**2. PLAN BEFORE ACTING** — the agent must write:
```
<plan>
Goal: [what I need to find]
Search 1: [first tool + query]
Search 2: [second tool + query, if needed]
</plan>
```
This appears as the first thought bubble. It forces the model to think before
acting — prevents "search everything" behaviour.

**3. WORKING MEMORY** — explicit instruction to use `note()` immediately on
each discovery and `recall_notes()` before the final answer. Without this
instruction, the model ignores these tools.

**4. TOOL SELECTION GUIDE** — a table of when to use each tool. Prevents
the model from using `search_code` when `search_symbol` would be faster,
or using `list_files` when the repo map already has the layout.

**5. RULES** — hard constraints: always cite file + line, stop after 3 failed
searches, group related searches into one turn (they run in parallel), never
guess what code does — read it.

### Faithfulness Grading Prompt (generation.py)

```
RAG grader. Does the answer match the sources? Return ONLY JSON.
```
With this system prompt and user message:
```
Sources: [file headers and snippets]
Question: [original question]
Answer: [generated answer]
Grade faithfulness. Return: {"faithful": bool, "confidence": "high|medium|low", "note": str}
```

Why just source headers (not full text)? Truncating to headers avoids false "low"
grades — the grader would say "this claim isn't in the sources" because the full
source text was cut off before that section.

---

## 20. D3.js Force Simulation (Explore View)

### What D3 Is

**D3.js** (Data-Driven Documents) is a JavaScript library for binding data to DOM
elements and animating them. It's not a charting library — it's a toolkit for
building any data visualisation from scratch.

The Explore view uses D3's **force simulation** to position concept nodes.

### How Force Simulation Works

A force simulation is a physics engine in the browser:
- Every **node** has a position (x, y) and velocity (vx, vy)
- **Forces** push and pull nodes each "tick" (simulation step)
- After ~300 ticks the simulation reaches equilibrium and stops

```javascript
const simulation = d3.forceSimulation(nodes)
    .force("link",    d3.forceLink(edges).id(d => d.id).distance(120))
    .force("charge",  d3.forceManyBody().strength(-400))  // repulsion
    .force("center",  d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(60));  // prevent overlap
```

Forces used:
- **forceLink**: pulls connected nodes toward each other (edges as springs)
- **forceManyBody** (negative): repels all nodes from each other — prevents collapse
- **forceCenter**: pulls all nodes toward the canvas center — prevents drift
- **forceCollide**: prevents nodes from overlapping (circle radius = 60px)

### The Tick Function

On each simulation tick, D3 updates node positions and you re-render:

```javascript
simulation.on("tick", () => {
    // Update node positions
    nodeElements
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

    // Update edge positions
    linkElements
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
});
```

After ~300 ticks the `end` event fires and the simulation stops updating. The result
is a graph where related nodes cluster together naturally.

### Our Explore View: Topological Layout (Not Force)

The Explore view uses a **topological sort layout** rather than force simulation,
because concept cards have a defined reading order ("understand A before B"):

1. Assign each concept a column depth via topological sort (longest path from root)
2. Sort within each column by `reading_order`
3. Center columns vertically
4. Draw Bézier arrows between connected cards

This produces a directed left-to-right flow — clearer for teaching than a force
graph where nodes float to random positions. Force simulation is used for the
Architecture diagram (import relationships) where there's no natural reading order.

### Drag and Pan

```javascript
const drag = d3.drag()
    .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;  // fix position while dragging
        d.fy = d.y;
    })
    .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
    })
    .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;  // release — let physics take over again
        d.fy = null;
    });
```

`fx`/`fy` are "fixed" coordinates — setting them pins a node in place and overrides
the physics. Releasing them (`null`) lets the simulation resume.

---

## 21. Agent-Based Tour Generation — Why Agents Beat One-Shot

### The Problem with One-Shot

The original tour generation sent 50 ranked code chunks (about 35,000 characters) to the
LLM in a single call and asked it to simultaneously:

1. Understand the codebase architecture
2. Trace the execution pipeline
3. Identify non-obvious design decisions
4. Format everything as valid JSON with a proper dependency graph

This is cognitively equivalent to handing a new engineer a random pile of printouts and
asking them to immediately write a teaching guide. They'll produce something plausible-sounding
but shallow — pattern-matching against surface-level signals rather than tracing the actual flow.

The result: concepts that are individually reasonable but miss the system-level pipeline.
Dependency graphs guessed from naming conventions rather than traced from imports. The
foundational concept (the pipeline overview) often absent entirely.

### The Three-Phase Agent Pattern

The agent separates understanding from formatting into three focused phases:

```
Phase 1 — MAP      (analogous to architecture_overview)
  Input:  Module-level chunks from entry files — imports + calls visible, no implementation
  Task:   "What is the main pipeline and which files own each stage?"
  Output: { entry_file, pipeline_stages: [{name, file, key_aspect}] }
  Why:    Module chunks reveal the call graph without implementation noise. The LLM
          traces REAL imports and REAL function calls — not guessing from filenames.

Phase 2 — INVESTIGATE  (agentic ReAct loop, once per stage — up to 6 rounds each)
  Input:  Stage name + primary file + pipeline context
  Tools:  read_file, search_symbol, find_callers, trace_calls
  Task:   "What is the non-obvious design decision in this stage? Follow the evidence."
  Output: { name, subtitle, insight, key_functions, naive_rejected, gaps }
  Why:    A design decision often spans multiple files. One-shot Phase 2 could only see
          pre-selected chunks from one file. The ReAct loop lets the agent trace callers
          (how is this used?), follow the call graph (what does it depend on?), and find
          the exact function that implements the decision — stopping only when it has
          verbatim evidence, not when a token budget runs out. See Chapter 24.

Phase 3 — SYNTHESIZE
  Input:  Pipeline map + per-stage insights (structured findings — NO raw code)
  Task:   "Convert this traced understanding into tour JSON."
  Output: Full tour JSON with proper dependency tree
  Why:    Phase 3 only has to format and assign relationships — the understanding
          is already done. Separation of concerns: no LLM call does two hard things.
```

This mirrors exactly how a good engineer reads an unfamiliar codebase: high-level overview
first, then targeted deep-dives into each component, then consolidation into a mental model.

### The Connection to Claude Code's Prompts

Claude Code's internal prompt library includes structured investigation prompts:

| Claude Code prompt | TourAgent phase |
|---|---|
| `architecture_overview` — "Map the main components and how they connect" | Phase 1 (Map) |
| `explain_tool` — "What does this tool do and how does it work?" | Phase 2 (Investigate) |
| `how_does_it_work` — "Trace the execution for this operation" | Phase 2 (Investigate) |
| `compare_tools` — "How do these two approaches differ?" | Embedded in naive_rejected |

The insight is that **structured prompts with scoped context** consistently outperform
a single prompt with everything. The LLM focuses better when it has one clear job.

### Why Context Scoping Matters

Consider Phase 2 investigating `retrieval/hybrid_search.py`. It receives:
- The pipeline context: "This file is stage 3 of 5, responsible for hybrid dense+sparse search"
- All chunks from `hybrid_search.py` only

Without context scoping, the LLM would see 50 chunks from 15 different files and might
focus on the wrong things. With context scoping, it goes deep on one file knowing exactly
why that file matters. This produces insights like:

> "The naive approach embeds the question directly. HyDE generates a *hypothetical code
> snippet* first, then embeds that — searching code-space-to-code-space rather than
> query-space-to-code-space. Direct question embedding consistently misses exact-match
> code because questions and code live in different regions of embedding space."

This is the kind of non-obvious insight that makes a tour genuinely educational rather
than a list of class names.

### Trade-offs

| | One-shot | Agent (3-phase, Phase 1+2 agentic) |
|---|---|---|
| LLM calls | 1 | 10–30 (ReAct rounds + synthesize) |
| Wall-clock time | ~10–20s | ~60–120s |
| Insight depth | Surface-level (pattern matching) | Grounded (traced across multiple files) |
| Dependency graph | Guessed from naming | Traced from imports/calls |
| Context per call | ~35,000 chars (overwhelming) | ~2,000–5,000 chars (focused) |
| Key function names | Sometimes hallucinated | Verbatim from code (copied from RESULT blocks) |
| Failure mode | JSON truncation from token pressure | One phase can fail, others succeed |

The agent is slower but the output quality is categorically better. For a **learning tool**
where the goal is teaching — not just listing — quality wins over speed.

### Live Agent Trace Panel

The UI shows a real-time log of each phase as it runs:

- `info` — chunk/file counts after loading
- `thinking` — LLM is reasoning (mapping or synthesizing)
- `react` — one ReAct loop round: shows THINK text + TOOL call (Phase 1 and Phase 2)
- `found` — pipeline stages discovered, shown as pills
- `file` — currently investigating a specific file
- `finding` — key insight extracted from a stage

This transparency serves a dual purpose: users understand why specific concepts appear
in the tour (because those files were investigated, not randomly selected), and it makes
the agent's reasoning process visible — which is the entire point of Cartographer as
a learning tool.

### Implementation: Generator Pattern for SSE

The `TourAgent.build()` method is a **Python generator** — it yields progress events
throughout execution rather than blocking until complete:

```python
def build(self, repo: str) -> Generator[dict, None, None]:
    yield {"stage": "mapping", "progress": 0.05, "message": "Loading…"}

    pipeline_map = self._phase_map(repo)   # LLM call #1

    yield {"stage": "mapping", "progress": 0.25,
           "trace": {"type": "found", "stages": [s["name"] for s in stages]}}

    for i, stage in enumerate(stages):
        yield {"stage": "investigating", "progress": 0.25 + i * step, ...}
        insight = self._phase_investigate(repo, stage, pipeline_context)
        yield {"stage": "investigating", ..., "trace": {"type": "finding", ...}}

    tour = self._phase_synthesize(repo, pipeline_map, insights)
    yield {"stage": "done", "progress": 1.0, **tour}
```

`DiagramService.build_tour_stream()` iterates over this generator and forwards each event
as an SSE message to the frontend. The cache-and-store logic lives in the service, not the
agent — the agent is a pure computation that knows nothing about caching.

The `trace` key in each event is extra-band metadata for the UI log panel. It's stripped
before storing the tour to disk (the cache only needs the tour content, not the trace steps).

### Visual Numbering vs. LLM Reading Order

The LLM assigns `reading_order` (1, 2, 3, …) to concepts. But the topological layout
algorithm places concepts into columns by dependency depth — the LLM's sequential numbering
may not match visual position.

Fix: after computing layout positions, derive visual numbers from actual pixel coordinates:

```javascript
const visualNumber = {};
Object.entries(basePositions)
  .sort(([, a], [, b]) => a.x !== b.x ? a.x - b.x : a.y - b.y)  // left→right, top→bottom
  .forEach(([id], i) => { visualNumber[Number(id)] = i + 1; });
```

The badge on each card shows `visualNumber[concept.id]` — the number that matches where
your eye actually travels across the canvas. `reading_order` is still used for sort-within-column,
but the displayed badge always reflects visual position.

## 22. Quality Patterns: Evaluator-Optimizer, Terse Docs, and Tool Descriptions

This section covers the four quality improvements made across the codebase to push output quality toward 8+/10.

---

### 22.1 The Evaluator-Optimizer Pattern

**Problem:** A single LLM generation pass can't simultaneously do two hard things at once — synthesise a concept tour structure AND enforce naming conventions. When you ask the model to "return concept names as technique descriptions, not file names", it follows the rule most of the time but breaks down on edge cases: short pipeline stages with sparse chunks tend to fall back to their file names.

**Solution:** Separate generation from quality checking. The synthesis call (Phase 3) focuses on structure, dependency logic, and content. A separate cheap validator call then checks a single, easily-verifiable property: "is this concept name a technique or a file/function name?"

```
Phase 3: Synthesise       →  {concepts: [...]}
Validator: Check names    →  {status: "ok"} or {status: "fixed", concepts: [...]}
```

The validator receives only the concept names and subtitles — not the full tour JSON, not the raw code. This is intentional: a focused context makes it far more accurate at its specific task than if it were asked to review everything.

**When to use this pattern:**
- You have a clear, verifiable quality criterion (binary pass/fail)
- The criterion is difficult to enforce reliably in a single pass
- The check is cheaper than the original generation (it should be)

**Implementation:** `TourAgent._validate_concepts()` in `backend/services/tour_agent.py`

---

### 22.2 Terse, Purpose-Driven Documentation

**Problem:** LLM-generated READMEs default to exhaustive class listings:
```
## Key Components
- `RetrievalService` — handles retrieval
- `EmbeddingService` — handles embedding
- `QdrantStore` — handles vector storage
```

These are what `grep` gives you. A developer reading the README doesn't need a table of contents — they need the mental model that makes the code navigable.

**The better question:** "What would a new engineer misunderstand without this section?"

**Principles applied to `readme_service.py`:**

1. **WHY over WHAT**: "AST chunking splits at function boundaries so every retrieved chunk is self-contained — sliding-window chunking breaks function signatures mid-body" is 10× more useful than "code_chunker.py — chunks code"

2. **Architecture explains splits**: Each module description should say why it's a separate module, not just what it contains. The key question: what would happen if this module were merged with its caller?

3. **One decision per section**: The "Key Design Decisions" section names the non-obvious choices and the alternatives rejected. A developer who reads this knows what to argue about, not just what to accept.

4. **Remove duplicates with the codebase**: Anything that a reader could discover in 30 seconds with `ls` or `grep` is not worth writing.

**Bad section (removed):**
```
## Key Components
[4-6 bullet points. Each: `ClassName` — what it does in one sentence]
```

**Replaced with:**
```
## Key Design Decisions
[2-3 non-obvious choices: why this approach over the simpler alternative]
```

---

### 22.3 Tool Documentation as First-Class Engineering

**Principle:** Tool descriptions are read by the LLM every time it decides what to do. If the description is ambiguous or doesn't define decision boundaries, the model makes poor tool-selection decisions — calling `search_code` when `search_symbol` would be faster, or using `read_file` when `get_file_chunk` was appropriate.

**Three additions to every tool description:**

**1. Decision boundaries** — when to use THIS tool vs similar tools:
```python
"""
PREFER THIS OVER search_code when:
  - You already know the exact function/class name
PREFER search_code OVER THIS when:
  - You have a concept, not a name
"""
```

**2. Mode guide** — when the tool has a `mode` parameter, document each mode's use case explicitly:
```python
"""
MODE GUIDE:
  - 'hybrid'  — best for most queries; semantic + keyword combined
  - 'semantic' — use for conceptual questions
  - 'keyword'  — use for exact identifiers like class names or error types
"""
```

**3. Format constraints** — poka-yoke argument validation: "end_line: keep range ≤ 80 lines for focus" prevents the model from requesting enormous file chunks that bloat context.

**Implementation:** `backend/mcp_server.py` — `search_code`, `find_callers`, `get_file_chunk`

---

### 22.4 Grounding Agents in Repository Purpose

**Problem:** The ReAct agent starts each session with a repo map (file structure, class names) but no statement of intent. It knows "there is a `RetrievalService` in `retrieval_service.py`" but not why the repo exists or what problem it solves. This causes the agent to treat structurally similar repos (two Python projects with a service layer) as interchangeable.

**Solution:** Inject a README summary alongside the repo map at session start. The README anchors every subsequent search in purpose, not just structure.

```
╔══ REPO PURPOSE ══╗
[first ~400 chars of README, markdown headings stripped]
╚══════════════════╝

╔══ REPO MAP: owner/repo (N chunks, M files) ══╗
  Entry files : main.py, server.py
  Key classes : GenerationService, QdrantStore
  ...
╚══════════════════════════════════════════════╝

[user question]
```

**Why 400 chars?** Enough for the project purpose and core differentiators. The full README goes to the tour agent's Phase 0; here we just want 1-3 sentences to set context. Larger extracts start to flood the tool-call context with documentation text.

**Implementation:** `AgentService._get_readme_summary()` in `backend/services/agent.py`

---

### 22.5 Richer Contextual Embeddings

**Problem:** The contextual retrieval prompt was generating one-sentence descriptions. A single sentence often can't capture: what the chunk does, where it fits in the file, and why it matters to the system. A description missing any of these three angles produces embeddings that fail on 1-2 of the 3 retrieval query types.

**Better format (updated `_CONTEXT_SYSTEM`):**
- 2-3 sentences
- Answers: (1) what, (2) where in the file's flow, (3) what breaks without it
- Uses XML `<document>/<chunk>` tags so the model clearly sees the whole-file context vs the specific chunk

The `<document>/<chunk>` format is a deliberate disambiguation: it's visually unambiguous which part is the full file and which is the specific chunk being described. This reduces a class of errors where the model describes the file instead of the specific chunk.

---

## Chapter 23 — Quality Improvements from the Anthropic Cookbook

This chapter covers six fixes (W1–W6) and four missed opportunities (M1–M4) surfaced by reviewing Cartographer against Claude's own codebase and the Anthropic cookbook. All changes target the three areas that most affect perceived quality: tour concept names, retrieval grounding, and agent tool use.

---

### 23.1 Evaluator-Optimizer Loop (W1)

**Pattern:** Anthropic cookbook "evaluator-optimizer loop":
```
generate → evaluate → if PASS: return
                       if FAIL: accumulate corrections → generate again (max N rounds)
```

**Problem:** The original `_validate_concepts()` ran the evaluator once. If the first correction round produced another artifact name, it was returned unchecked.

**Fix:** Added `MAX_ROUNDS = 2` loop. The key property is **memory accumulation** — each round's corrections are fed back into the next round's prompt as "Previous correction attempts that were still flagged — do NOT repeat these." Without this, the model can replace one artifact name with another artifact name.

```python
prior_corrections: list[str] = []
for round_num in range(MAX_ROUNDS):
    prior_context = ""
    if prior_corrections:
        prior_context = "Previously tried corrections that were still flagged:\n" + ...
    # ... run evaluator with prior_context injected ...
    if result["status"] == "ok":
        break   # PASS — done
    # record corrections into prior_corrections for next round
```

**Where:** `tour_agent.py:_validate_concepts()`

---

### 23.2 Anthropic Prompt Caching (W2)

**Pattern:** For contextual retrieval (one LLM call per chunk in a file), all chunks share the same document. Mark the document block as `cache_control: {type: ephemeral}` so the Anthropic KV state is computed once per file and reused for all chunks.

**Cost:** 1 full-cost call per file + N cache-read calls per chunk. Cache-read tokens cost ~10% of regular tokens. For a file with 20 chunks, this is ~90% cost reduction for the cached portion.

**Requirement:** The cached block must be ≥ 1024 tokens to qualify. Source files are typically 200–6000 tokens — most qualify.

```python
def _anthropic_contextualise(client, model, system, doc_text, chunk_question):
    resp = client.messages.create(
        model=model, max_tokens=200, system=system,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"<document>\n{doc_text}\n</document>",
             "cache_control": {"type": "ephemeral"}},         # ← cached
            {"type": "text", "text": "\n\n" + chunk_question},  # ← not cached
        ]}],
    )
```

**Where:** `ingestion_service.py:_anthropic_contextualise()`, called when `provider == "anthropic"`.

---

### 23.3 Terse Contextual Retrieval (W3)

**Problem:** The original `_CONTEXT_SYSTEM` prompt asked for 2-3 sentences covering what/where/failure scenarios. This is good documentation style but bad embedding style — failure scenarios pollute the embedding space with negative language.

**Insight from cookbook:** Contextual enrichment text is prepended to chunks *before embedding*. Its only job is to make the embedding match the queries developers type. Not to explain, not to warn — to situate.

**Fix:** Trimmed `_CONTEXT_SYSTEM` to:
> "Write 1-2 sentences that situate this chunk within the document: name the function/class, state its role in the file's pipeline, and name the key identifier(s) a developer would search for to find this code. NEVER write 'This chunk', 'This code', or failure scenarios."

Used cookbook's exact phrasing in the user message: *"Please give a short succinct context to situate this chunk within the overall document for the purpose of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."*

**Where:** `ingestion_service.py:_CONTEXT_SYSTEM` and `_enrich_one()`

---

### 23.4 Parallel Search as First-Class Strategy (W4)

**Problem:** The agent's STRATEGY section listed search as step 4 but didn't make parallel execution explicit. The model often issued one `search_code` call, waited for results, then issued the next — wasting turns.

**Fix:** Promoted "fire ALL searches for the same question in ONE turn" to an explicit rule in STRATEGY step 4, with an example:
```
4. FIND — fire ALL searches for the same question in ONE turn (parallel execution)
           e.g. search_code("forward pass") + search_code("loss function") together
           NEVER send one search, wait, then send the next — that wastes turns
```

Added to RULES: `PARALLEL: group all searches covering the same question into one turn — they execute concurrently`

**Why this matters:** In a ReAct loop, each LLM call is the bottleneck. Parallel tool calls reduce a 3-search investigation from 3 LLM rounds to 1. The MCP server already handles concurrent tool calls — the bottleneck was the model not knowing to group them.

**Where:** `agent.py:SYSTEM_PROMPT`

---

### 23.5 README Badge Stripping (W5)

**Problem:** The README summary injected into the agent's context often started with badge lines (`[![CI](...)](#...)`) which wasted the 200-char budget on noise.

**Fix:** Strip badge lines with regex before extracting the first substantive sentence:
```python
text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)  # badge links
text = re.sub(r'!\[.*?\]\(.*?\)', '', text)              # inline images
# then find first line ≥20 chars, not starting with 'http' or '|'
```

**Where:** `agent.py:_get_readme_summary()`

---

### 23.6 Non-Contradictory MCP Tool Documentation (W6)

**Problem:** The MCP server's `instructions` block said "Use `list_files` to browse" — which directly contradicted the SYSTEM_PROMPT that said to prefer `search_code`. Two competing instructions caused the model to oscillate between approaches.

**Design principle (from Anthropic's tool documentation guide):** Tool descriptions should specify *when to use it over similar tools* and *when NOT to use it* — a "PREFER THIS OVER / PREFER X OVER THIS" boundary on every tool. Never let server-level and tool-level instructions contradict.

**Fix:** Replaced the contradictory 8-line `instructions` block with a neutral 2-sentence description:
> "Code search and navigation server for indexed GitHub repositories. Each tool description specifies exactly when to use it over similar tools — read those descriptions to make the right tool choice."

**Where:** `mcp_server.py` — FastMCP constructor `instructions` argument.

---

### 23.7 Parallel Tour Investigation (M1)

**Problem:** Phase 2 of the tour agent investigated each pipeline stage sequentially. A 5-stage pipeline made 5 serial LLM calls — all independent, none waiting for the others.

**Pattern (from ingestion_service.py `_add_context`):** Use `ThreadPoolExecutor` for independent tasks. 3 workers = ~3x speedup without overwhelming free-tier rate limits (15 RPM Gemini). All-at-once parallelism hits the rate limiter on 6+ stage pipelines.

**Generator constraint:** A Python generator (`yield`) cannot `yield` from inside a thread. Solution: emit "Queued: stage N" events *before* the pool starts, run the pool synchronously, then emit "Found: ..." events *after* all futures complete — in pipeline order.

```python
# 1. Emit start events (before pool)
for i, stage in enumerate(stages):
    yield {"stage": "investigating", "message": f"Queued: {stage['name']}"}

# 2. Run in parallel — can't yield here
insights: list[dict | None] = [None] * n_stages
with ThreadPoolExecutor(max_workers=3) as pool:
    future_to_idx = {pool.submit(_phase_investigate, ...): i for i, ... in ...}
    for future in as_completed(future_to_idx):
        insights[future_to_idx[future]] = future.result()

# 3. Emit found events (after pool, in order)
for i, (stage, insight) in enumerate(zip(stages, insights)):
    yield {"stage": "investigating", "message": f"Found: {insight['name']}"}
```

**Where:** `tour_agent.py:build()` Phase 2 section.

---

### 23.8 Broader Agent Search (M2)

**Change:** `search_code` top_k bumped from 6 → 12 in `mcp_server.py`.

**Why:** The agent's ReAct loop processes all retrieved chunks before deciding the next action. At top_k=6, a query that spans both a class definition and its call sites often misses one side — the agent then issues a second nearly-identical search. At top_k=12, both sides typically land in one call.

**Cost:** Doubled context per search call (~400 → ~800 tokens), but halved the number of search rounds on cross-cutting questions. Net: faster answers.

---

### 23.9 Model Tiering (M3)

**Problem:** All tasks — high-volume chunk enrichment (one call per chunk at ingest time) and high-stakes synthesis (tour generation, diagram, README) — used the same model. Enrichment doesn't need synthesis quality; using the strong model for it wastes quota.

**Design:** Two tiers on the same provider/client — no second client needed:
- `self._model`: strong model (e.g. `gemini-2.5-flash`) — synthesis, diagrams, README, tour
- `self._fast_model`: lighter model (e.g. `gemini-2.0-flash-lite`) — contextual enrichment

**Thread safety:** The tier selection must be thread-safe because `_add_context` uses `ThreadPoolExecutor`. Solution: pass `model` as a value through the `params` dict (created fresh per call), never mutate `self._model`.

```python
def generate(self, ..., fast: bool = False) -> str:
    model  = self._fast_model if fast else self._model
    params = {"temperature": ..., "model": model}   # ← local dict, thread-safe
    ...
    # _groq_complete reads params["model"] instead of self._model
```

**Where:** `generation.py:GenerationService`, `ingestion_service.py:_enrich_one`

---

### 23.10 Negative Example Feedback (M4)

**Pattern:** Online learning without retraining. When the evaluator corrects a concept name (artifact → technique), persist the bad name to disk. On the next run for the same repo, inject it into Phase 3's prompt as a FORBIDDEN name.

**Effect:** After the first run that triggers corrections, subsequent runs for the same repo avoid generating those artifact names from the start — saving one evaluator round and producing cleaner output immediately.

**Storage:** `backend/tour_feedback/{repo_slug}_feedback.json` — flat dict `{bad_name: good_name}`. Accumulates across runs; never shrinks. Capped at 20 entries in the prompt to stay token-light.

**Data flow:**
```
build() run 1:
  Phase 3 synthesize → generates "ingestion_service.py" (artifact name)
  _validate_concepts → corrects to "Streaming Ingestion Pipeline"
  _save_feedback → writes {"ingestion_service.py": "Streaming Ingestion Pipeline"}

build() run 2:
  _phase_synthesize → _synthesize_negative_block() loads feedback
  Prompt now includes: PREVIOUSLY REJECTED: "ingestion_service.py" → "Streaming Ingestion Pipeline"
  Phase 3 avoids the artifact name before validation even runs
```

**Where:** `tour_agent.py` — `_load_feedback()`, `_save_feedback()`, `_synthesize_negative_block()`, `_validate_concepts()`, `_phase_synthesize()`

**Implementation:** `_CONTEXT_SYSTEM` and `_enrich_one()` in `backend/services/ingestion_service.py`

---

## 24. Full ReAct Pipeline: Agentic Phase 1 + Phase 2

This chapter documents the upgrade from a static 3-phase pipeline to a fully agentic one, and explains exactly WHY each phase needed to become a ReAct loop to reach expert-level output quality.

---

### 24.1 The Root Cause of Poor Tour Quality

Before this upgrade, tours consistently produced hollow concept cards — names like `ingestion_service.py`, `health`, or `AgentService._build_initial_messages`. The root cause was in Phase 1.

**Phase 1 (static)** received a pre-assembled snapshot: a flat list of all files, 14 random module chunks, and asked a single LLM call to identify "design decisions." The model could only match against what it was handed. The most visible signals in a static snapshot are often bootstrap files (`main.py`, `app.py`) and identifiers — exactly the artifacts we don't want.

**Phase 2 (static)** then investigated those artifact names. It called `_file_chunks(repo, stage_file)` to get pre-indexed chunks and made a single LLM call. If the design decision was implemented across three files, or the key function was two call layers deep, the one-shot call missed it and produced an empty insight.

**The fix:** Make both phases agentic — give them tools and let them decide what to read.

---

### 24.2 Why Phase 1 Needed to Become a ReAct Loop

**The static Phase 1 problem:** Giving the LLM 14 random module chunks produces output biased toward whatever happens to be in those chunks. If one of those chunks is from `main.py`, the model sees "imports every router, service, and config" — and infers "this is the orchestrator." But `main.py` IS the orchestrator. It's bootstrap code that wires things together, not a design decision.

**The ReAct solution:** The agent explores like a developer would:

```
list_files("") → see top-level structure
read_file("pyproject.toml") → understand tech stack from declared dependencies
read_file("retrieval/hybrid_search.py") → see actual algorithm logic
search_symbol("HybridSearcher") → find where it's defined
find_callers("hybrid_search") → see how it's invoked
DONE: {pipeline_stages: [{name: "Hybrid Dense-Sparse Retrieval", ...}]}
```

The key insight from studying claude-code's `/init` command: **start from manifest files, not source files**. `pyproject.toml` says `qdrant-client`, `sentence-transformers`, `tree-sitter`. From this alone, you know the tech stack — vector storage, embedding, AST parsing. Only then do you read source to find the design decisions. This is language-agnostic: it works for web apps, ML libraries, compilers, game engines.

**Phase 1 is a generator** — it yields trace events for the UI live-log panel as it reasons, then yields a final `{"type": "result", "data": pipeline_map}`.

---

### 24.3 Why Phase 2 Needed to Become a ReAct Loop

**The static Phase 2 problem:** Even with good stage names from agentic Phase 1, Phase 2 still had to answer: "What is the non-obvious design decision, why does it matter, and which functions implement it?" From pre-selected chunks of one file.

The problem: design decisions often span files.

Example — investigating "Hybrid Dense-Sparse Retrieval" from `retrieval/hybrid_search.py`:
- The sparse vector generation is in `ingestion/chunker.py` (ingest time, not retrieval time)
- The fusion formula is in `hybrid_search.py`
- The actual call that triggers both is in `retrieval_service.py`

A one-shot call on `hybrid_search.py` chunks misses the ingest-time sparse generation entirely and produces a shallow insight. The agentic version:

```
read_file("retrieval/hybrid_search.py")
  → THINK: "fusion is here but where are sparse vectors generated?"
  → find_callers("hybrid_search")
  → THINK: "called from retrieval_service.py — see the actual entry point"
  → search_symbol("SparseVectorizer")
  → THINK: "runs at ingest time in chunker.py — that's the non-obvious part"
  → DONE: {key_functions: ["SparseVectorizer", "hybrid_search", "_rrf_fuse"], ...}
```

**Phase 2 is NOT a generator** — it's called from `ThreadPoolExecutor` (3 workers in parallel, one per stage). Generators can't be used in thread pools. Instead, it accumulates trace internally (prints to stdout) and returns the result dict directly.

**Intentional asymmetry:** Phase 2 does NOT have `list_files`. Phase 1 explores broadly; Phase 2 investigates narrowly. Giving Phase 2 `list_files` would produce unfocused behavior — it would start browsing the repo instead of tracing the specific design decision it was given.

---

### 24.4 The ReAct Loop Mechanics

Both agentic phases use the same pattern:

```
transcript = "Repository: {repo}\n\nContext: {seeding info}\n\nBegin exploration..."

for round in range(max_rounds):
    raw = llm.generate(SYSTEM_PROMPT, transcript)

    # Parse the two-line output format
    THINK: [what I learned, what to do next]
    TOOL: tool_name("argument")
    # OR
    DONE: {"structured": "result"}

    if DONE:
        return parse_json(done_block)

    if TOOL:
        result = execute_tool(tool_name, argument)
        result = _token_budget(result, max_tokens=N)  # prevent transcript balloon
        transcript += f"\nTHINK: {think}\nTOOL: {display}\nRESULT:\n{result}\n"
    else:
        transcript += f"\n[No valid action — output TOOL: or DONE:]\n"

# Forced DONE after round limit
transcript += "\nROUND LIMIT REACHED. Output DONE: now."
raw = llm.generate(SYSTEM_PROMPT, transcript, max_tokens=1200)
return parse_json(raw)
```

**Token budget on tool results** (`_token_budget(result, max_tokens=N)`) is critical. Without it, a large file read in round 1 balloons the transcript and later rounds exceed the model's context window. Token budget truncates at `N * 4 chars` (4 chars/token heuristic) and snaps to a newline boundary so code isn't cut mid-line.

**Why `THINK:` + `TOOL:` on separate lines, not a structured JSON format:** Plain-text action format is more robust to LLM output variation than requiring `{"action": "tool_call", "name": "...", "args": {...}}`. The model reliably outputs two lines; JSON nesting in the middle of generation is error-prone. Simple regex parsing is also more maintainable.

---

### 24.5 System Prompt Design: No Domain Heuristics

Both system prompts (`_AGENTIC_MAP_SYSTEM`, `_AGENTIC_INVESTIGATE_SYSTEM`) follow the principle from claude-code's `/init`: **state principles, never examples from a specific domain**.

**What we had before (broken):**
> "Look for ingestion, embedding, retrieval, and inference stages. If you see a `routers/` directory, skip it..."

These break silently on any non-web, non-LLM repo. A game engine has no `ingestion/`. A compiler has no `routers/`. A math library has no `inference/`.

**What the prompts say now:**
> "A design decision is a non-obvious choice — there was a simpler alternative that was deliberately rejected. Design decisions are the concepts a new engineer must understand to work on this system effectively."

No domain terms. No directory name assumptions. The only "example" names used are abstract: "Lazy Evaluation Cache", "Incremental Recomputation" — these describe the *format* (technique + mechanism) not a specific domain.

---

### 24.6 Phase 3 Quality: The `ask` Field Problem

Phase 3 (Synthesize) assembles Phase 2 findings into the final JSON. One persistent quality gap: the `ask` field — the question shown on each concept card that prompts reflection.

**What weak `ask` fields look like:**
- "Why was the naive approach rejected?" — applies to any concept in any codebase, zero information
- "What would happen if this were removed?" — vague, doesn't name anything specific

**What strong `ask` fields look like:**
- "What does `_rrf_fuse` return when dense and sparse scores disagree by more than 0.5?"
- "Why does `SparseVectorizer` need to run at ingest time rather than at query time?"
- "What breaks in `hybrid_search` output if the BM25 index is empty?"

The rule enforced in the Phase 3 prompt: **`ask` MUST name a specific function from `key_items` and describe a concrete failure mode or edge case.** Generic asks are explicitly listed as BAD with the reason: "zero-information — applies to any concept anywhere."

This connects to the broader principle: a question is only educational if it's answerable by someone who specifically read THIS concept's implementation — not by someone who read the concept name.

---

### 24.7 What Changed in Code

| Location | Change |
|---|---|
| `_AGENTIC_MAP_SYSTEM` | Already existed from prior session |
| `_phase_map_agentic()` | Already existed (generator, yields trace events) |
| `_AGENTIC_INVESTIGATE_SYSTEM` | **New** — system prompt for Phase 2 ReAct loop |
| `_phase_investigate_agentic()` | **New** — non-generator, called from ThreadPoolExecutor |
| `build()` pool.submit call | Changed from `_phase_investigate` → `_phase_investigate_agentic` |
| `_SYNTHESIZE_SYSTEM` | Tightened: added ASK RULE about naming key_functions |
| Phase 3 prompt `ask` template | Tightened: 5 GOOD/BAD examples, description-0 data-flow rule |

**Fallback chain:**
```
_phase_investigate_agentic()
  → on loop exhaustion: forces DONE from transcript
    → on parse failure: _phase_investigate() (static one-shot fallback)
```

The fallback is never the happy path — it only triggers on LLM parse failures (malformed JSON, empty response). In normal operation, the agentic loop completes within 3–5 rounds.

---

### 24.8 Demonstrating ReAct in the UI

The Explore view's TracePanel shows Phase 1's ReAct rounds live as the tour generates. Each `react` trace event shows:
- The `THINK:` text — what the agent concluded and why it's making the next call
- The `TOOL:` call — which tool, which argument

This is intentionally educational. Cartographer's purpose is to teach how AI systems work, and a live ReAct loop is one of the most instructive patterns to observe. The user sees: "the model read the manifest, concluded it was a RAG system, then specifically read the hybrid search file because that's the interesting decision." That's a qualitatively better explanation of how AI reasoning works than any documentation could provide.

---

## 25. Tour Quality: Signal-Checklist Stopping, Gaps, and Claude Code /init Study

This chapter documents a root-cause analysis of why tours were producing too few, too shallow concept cards — and the fix derived from studying Claude Code's source directly.

---

### 25.1 The Root Cause: Count-Based Stopping

Phase 1 MAP's original DONE criterion was: **"when you have 5-8 decisions, call DONE."**

This sounds reasonable. In practice it meant the agent called DONE after reading 2-3 directories — because it could assemble 5 stage names from a handful of chunks. It had satisfied the count without exploring the breadth of the repo.

The symptom: tours with 4 cards, all from `backend/services/`. The retrieval layer, the ingestion layer, the UI layer — never reached.

The fix wasn't to increase `max_rounds` (we tried that — it helped but the fundamental incentive was wrong). The fix was to change **what DONE means**.

---

### 25.2 Studying Claude Code's /init Source

To find a better stopping criterion we read Claude Code's `/init` source at `github.com/codeaashu/claude-code/`. Key findings:

**Claude Code's /init has no round limit.** Its Phase 2 (Explore subagent) terminates when it has found a specific checklist of signals, not when it has reached a count:

```
- Build, test, and lint commands (especially non-standard ones)
- Languages, frameworks, and package manager
- Project structure (monorepo / multi-module / single project)
- Code style rules that differ from language defaults
- Non-obvious gotchas, required env vars, workflow quirks
- Git worktree usage
```

And the termination signal is this exact line from the prompt:
> **"Note what you could NOT figure out from code alone — these become interview questions."**

The agent doesn't stop when it has found N things. It stops when it can also enumerate what it *couldn't* find. Gaps are a first-class output, not a failure mode.

**The Explore agent has a minimum query threshold.** It cannot call DONE after a single lookup. The minimum is 3 tool calls — enforced in the agent definition.

**Manifest files are always read first.** `package.json`, `Cargo.toml`, `pyproject.toml`, `go.mod` before any source file. The reasoning: dependencies reveal the tech stack for any repo without needing directory-name heuristics. `torch + transformers` → ML pipeline. `llvm-sys` → compiler. `qdrant-client` → vector search. This is language-agnostic and universally reliable.

---

### 25.3 The Signal-Checklist DONE Criterion

We replaced the count-based stopping with five named signals that the agent must confirm before calling DONE:

```
① Entry point — which file/function starts the core execution path?
② Core concepts — what are the key algorithms, data structures, subsystems, or abstractions
   that define how this system works?
③ Key dependencies — from the manifest, which non-trivial libraries were chosen?
   Each non-trivial library reveals something about what the system does.
④ Directory breadth — at least one file READ (not just listed) from every non-trivial
   top-level directory. Listing a directory without reading a file inside it does NOT count.
⑤ Gaps — what can the code NOT tell you? (rationale, why a library was chosen over another,
   decisions that live only in commit history or the author's head)
```

The THINK line before DONE must state what was found for each signal — or "not found yet" if one is missing. This forces the model to self-check before declaring done.

Signal ④ is the one that was causing premature stopping. The model was satisfying the count before it had read across directories. Now it must prove breadth.

Signal ⑤ is borrowed directly from Claude Code. Gaps are now a field in the Phase 1 DONE JSON and are passed to Phase 3 synthesis — the synthesiser uses them to write `ask` fields that point readers toward genuinely open questions.

---

### 25.4 The Gaps Field: What Code Alone Cannot Tell You

Every codebase has decisions that aren't visible in the code itself:
- Why was library X chosen over library Y?
- Why was this algorithm used instead of the obvious one?
- What constraint drove this design?

These live in commit history, design docs, Slack threads, or the author's head. Code can show *what* was done, not always *why*.

**In Claude Code's /init:** gaps become interview questions — the tool asks the user to fill them in before writing CLAUDE.md.

**In Cartographer:** we have no user to interview. Instead, gaps surface in the tour's `ask` fields — the question on each concept card that invites the reader to dig deeper. An `ask` like "Why does `HybridSearcher` use RRF fusion instead of weighted sum — what failure mode does RRF avoid?" is only possible if Phase 1 flagged "the choice of fusion formula" as something the code doesn't explain.

The data flow:
```
Phase 1 DONE JSON → gaps field
  → _phase_synthesize receives gaps_section
    → Phase 3 prompt: "What the code alone could NOT answer (use these in 'ask' fields)"
      → stronger, more specific ask fields on concept cards
```

---

### 25.5 Claude Code's Architecture: What They Do vs What We Do

Studying /init revealed an intentional difference worth understanding:

**Claude Code /init: one broad sweep → synthesis**
Phase 2 is a single Explore subagent pass. After it completes, the primary agent goes straight to writing CLAUDE.md. There is no per-component deep dive.

This is correct for their use case: CLAUDE.md needs *breadth* (build commands, code style, gotchas across the whole repo). It doesn't need to understand *why* each design decision was made.

**Cartographer: broad sweep → N deep dives → synthesis**
Phase 1 maps the architecture. Phase 2 investigates each stage in parallel (ThreadPoolExecutor, 3 workers). Phase 3 synthesises.

This is correct for our use case: a concept tour needs to give a new contributor a mental model of the system — what each component IS, how it works, and what they must understand before touching it. That requires per-component depth, not just breadth.

The extra phase (Phase 2 × N) is not complexity for its own sake. It's what separates "here are the files that implement the system" from "here is what you need to understand to work on this system."

---

### 25.6 The Six Built-In Agent Types in Claude Code

Claude Code ships six built-in agent types. Understanding them helps when designing agents:

| Agent | Model | Tools | Purpose |
|---|---|---|---|
| **Explore** | haiku (fast) | Glob, Grep, Read, Bash (read-only) | Broad codebase survey — no writes |
| **General-Purpose** | inherit | All (`*`) | Multi-step research, code search |
| **Plan** | inherit | Read-only (no Edit/Write) | Architecture planning |
| **Verification** | inherit | Build/test tools, no writes | "Try to break it" — adversarial testing |
| **Claude Code Guide** | haiku | Fetch, Search | Answers questions about Claude Code |
| **Status Line Setup** | inherit | Read, Edit | Configures UI status line |

**Model tiering**: haiku for navigation (Explore, Guide), primary for reasoning (Plan, Verify). The expensive model is reserved for synthesis and adversarial evaluation — cheap model for enumeration.

**Verification agent's system prompt** (verbatim): *"Your job is not to confirm the implementation works — it's to try to BREAK it."* It outputs `VERDICT: PASS | FAIL | PARTIAL`. This is a cleaner output format than our evaluator's `ok/fixed` — `PARTIAL` gives finer signal when some concepts pass and others don't.

---

### 25.7 What Changed in Code

| Location | Change | Why |
|---|---|---|
| `_AGENTIC_MAP_SYSTEM` DONE criterion | Count (`5-8`) → 5-signal checklist | Count allowed premature DONE after 2 dirs |
| `_AGENTIC_MAP_SYSTEM` DONE JSON | Added `gaps` field | Surface unknowns for Phase 3 ask fields |
| `_AGENTIC_MAP_SYSTEM` EXPLORATION STRATEGY step 3 | Manifest read added explicitly | Highest-signal file — reveals tech stack |
| `_phase_synthesize` | Added `gaps_section` variable + injected into prompt | Phase 3 uses Phase 1 gaps in ask fields |
| `_AGENTIC_INVESTIGATE_SYSTEM` subtitle example | `'Sequential embedding halves throughput'` → generic write-latency example | "embedding" is RAG-specific |
| `_AGENTIC_MAP_SYSTEM` STAGE NAME RULES | `'Hybrid Sparse-Dense Retrieval'` → `'Incremental Recomputation'` | Cartographer-specific example |
| `generation.py` SambaNova model | `Meta-Llama-3.1-405B-Instruct` → `DeepSeek-V3.1` | Code never matched the print statement after the deprecation fix |
| `generation.py` OpenRouter comment | "DeepSeek-V3" → "Qwen3-Coder" | Model had already been changed, comment was stale |

---

## 26. Tour Framing, Prompt Hygiene, Model Tiering, and Navigation Tools

This chapter covers four closely related improvements made in the session after Chapter 25 — all aimed at the same root problem: the tour was producing vague, abstract concept cards instead of useful contributor knowledge.

---

### 26.1 The Wrong Organizing Principle

The original framing across all tour prompts was:

> **"Find design decisions where a simpler alternative was rejected."**

This sounds like a good heuristic for understanding a codebase. In practice it consistently produced cards like:

- `Repository Abstraction Mechanism`
- `Retrieval Strategy Architecture`
- `Embedding Pipeline Pattern`

These names end in generic category words. They describe *how something is classified* (it's a pattern, it's a strategy), not *what it actually does*. A new contributor reading these cards learns nothing concrete. They can't trace anything to code. They don't know what to read first.

The root cause: "design decision where simpler alternative was rejected" is an *archaeological* frame. It asks the LLM to reconstruct the author's decision-making history. But the LLM doesn't have access to that history — it only has the code. So it pattern-matches and produces plausible-sounding category labels.

---

### 26.2 The Curriculum Frame

The fix: change the organizing question.

**Old:** "What design decisions were made where a simpler path was rejected?"
**New:** "What must a new contributor understand to work on this codebase confidently?"

This is the difference between archaeology and teaching. A teacher doesn't reconstruct history — they build a curriculum. The right mental model for a tour is: **a table of contents for a book about this specific codebase.**

Each concept card is a chapter. A good chapter title:
- Names something you can point to in the code
- Is specific to what *this* system does, not any system
- Describes what the thing IS, not how it's categorised

```
BAD: "Retrieval Strategy Architecture"   (category label — every system has some retrieval strategy)
BAD: "Sparse-Dense Fusion Pattern"       (methodology label — doesn't name the mechanism)
GOOD: "Two-Stage Candidate Retrieval"    (names what this system specifically does)
GOOD: "Token Budget Enforcement"         (points to a concrete mechanism with a specific cost)
```

The change propagated through every prompt in the pipeline:
- `_AGENTIC_MAP_SYSTEM`: "KEY CONCEPTS a new contributor must understand" (not "KEY DESIGN DECISIONS")
- `_MAP_SYSTEM`: "table of contents for a book about this codebase"
- `_AGENTIC_INVESTIGATE_SYSTEM`: WHAT/HOW/WHERE/WHY (not WHY-first / "naive alternative")
- `_SYNTHESIZE_SYSTEM`: description field = "what this IS and its role" (not "why chosen over naive alternative")
- `_VALIDATE_SYSTEM`: now catches pattern labels, not just artifact names

---

### 26.3 Prompt Hygiene — No Hardcoded Domain Examples

Every prompt rule must work for any repository — a game engine, a compiler, a math library, a CLI tool. The moment a prompt contains domain-specific terms, the LLM anchors to that domain and steers toward similar-sounding output even for unrelated repos.

**What we found and removed:**

| Location | Bad example | Why bad |
|---|---|---|
| `_AGENTIC_MAP_SYSTEM` STAGE NAME RULES GOOD examples | `'Computation Graph Traversal'`, `'Token Embedding Layer'` | Only plausible for ML/neural net repos |
| `_AGENTIC_MAP_SYSTEM` STAGE NAME RULES BAD example | `'Hybrid Sparse-Dense Retrieval'` | Cartographer-specific (RAG terminology) |
| `_AGENTIC_INVESTIGATE_SYSTEM` subtitle BAD example | `'Sequential embedding halves throughput'` | "embedding" is RAG-specific |
| `_fmt_files_by_directory` docstring | Cartographer directory names as examples | Comments visible to developers, not LLMs — but set a bad precedent |

**The fix:** Describe the *shape* of a good name without concrete domain examples. Instead of:
```
GOOD: 'Computation Graph Traversal', 'Token Embedding Layer'
```
Write:
```
GOOD: specific to what THIS system does — name the actual mechanism,
      algorithm, or abstraction you found in the code.
```

The model is capable of generating good names from principles. Giving it ML examples as "good" teaches it that ML-sounding names are the goal.

**Test for any new prompt rule:** Would this rule produce sensible output on `karpathy/micrograd`? On a Rust CLI parser? On a game engine? If a rule only makes sense for RAG/web apps, it has a hardcoded assumption.

---

### 26.4 Two-Tier Validation: Artifacts and Pattern Labels

`_VALIDATE_SYSTEM` originally caught only one failure mode:

**ARTIFACT**: name is a code identifier (file path, class name, function name). Signs: ends in `.py/.ts/.js`, contains underscores, is CamelCase matching a class.

But category labels passed validation. "Repository Abstraction Mechanism" contains no underscores, isn't a filename, doesn't match a class name — yet it's useless.

We added a second failure mode:

**CATEGORY LABEL**: name ends in `Strategy`, `Mechanism`, `Pattern`, `Architecture`, `System`, `Layer`, `Framework`, or `Approach`. These suffix words classify what something *is*, not what it *does*. They are meaningful in a design document where you've already explained the concept — but as a chapter title they say nothing.

The validator now distinguishes:
1. Artifact names → rename to the technique they implement
2. Category labels → rename to describe the specific mechanism

A good name passes both tests: not a code artifact, not a category label. It names something concrete that exists in *this* codebase.

---

### 26.5 Model Tiering: `generate_synthesis()`

**The problem:** Phase 1 and Phase 2 use up the Gemini rate-limit window (15 RPM free tier). By the time Phase 3 synthesis runs, Gemini is exhausted — the fallback cascade reaches Cerebras llama3.1-8b. An 8B model writing 3000-token structured JSON truncates mid-object and crashes.

**Why this matters:** Tool calls (read_file, list_files, search_symbol) only need to output one line of JSON — an 8B model handles that fine. Synthesis needs to output a complete, deeply nested JSON object with 6-8 concepts, all fields filled, all strings non-empty. That is exactly where model quality determines output quality.

**The fix:** `generate_synthesis()` in `generation.py`:

```python
def generate_synthesis(self, system: str, prompt: str, **kwargs) -> str:
    # 1. Clear the Gemini exhaustion window — if 60s has passed, Gemini may be ready again
    self._exhausted_until.pop('gemini', None)
    self._exhausted_until.pop('gemma4', None)
    # 2. Block Cerebras 8B for this call only (save/restore state after)
    saved_cerebras = self._exhausted_until.get('cerebras')
    self._exhausted_until['cerebras'] = time.monotonic() + 3600
    self._skip_thinking = True
    try:
        return self.generate(system, prompt, **kwargs)
    finally:
        self._skip_thinking = False
        # Restore Cerebras state — block was call-scoped, not permanent
        if saved_cerebras is None:
            self._exhausted_until.pop('cerebras', None)
        else:
            self._exhausted_until['cerebras'] = saved_cerebras
```

**Design decisions in `generate_synthesis()`:**

- **Pop Gemini's exhaustion entry, don't set it.** If 60 seconds has passed since the rate limit hit, the window has expired and Gemini is ready again. Popping lets the cascade try Gemini first. This is the common case — Phase 1+2 finish in ~30-40s, then synthesis runs.

- **Block Cerebras with a 3600s window, not a flag.** Using the same `_exhausted_until` dict that the cascade already consults keeps the blocking mechanism consistent — one code path handles all provider skipping. The 3600s (1 hour) ensures no retry within the call.

- **Save and restore Cerebras state.** If Cerebras was already exhausted from a prior call, we don't want to accidentally un-exhaust it after synthesis. The save/restore pattern makes the block surgical: it affects only this call, leaves global state unchanged.

- **`_skip_thinking = True` for synthesis.** Gemma4 uses a thinking chain which can consume half the output budget on reasoning before writing JSON. Synthesis needs the full budget for JSON output.

`generate_synthesis()` is called for Phase 3 synthesis and for both forced-DONE paths (Phase 1 and Phase 2 round-limit fallbacks). Forced DONE on a long transcript is exactly when 8B model failure is most likely — it's the highest-stakes call in the pipeline.

**The general principle** (applicable to any multi-phase AI system):

```
Cheap/fast model: routing decisions, tool calls, basic planning
Strong model:     complex reasoning, ambiguous judgment, final synthesis
```

Never let the synthesis step fall through to the weakest model in the cascade. The tool-call loop can tolerate a weaker model because each call outputs one line. The synthesis step cannot — it outputs the entire result.

---

### 26.6 Navigation Tools: `glob` and `grep`

The Phase 1 MAP agent previously had five tools: `list_files`, `read_file`, `search_symbol`, `find_callers`, `trace_calls`. To find the entry point of an unknown repo it had to:

1. `list_files("")` — see top-level dirs
2. `list_files("src/")` — see what's inside
3. `read_file("src/main.py")` — check if it's the entry point
4. Repeat for other directories if wrong

Three to four tool calls just to locate the entry point.

**What Claude Code's Explore agent has:** `glob(pattern)` and `grep(pattern)` — find files by name pattern and find content by regex, without reading every file. One call each.

We added both, implemented over the in-memory chunk cache:

**`_agentic_glob(repo, pattern)`**
```python
# fnmatch pattern matching over cached chunk filepaths
matched = sorted(p for p in paths if _fnmatch.fnmatch(p, pattern))
```
Example: `glob("**/*.py")` lists all Python source files in one call. Zero network requests — it filters the chunk data already loaded by Phase 0.

**`_agentic_grep(repo, pattern)`**
```python
# Regex search over cached chunk text, one match per chunk, capped at 20
for c in chunks:
    for line in c["text"].splitlines():
        if rx.search(line):
            results.append(f"{filepath}  ({chunk_name}):\n  {line.strip()}")
            break  # one match per chunk keeps results scannable
```
Example: `grep("def main|if __name__")` finds the entry point across all source files without reading any of them. The result shows filepath + chunk name + matching line — enough for the agent to decide which file to `read_file()` next.

**Why one match per chunk (not per file):** A large file produces many chunks. Without the per-chunk break, one 500-line file could consume 15 of the 20 result slots, crowding out matches from other files. One match per chunk gives breadth across the whole repo.

**Where each tool goes:**

| Tool | Phase 1 (MAP) | Phase 2 (INVESTIGATE) |
|---|---|---|
| `glob` | Yes — discover file inventory | No — already know target file |
| `grep` | Yes — find entry points, patterns | Yes — find usages without reading every file |

Phase 2 gets `grep` but not `glob` because investigation is already focused on one file. Glob adds no value when you're asking "what does this specific function do?"

**The key architectural insight:** Both tools use `_all_chunks(repo)` which is cached in `_chunk_cache` from Phase 0. Adding navigation tools that don't require new network calls is "free" — the data is already in memory.

---

### 26.7 The MCP Server and the Chat Agent

Two separate clients use `mcp_server.py`:

**1. Cartographer's own chat agent** (`backend/services/agent.py`)
Always connected. When you ask a question in the chat UI, `agent.py` calls `mcp_server.py` over HTTP to execute tools (`search_code`, `find_callers`, `list_files`, etc.). The MCP server is the tool backend for the chat agent. This requires no external setup.

**2. Claude Code (the Anthropic CLI)**
Optional external connection. A user running `claude` in their terminal can point it at Cartographer's MCP server endpoint via an `mcp.json` config file. Claude Code then discovers and calls Cartographer's tools from within a Claude Code session — letting you use Cartographer's search/navigation capabilities without the chat UI.

These are two different consumers of the same server. The chat agent always uses it. Claude Code using it is an optional power-user scenario.

`glob` and `grep` were added to the tour agent's ReAct loop — they are *not* yet registered as `@mcp.tool()` functions in `mcp_server.py`. That means the chat agent and external Claude Code sessions do not yet have access to them. Adding them to `mcp_server.py` would give both consumers the same capability.

---

### 26.8 What Changed in Code

| Location | Change | Why |
|---|---|---|
| All tour prompts | "design decisions" → "key concepts a new contributor must understand" | Curriculum frame produces concrete names; archaeology frame produces category labels |
| `_AGENTIC_MAP_SYSTEM` STAGE NAME RULES | Removed ML-specific GOOD examples | Hardcoded domain examples anchor the model to that domain |
| `_AGENTIC_MAP_SYSTEM` EXPLORATION STRATEGY | Added step 3: `grep` for entry point, `glob` for file inventory | Reduces navigation round-trips before committing to `read_file` |
| `_VALIDATE_SYSTEM` | Added FAILURE MODE 2: pattern labels (Strategy/Mechanism/Pattern/Architecture/etc.) | Category label names pass artifact check but are still useless |
| `_AGENTIC_INVESTIGATE_SYSTEM` | WHAT/HOW/WHERE/WHY framing; `naive_rejected` made optional | Not every concept has a rejected alternative |
| Phase 3 `description` template | "what this IS and its role" replaces "why chosen over naive alternative" | Explains the concept, not the author's past deliberation |
| Phase 3 `ask` template | "specific constraint a contributor needs to know" replaces "what breaks if simpler behaviour used" | More generative — works for concepts without rejected alternatives |
| `generation.py` | Added `generate_synthesis()` | Ensures synthesis always gets a strong model; blocks 8B fallback |
| `generation.py` `_is_exhausted()` | Added `"500"`, `"internal error"`, `"internal_error"` | Gemini transient 500s weren't triggering fallback |
| `tour_agent.py` | Added `_agentic_glob()` and `_agentic_grep()` | Navigation without read_file round-trips; implemented over chunk cache |
| Phase 1 tool dispatch | `glob` and `grep` cases added | Route new tool names to implementations |
| Phase 2 tool dispatch | `grep` case added (no `glob` — investigation is focused) | grep useful for finding usages; glob not needed when target file is known |

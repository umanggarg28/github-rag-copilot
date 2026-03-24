"""
main.py — FastAPI application entry point.

FastAPI concepts used here:

  Lifespan (startup/shutdown):
    Services like Embedder load a 600MB model. We initialise them once at
    startup and share the instance across all requests — not per-request.
    FastAPI's `lifespan` context manager replaces the old @app.on_event("startup").

  Dependency injection:
    `Depends(get_ingestion_service)` makes FastAPI call get_ingestion_service()
    and pass the result as a parameter. This decouples services from endpoints,
    making them easier to test (swap in a mock) and reason about.

  StreamingResponse:
    For the /query/stream endpoint, we use SSE (Server-Sent Events).
    The browser receives tokens as they arrive, not all at once.
    Each event is formatted as "data: <token>\n\n" per the SSE spec.

  CORS:
    The React frontend runs on localhost:5173; the backend on localhost:8000.
    Different origins — browser blocks requests by default (Same-Origin Policy).
    CORSMiddleware adds headers that tell the browser to allow cross-origin requests.

Endpoints:
  POST /ingest              — ingest a GitHub repo
  GET  /repos               — list all indexed repos
  DELETE /repos/{slug}      — delete a repo from the index
  POST /search              — retrieve chunks (no generation)
  POST /query               — RAG: retrieve + generate answer
  GET  /query/stream        — RAG with streaming SSE response
  POST /agent/query         — Agentic RAG: ReAct loop (synchronous)
  GET  /agent/stream        — Agentic RAG: ReAct loop with SSE progress stream
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.models.schemas import (
    IngestRequest, IngestResponse,
    SearchRequest, SearchResponse, CodeChunk,
    QueryRequest, QueryResponse,
    ReposResponse, RepoInfo,
    AgentRequest, AgentResponse, AgentToolCall,
)
from backend.config import settings
from backend.services.ingestion_service import IngestionService
from backend.services.generation import GenerationService, classify_query
from backend.services.agent import AgentService
from backend.services.graph_service import GraphService
from retrieval.retrieval import RetrievalService
from ingestion.qdrant_store import QdrantStore


# ── Shared service instances ───────────────────────────────────────────────────
# These are module-level singletons initialised at startup. FastAPI's lifespan
# ensures they're ready before the first request arrives.

_ingestion_service: IngestionService | None = None
_retrieval_service: RetrievalService | None = None
_generation_service: GenerationService | None = None
_agent_service: AgentService | None = None
_graph_service: GraphService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context: runs startup code before yield, shutdown code after.

    Everything before `yield` runs when the server starts.
    Everything after `yield` runs when the server shuts down (Ctrl+C).

    Loading models here (not at import time) means startup errors are visible
    in the server log, not buried in a traceback from a module-level call.
    """
    global _ingestion_service, _retrieval_service, _generation_service, _agent_service, _graph_service
    print("Starting up — loading models and connecting to Qdrant...")
    _ingestion_service = IngestionService()
    _retrieval_service = RetrievalService()
    _graph_service = GraphService(QdrantStore())
    _generation_service = GenerationService()
    # AgentService is optional — only initialised when ANTHROPIC_API_KEY is set.
    # If no key, the /agent/* endpoints return a clear error rather than crashing.
    if settings.anthropic_api_key:
        _agent_service = AgentService(_retrieval_service)
        print("AgentService ready (agentic RAG enabled).")
    else:
        print("No ANTHROPIC_API_KEY — /agent/* endpoints disabled.")
    print("All services ready.\n")
    yield
    # Cleanup on shutdown (if needed) goes here
    print("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GitHub RAG Copilot",
    description="Ask questions about any GitHub repository.",
    version="0.1.0",
    lifespan=lifespan,
)

# Build allowed origins list.
# Always include local dev ports; add the production Vercel URL when set.
_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",
]
if settings.frontend_url:
    _origins.append(settings.frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependency providers ───────────────────────────────────────────────────────
# These functions are called by FastAPI's Depends() system.
# They return the shared singleton — no new object is created per request.

def get_ingestion_service() -> IngestionService:
    if _ingestion_service is None:
        raise RuntimeError("IngestionService not initialised")
    return _ingestion_service

def get_retrieval_service() -> RetrievalService:
    if _retrieval_service is None:
        raise RuntimeError("RetrievalService not initialised")
    return _retrieval_service

def get_generation_service() -> GenerationService:
    if _generation_service is None:
        raise RuntimeError("GenerationService not initialised")
    return _generation_service

def get_graph_service() -> GraphService:
    if _graph_service is None:
        raise RuntimeError("GraphService not initialised")
    return _graph_service

def get_agent_service() -> AgentService:
    if _agent_service is None:
        raise HTTPException(
            status_code=503,
            detail="Agentic RAG requires ANTHROPIC_API_KEY — not configured on this server.",
        )
    return _agent_service


# ── Routes: Ingestion ──────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_repo(
    request: IngestRequest,
    svc: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """
    Ingest a GitHub repository into the vector index.

    Downloads the repo zip, filters files, chunks them, embeds each chunk,
    and upserts to Qdrant. Idempotent — safe to re-run for the same repo.
    Set `force=true` to delete and re-index from scratch.
    """
    try:
        result = svc.ingest(request.repo_url, force=request.force)
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/repos", response_model=ReposResponse, tags=["ingestion"])
async def list_repos(
    svc: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """List all indexed repositories and their chunk counts."""
    repos = svc.list_repos()
    total = sum(r["chunks"] for r in repos)
    return ReposResponse(
        repos=[RepoInfo(slug=r["slug"], chunks=r["chunks"]) for r in repos],
        total_chunks=total,
    )


@app.delete("/repos/{owner}/{name}", tags=["ingestion"])
async def delete_repo(
    owner: str,
    name: str,
    svc: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """Delete all chunks for a repository from the index."""
    slug = f"{owner}/{name}"
    deleted = svc.delete_repo(slug)
    return {"repo": slug, "chunks_deleted": deleted}


# ── Routes: Code Graph ────────────────────────────────────────────────────────

@app.get("/repos/{owner}/{name}/graph", tags=["graph"])
async def get_repo_graph(
    owner: str,
    name: str,
    graph_svc: Annotated[GraphService, Depends(get_graph_service)],
):
    """
    Build and return the call graph for an indexed repository.

    Returns a graph in D3-ready format:
      nodes — functions and classes with metadata (name, file, line numbers)
      edges — call relationships ("A calls B")

    Node size in the UI is proportional to caller_count (in-degree):
    functions called by many others appear larger — the "hub" functions.

    NOTE: Call data requires re-ingestion after upgrading to this version.
    Repos indexed before call extraction was added will have nodes but no edges.
    Re-ingest with force=true to get the full graph.
    """
    repo = f"{owner}/{name}"
    try:
        graph = graph_svc.build_graph(repo)
        return graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph build failed: {e}")


# ── Routes: Search ─────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    request: SearchRequest,
    svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
):
    """
    Retrieve relevant code chunks without generating an answer.

    Useful for exploring what the index contains for a given query,
    or for building a custom frontend that handles generation separately.
    """
    results = svc.search(
        query=request.query,
        top_k=request.top_k,
        repo_filter=request.repo,
        language_filter=request.language,
        mode=request.mode,
    )
    return SearchResponse(
        query=request.query,
        results=[CodeChunk(**r) for r in results],
        total=len(results),
    )


# ── Routes: Query (RAG) ────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query(
    request: QueryRequest,
    retrieval_svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
    generation_svc: Annotated[GenerationService, Depends(get_generation_service)],
):
    """
    Full RAG pipeline: retrieve relevant code, then generate an answer.

    The response includes the answer, the sources used (with file + line
    citations), and the query_type (technical/creative) that was detected.
    """
    query_type = classify_query(request.question)

    results = retrieval_svc.search(
        query=request.question,
        top_k=request.top_k,
        repo_filter=request.repo,
        language_filter=request.language,
        mode=request.mode,
        relevance_threshold=request.relevance_threshold,
    )

    if not results:
        return QueryResponse(
            question=request.question,
            answer="No relevant code found in the indexed repositories for this question.",
            sources=[],
            query_type=query_type,
        )

    context = retrieval_svc.format_context(results)
    answer  = generation_svc.answer(request.question, context, query_type)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[CodeChunk(**r) for r in results],
        query_type=query_type,
    )


@app.get("/query/stream", tags=["query"])
async def query_stream(
    question: Annotated[str, Query(description="Question about the codebase")],
    retrieval_svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
    generation_svc: Annotated[GenerationService, Depends(get_generation_service)],
    repo: str | None = None,
    language: str | None = None,
    top_k: int | None = None,
    mode: str = "hybrid",
    relevance_threshold: float = 0.0,
):
    """
    Streaming RAG endpoint using Server-Sent Events (SSE).

    Why GET instead of POST for streaming?
      The browser's native EventSource API (for SSE) only supports GET.
      We pass parameters as query strings instead of a request body.

    SSE format:
      Each event is a string "data: <content>\n\n"
      The browser splits on double newline to separate events.
      We send one event per token, then a final "data: [DONE]\n\n".

    Usage (JavaScript):
      const es = new EventSource(`/query/stream?question=...`);
      es.onmessage = (e) => {
        if (e.data === "[DONE]") { es.close(); return; }
        setAnswer(prev => prev + e.data);
      };
    """
    query_type = classify_query(question)

    results = retrieval_svc.search(
        query=question,
        top_k=top_k,
        repo_filter=repo,
        language_filter=language,
        mode=mode,
        relevance_threshold=relevance_threshold,
    )

    if not results:
        async def no_results():
            yield "data: No relevant code found in the indexed repositories.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    context = retrieval_svc.format_context(results)

    def token_stream():
        import json
        # First event: send sources + query_type as structured JSON so the
        # frontend gets everything in one SSE connection (no second POST /query call).
        meta = {
            "sources":    [CodeChunk(**r).model_dump() for r in results],
            "query_type": query_type,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Subsequent events: stream tokens
        for token in generation_svc.stream(question, context, query_type):
            # Escape newlines — SSE uses \n\n as event delimiter
            safe_token = token.replace("\n", "\\n")
            yield f"data: {safe_token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ── Routes: Agentic RAG ────────────────────────────────────────────────────────
#
# These endpoints wrap AgentService, which runs a ReAct loop:
#   question → think → search → observe → think → search → ... → answer
#
# Why two endpoints?
#   POST /agent/query  — synchronous. Wait for the full answer, return JSON.
#                        Simple to integrate, but slow (the whole loop runs first).
#   GET  /agent/stream — streaming SSE. Watch the agent's thinking in real time.
#                        Shows each tool call as it happens (like watching AI think).
#
# The streaming endpoint is the "wow" version — users can see the agent reasoning
# live: "Searching for backward()... found engine.py... now looking for callers..."

@app.post("/agent/query", response_model=AgentResponse, tags=["agent"])
async def agent_query(
    request: AgentRequest,
    agent_svc: Annotated[AgentService, Depends(get_agent_service)],
):
    """
    Run the agentic RAG loop synchronously.

    The agent searches the codebase multiple times, from different angles,
    until it has enough evidence to answer confidently. Returns the full
    reasoning trace (tool_calls) alongside the answer.

    Slower than /query but more thorough — the agent decides what to search,
    not a fixed single retrieval. Best for complex multi-hop questions like
    "how does the training loop interact with the optimizer?" that require
    understanding how multiple pieces connect.
    """
    try:
        result = agent_svc.run(request.question, repo_filter=request.repo)
        return AgentResponse(
            answer=result["answer"],
            tool_calls=[AgentToolCall(**tc) for tc in result["tool_calls"]],
            iterations=result["iterations"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


@app.get("/agent/stream", tags=["agent"])
async def agent_stream(
    question: Annotated[str, Query(description="Question about the codebase")],
    agent_svc: Annotated[AgentService, Depends(get_agent_service)],
    repo: str | None = None,
):
    """
    Run the agentic RAG loop with real-time SSE progress streaming.

    Unlike /query/stream (which just streams tokens), this endpoint lets you
    watch the agent's full reasoning process as it happens:

      event: tool_call   → agent is about to call a tool (shows name + args)
      event: tool_result → tool returned, agent is reading the result
      (default event)    → text token of the final answer
      event: done        → agent finished (includes iteration count)

    This is the "glass box" view of the agent — users can see exactly what
    it searched for and what it found, not just the final answer. Critical
    for trust and debugging in production RAG systems.

    SSE event format for each type:
      event: tool_call
      data: {"tool": "search_code", "input": {"query": "backward pass"}}

      event: tool_result
      data: {"tool": "search_code", "output": "Source 1: engine.py..."}

      (default)
      data: According to the code...

      event: done
      data: {"iterations": 3}
    """
    import json

    def event_stream():
        try:
            for event in agent_svc.stream(question, repo_filter=repo):
                etype = event["type"]

                if etype == "tool_call":
                    payload = json.dumps({"tool": event["tool"], "input": event["input"]})
                    yield f"event: tool_call\ndata: {payload}\n\n"

                elif etype == "tool_result":
                    payload = json.dumps({"tool": event["tool"], "output": event["output"]})
                    yield f"event: tool_result\ndata: {payload}\n\n"

                elif etype == "token":
                    safe = event["text"].replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

                elif etype == "done":
                    payload = json.dumps({"iterations": event["iterations"]})
                    yield f"event: done\ndata: {payload}\n\n"
                    yield "data: [DONE]\n\n"
        except Exception as e:
            # Surface the real error to the frontend instead of silently closing
            err_msg = str(e)
            if "credit" in err_msg.lower() or "billing" in err_msg.lower():
                err_msg = "Anthropic API credits exhausted. Add credits at console.anthropic.com."
            elif "api_key" in err_msg.lower():
                err_msg = "ANTHROPIC_API_KEY not configured on this server."
            payload = json.dumps({"message": err_msg})
            yield f"event: error\ndata: {payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    """Simple health check — returns 200 if the server is running."""
    return {"status": "ok"}

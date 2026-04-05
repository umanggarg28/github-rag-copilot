"""
main.py — FastAPI application entry point.

NEW in this version: MCP (Model Context Protocol) integration.

Two things changed architecturally:

  1. We now run an MCP SERVER alongside the REST API.
     FastMCP.streamable_http_app() returns a Starlette ASGI app.
     We mount it at /mcp — so one process speaks both protocols:
       - REST API:  POST /ingest, GET /query/stream, etc.
       - MCP:       POST /mcp  (JSON-RPC 2.0 over HTTP)

  2. The agent now uses an MCP CLIENT instead of calling services directly.
     AgentService no longer imports RetrievalService.
     It connects to /mcp, discovers tools dynamically, calls them via protocol.
     This is exactly how Claude Code interacts with external tools.

Everything else (lifespan, dependency injection, CORS, SSE streaming) is unchanged.

Endpoints:
  POST /ingest                   — ingest a GitHub repo
  GET  /repos                    — list all indexed repos
  DELETE /repos/{owner}/{name}   — delete a repo
  POST /search                   — retrieve chunks (no generation)
  POST /query                    — RAG: retrieve + generate
  GET  /query/stream             — RAG with SSE streaming
  POST /agent/query              — Agentic RAG (synchronous)
  GET  /agent/stream             — Agentic RAG with SSE streaming
  GET  /repos/{owner}/{name}/tour    — guided concept tour (ExploreView)
  GET  /repos/{owner}/{name}/diagram — architecture / class diagrams
  GET  /mcp-status               — MCP server info (tools, resources, prompts)
  POST /mcp                      — MCP protocol endpoint (for MCP clients)
"""

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from pydantic import BaseModel
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
from backend.services.agent import AgentService, AGENT_MODELS
from backend.services.diagram_service import DiagramService
from backend.services.repo_map_service import RepoMapService
from backend.mcp_server import mcp, init_services as init_mcp_services
from backend.mcp_client import MCPClient
from retrieval.retrieval import RetrievalService, Reranker
from ingestion.qdrant_store import QdrantStore
from ingestion.embedder import Embedder


# ── Shared service instances ───────────────────────────────────────────────────

# In-memory staleness tracker: repo_slug → ISO timestamp of last successful ingest.
# Keyed by slug so it survives backend restarts only within a process lifetime.
# Simple and dependency-free — no need for a persistent store for this UX hint.
_repo_indexed_at: dict[str, str] = {}
_repo_contextual_at: dict[str, str] = {}  # slug → ISO timestamp of last contextual re-index

_ingestion_service:  IngestionService  | None = None
_retrieval_service:  RetrievalService  | None = None
_generation_service: GenerationService | None = None
_agent_service:      AgentService      | None = None
_diagram_service:    DiagramService    | None = None
_repo_map_service:   RepoMapService    | None = None
_mcp_client:         MCPClient         | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load models, connect to Qdrant, wire MCP server + client.
    Shutdown: (add cleanup here if needed).
    """
    global _ingestion_service, _retrieval_service, _generation_service
    global _agent_service, _diagram_service, _repo_map_service, _mcp_client

    print("Starting up — loading models and connecting to Qdrant...")

    # ── Single shared Embedder ─────────────────────────────────────────────────
    # The embedding model is 600MB. Loading it twice wastes ~600MB RAM.
    # We create one instance and pass it to both IngestionService (for indexing)
    # and RetrievalService (for query embedding). Same model, one load.
    _embedder = Embedder()

    # ── Single shared QdrantStore ──────────────────────────────────────────────
    # One client, one connection pool. All services use this same instance.
    # Previously we created 3 separate QdrantStore() calls — each opened its
    # own HTTP connection pool and auth session, wasting resources and making
    # it harder to reason about state.
    _qdrant_store = QdrantStore()

    # Core services — all share the same store + embedder instances
    _reranker           = Reranker()   # shared; model loads lazily on first search
    _generation_service = GenerationService()
    # Pass gen to RetrievalService so HyDE and query expansion can use the LLM.
    # Pass gen to IngestionService so contextual retrieval can use the LLM.
    # Both use the same GenerationService instance — one provider, no double-init.
    _retrieval_service  = RetrievalService(embedder=_embedder, store=_qdrant_store, reranker=_reranker, gen=_generation_service)
    _ingestion_service  = IngestionService(store=_qdrant_store, embedder=_embedder, gen=_generation_service)
    _diagram_service    = DiagramService(_qdrant_store, _generation_service)
    _repo_map_service   = RepoMapService(_qdrant_store)

    # ── MCP server setup ───────────────────────────────────────────────────────
    # Inject shared service instances into the MCP server's tool functions.
    init_mcp_services(_retrieval_service, _qdrant_store)

    # ── MCP client + agent setup ───────────────────────────────────────────────
    if settings.cerebras_api_key or settings.groq_api_key or settings.gemini_api_key or settings.openrouter_api_key or settings.anthropic_api_key:
        # MCP server is mounted in this same process — connect to ourselves.
        # Use PORT env var so this works on any host: local (8000), HF Spaces (7860).
        _port          = int(os.getenv("PORT", "8000"))
        _mcp_client    = MCPClient(server_url=f"http://localhost:{_port}/mcp")
        _agent_service = AgentService(_mcp_client, repo_map_svc=_repo_map_service)
        print("AgentService ready (MCP-powered agentic RAG enabled).")
    else:
        print("No GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY — /agent/* endpoints disabled.")

    print("All services ready.\n")

    # ── MCP session manager lifecycle ──────────────────────────────────────────
    # FastMCP's StreamableHTTPSessionManager requires an asyncio task group to
    # manage concurrent sessions. When mounting as a sub-app, we must start this
    # task group explicitly using session_manager.run() as an async context manager.
    # Without this, any MCP request raises "Task group is not initialized".
    # session_manager is lazily created when streamable_http_app() is first called
    # (which happens in the app.mount() call below the lifespan definition).
    async with mcp.session_manager.run():
        yield

    print("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cartographer",
    description="Ask questions about any GitHub repository. Powered by MCP.",
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allow any localhost port (Vite dev server picks 5173, 5174, 5175, 5176...
# depending on what's already running). Allow FRONTEND_URL in production.

_origins = []
if settings.frontend_url:
    _origins.append(settings.frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount MCP server ───────────────────────────────────────────────────────────
# FastMCP.streamable_http_app() returns a Starlette ASGI sub-application.
# Mounting it at "/mcp" means:
#   - All requests to /mcp are handled by the MCP server (JSON-RPC)
#   - All other requests are handled by FastAPI (REST)
#
# The FastMCP instance was created in mcp_server.py with streamable_http_path="/"
# so within the sub-app, the endpoint is at "/" → full URL becomes /mcp.
#
# Any MCP client (Claude Desktop, Claude Code, or our own MCPClient) can now
# connect to http://localhost:8000/mcp and discover + use our tools.

app.mount("/mcp", mcp.streamable_http_app())


# ── Rate limiter ───────────────────────────────────────────────────────────────
# Sliding window counter: track timestamps of recent requests per IP.
# On each request, drop timestamps older than 60s, then check the count.
# No external dependency — a deque per IP in a defaultdict is sufficient
# for a single-process server. For multi-process deployments, use Redis.

_rate_windows: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(request: Request) -> None:
    """
    Raise 429 if the caller has exceeded INGEST_RATE_LIMIT requests/minute.

    Uses the X-Forwarded-For header when behind a proxy (e.g. Render),
    falling back to request.client.host for direct connections.
    """
    limit = settings.ingest_rate_limit
    if limit <= 0:
        return  # disabled

    # Use the RIGHTMOST X-Forwarded-For entry — it's appended by the actual
    # proxy (Render, Cloudflare, etc.) and cannot be forged by the client.
    # The leftmost entry is user-controlled and trivially bypassable.
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[-1].strip() if forwarded else ""
    ip = ip or (request.client.host if request.client else "unknown")
    now = time.monotonic()

    window = _rate_windows[ip]
    # Drop timestamps older than 60 seconds
    while window and window[0] < now - 60:
        window.popleft()
    # Clean up empty deques so the dict doesn't grow unboundedly over time
    # (one entry per unique IP, never deleted otherwise — a slow memory leak).
    if not window:
        del _rate_windows[ip]
        return  # no prior requests in window — proceed

    if len(window) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {limit} requests per minute.",
        )

    window.append(now)


# ── Dependency providers ───────────────────────────────────────────────────────

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

def get_diagram_service() -> DiagramService:
    if _diagram_service is None:
        raise RuntimeError("DiagramService not initialised")
    return _diagram_service

def get_agent_service() -> AgentService:
    if _agent_service is None:
        raise HTTPException(
            status_code=503,
            detail="Agent mode requires GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY.",
        )
    return _agent_service



# ── Routes: MCP status ─────────────────────────────────────────────────────────

@app.get("/mcp-prompt", tags=["mcp"])
async def get_mcp_prompt(name: str, arguments: str = "{}"):
    """
    Expand an MCP prompt template and return the resulting text.

    Called by the frontend when a user selects a /prompt from the autocomplete.
    The prompt text is inserted into the chat textarea, ready to send.

    Args:
        name:      Prompt name (e.g. 'analyze_repo', 'explain_function')
        arguments: JSON-encoded dict of arguments (e.g. '{"repo":"karpathy/micrograd"}')
    """
    import json as _json
    try:
        args = _json.loads(arguments)
        result = await mcp.get_prompt(name, args)
        # Extract text from the first user message in the prompt
        text = ""
        for msg in result.messages:
            if hasattr(msg.content, "text"):
                text = msg.content.text
                break
        return {"name": name, "text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prompt error: {e}")


@app.get("/mcp-status", tags=["mcp"])
async def mcp_status():
    """
    Return live status of the connected MCP server.

    Shows all discovered tools, resources, and prompts.
    Used by the UI sidebar to display the MCP server panel.

    This endpoint demonstrates the client-side MCP API:
    the MCPClient connects, calls list_tools/list_resources/list_prompts,
    and returns what the server reports.
    """
    if _mcp_client is None:
        return {
            "connected": False,
            "error": "MCP client not initialized (no API key configured)",
            "tools": [], "resources": [], "prompts": [],
        }
    return await _mcp_client.get_server_info()


# ── Routes: Ingestion ──────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_repo(
    request: IngestRequest,
    svc: Annotated[IngestionService, Depends(get_ingestion_service)],
    _: None = Depends(_check_rate_limit),
):
    """
    Ingest a GitHub repository into the vector index.

    Downloads the repo zip, filters files, chunks by AST boundaries,
    embeds each chunk, and upserts to Qdrant. Safe to re-run (idempotent).
    Set force=true to delete and re-index from scratch.
    """
    try:
        # Ingestion is CPU+IO bound: downloads zip, runs AST parsing, embeds 600MB model.
        # Running it in the main event loop would block ALL other requests for minutes.
        # asyncio.to_thread() offloads it to a thread pool — the loop stays responsive.
        result = await asyncio.to_thread(svc.ingest, request.repo_url, request.force)
        # Invalidate stale diagrams and repo map — they were built from the old chunk set.
        if result.get("repo"):
            if _diagram_service:
                _diagram_service.invalidate(result["repo"])
            if _repo_map_service:
                _repo_map_service.invalidate(result["repo"])
        # Record ingestion timestamp for the staleness indicator in the UI.
        if result.get("repo"):
            now = datetime.now(timezone.utc).isoformat()
            _repo_indexed_at[result["repo"]] = now
            if request.force:
                _repo_contextual_at[result["repo"]] = now
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/ingest/stream", tags=["ingestion"])
async def ingest_stream(repo: str, request: Request):
    """
    Stream ingestion progress as Server-Sent Events (SSE).

    The client connects once and receives JSON events until the pipeline
    finishes or errors. Each event has the shape:
        { "step": "fetching|filtering|chunking|embedding|storing|done|error",
          "detail": "human-readable message" }

    Why SSE instead of WebSocket?
      SSE is a one-way, text-based HTTP stream — simpler to implement and
      natively supported by the browser EventSource API. Ingestion only needs
      server → client updates, so full-duplex WebSockets are unnecessary.

    Why asyncio.Queue + call_soon_threadsafe?
      ingest() is synchronous (CPU+IO bound). We run it in a thread via
      asyncio.to_thread(). The progress callback fires from that thread, but
      queue.put_nowait() is not thread-safe. call_soon_threadsafe() schedules
      the put on the event loop's thread, making it safe.
    """
    _check_rate_limit(request)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _progress(step: str, detail: str):
        # Called from the worker thread — schedule the enqueue on the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, {"step": step, "detail": detail})

    async def _run():
        try:
            await asyncio.to_thread(
                _ingestion_service.ingest,
                repo,
                False,  # force=False
                _progress,
            )
            # Invalidate stale diagrams and repo map after successful ingestion
            if _diagram_service:
                _diagram_service.invalidate(repo)
            if _repo_map_service:
                _repo_map_service.invalidate(repo)
            # Record ingestion timestamp for the staleness indicator
            _repo_indexed_at[repo] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"step": "error", "detail": str(e)})
        finally:
            # Sentinel value signals the event stream generator to stop.
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(_run())

    async def _event_stream():
        # Same keepalive pattern as the agent route: if no progress event arrives
        # within 15s (e.g. during embedding), send an SSE comment to prevent the
        # HF Spaces / Cloudflare proxy from dropping the idle connection.
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx/proxies not to buffer SSE — delivers events instantly.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/repos", response_model=ReposResponse, tags=["ingestion"])
async def list_repos(
    svc: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """List all indexed repositories and their chunk counts."""
    repos = svc.list_repos()
    total = sum(r["chunks"] for r in repos)
    return ReposResponse(
        repos=[
            RepoInfo(
                slug=r["slug"],
                chunks=r["chunks"],
                indexed_at=_repo_indexed_at.get(r["slug"]),
                # Read contextual_at from Qdrant payload — persists across restarts.
                # Falls back to in-memory dict for repos indexed in this session.
                contextual_at=(
                    svc.store.get_contextual_at(r["slug"])
                    or _repo_contextual_at.get(r["slug"])
                ),
            )
            for r in repos
        ],
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


@app.get("/repos/{owner}/{name}/tour", tags=["diagram"])
async def get_tour(
    owner:       str,
    name:        str,
    diagram_svc: Annotated[DiagramService, Depends(get_diagram_service)],
):
    """
    Generate (or return cached) a codebase tour for a repo.

    Returns a structured learning guide: 6-8 key concepts a student must
    understand, with descriptions, dependencies, and a suggested reading order.
    The frontend renders this as an interactive concept map (ExploreView).

    Unlike Mermaid diagrams, the tour returns structured JSON so the frontend
    can render rich interactive cards rather than a static SVG.
    """
    repo = f"{owner}/{name}"
    try:
        return await asyncio.to_thread(diagram_svc.build_tour, repo)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tour generation failed: {e}")


@app.get("/repos/{owner}/{name}/tour/stream", tags=["diagram"])
async def stream_tour(
    owner:       str,
    name:        str,
    diagram_svc: Annotated[DiagramService, Depends(get_diagram_service)],
    force:       bool = Query(False, description="If true, bypass cache and regenerate"),
):
    """
    Stream codebase tour generation as Server-Sent Events.

    Replaces the blank spinner with live progress events:
      { "stage": "loading|analysing|generating|parsing|done|error",
        "progress": 0.0-1.0,
        "message": "human-readable status" }   ← all stages except done/error
      { "stage": "done", "progress": 1.0, ...full_tour_data }

    Same data as GET /tour but streamed — no extra LLM calls, same token cost.
    """
    repo = f"{owner}/{name}"

    async def _event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                for event in diagram_svc.build_tour_stream(repo, force=force):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"stage": "error", "progress": 1.0, "error": str(e)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_running_loop().run_in_executor(None, _run)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/repos/{owner}/{name}/diagram/stream", tags=["diagram"])
async def stream_diagram(
    owner:        str,
    name:         str,
    diagram_svc:  Annotated[DiagramService, Depends(get_diagram_service)],
    type:         str  = Query("architecture", description="architecture | class"),
    force:        bool = Query(False, description="If true, bypass cache and regenerate"),
):
    """
    Stream diagram generation as Server-Sent Events.

    Progress stages:
      loading (0.10) → building (0.40) → enriching (0.70) → done (1.00)

    Final event: { "stage": "done", "diagram": {...}, "type": "architecture" }
    """
    repo = f"{owner}/{name}"

    async def _event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run():
            try:
                for event in diagram_svc.build_diagram_stream(repo, type, force=force):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"stage": "error", "progress": 1.0, "error": str(e)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_running_loop().run_in_executor(None, _run)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/repos/{owner}/{name}/diagram", tags=["diagram"])
async def get_diagram(
    owner:        str,
    name:         str,
    diagram_svc:  Annotated[DiagramService, Depends(get_diagram_service)],
    type:         str = Query("architecture", description="architecture | class | sequence | dataflow"),
):
    """
    Generate (or return cached) a Mermaid system design diagram for a repo.

    Four diagram types, each answering a different question:
      architecture — What are the main components and how do they connect?
      class        — What classes exist and how do they relate?
      sequence     — What happens step-by-step during the main operation?
      dataflow     — How does data move from input to output?

    The LLM reads indexed chunk names, generates valid Mermaid syntax,
    and returns it for the frontend to render. Results are cached per
    (repo, type) pair — re-ingest to regenerate.
    """
    repo = f"{owner}/{name}"
    try:
        return await asyncio.to_thread(diagram_svc.build_diagram, repo, type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagram generation failed: {e}")


# ── Routes: Search ─────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    request: SearchRequest,
    svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
):
    """Retrieve relevant code chunks without generating an answer."""
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
    """Full RAG pipeline: retrieve relevant code, then generate an answer."""
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
            answer="No relevant code found in the indexed repositories.",
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


class QueryStreamRequest(BaseModel):
    """Request body for POST /query/stream — streaming RAG with conversation history."""
    question:            str
    repo:                str | None  = None
    language:            str | None  = None
    top_k:               int | None  = None
    mode:                str         = "hybrid"
    relevance_threshold: float       = 0.0
    # Conversation history: prior [{role, content}] turns for follow-up questions.
    # Capped at 10 items (5 exchanges) to stay within LLM context limits.
    history:             list[dict]  = []


@app.post("/query/stream", tags=["query"])
async def query_stream(
    request: QueryStreamRequest,
    retrieval_svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
    generation_svc: Annotated[GenerationService, Depends(get_generation_service)],
):
    """
    Streaming RAG endpoint using Server-Sent Events (SSE).

    Accepts conversation history so follow-up questions have context.
    Sends a 'meta' event first (sources + query_type), then token events,
    then a final [DONE] sentinel. One SSE connection does everything.
    """
    question = request.question
    history  = request.history[-10:]  # cap at 5 exchanges
    query_type = classify_query(question)

    async def token_stream():
        import json

        # ── Phase 1: Retrieval (with keepalive) ───────────────────────────────
        # retrieval_svc.search() is blocking and can take 30-60s on first request
        # (HyDE LLM call + reranker model load). Run it in a background thread
        # and send keepalive pings every 15s so HF's proxy doesn't kill the
        # connection before sources arrive. Without this the frontend gets stuck
        # at "Searching code..." indefinitely.
        queue: asyncio.Queue = asyncio.Queue()

        def _run_retrieval():
            try:
                res, pipe = retrieval_svc.search(
                    query=question,
                    top_k=request.top_k,
                    repo_filter=request.repo,
                    language_filter=request.language,
                    mode=request.mode,
                    relevance_threshold=request.relevance_threshold,
                    return_pipeline=True,
                )
                queue.put_nowait(("results", (res, pipe)))
            except Exception as exc:
                queue.put_nowait(("error", exc))

        asyncio.get_event_loop().run_in_executor(None, _run_retrieval)

        results = pipeline_info = None
        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if kind == "results":
                results, pipeline_info = value
                break
            elif kind == "error":
                yield "data: ⚠ Retrieval failed. Please try again.\n\n"
                yield "data: [DONE]\n\n"
                return

        if not results:
            yield "data: No relevant code found in the indexed repositories.\n\n"
            yield "data: [DONE]\n\n"
            return

        context = retrieval_svc.format_context(results)

        # ── Phase 2: Send meta event (sources → frontend transitions to "generating") ─
        meta = {
            "sources":    [CodeChunk(**r).model_dump() for r in results],
            "query_type": query_type,
            "pipeline":   pipeline_info,
            "model":      generation_svc._model,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # ── Phase 3: Generation (with keepalive) ──────────────────────────────
        # Same queue pattern — generation_svc.stream() is also a blocking iterator.
        full_answer: list[str] = []

        def _run_generation():
            try:
                for token in generation_svc.stream(question, context, query_type, history=history):
                    queue.put_nowait(("token", token))
                answer = "".join(full_answer)
                grade = generation_svc.grade_answer(question, context, answer, query_type)
                queue.put_nowait(("grade", grade))
                queue.put_nowait(("done", None))
            except Exception as exc:
                queue.put_nowait(("error", exc))

        asyncio.get_event_loop().run_in_executor(None, _run_generation)

        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if kind == "token":
                full_answer.append(value)
                yield f"data: {value.replace(chr(10), chr(92) + 'n')}\n\n"
            elif kind == "grade":
                yield f"event: grade\ndata: {json.dumps(value)}\n\n"
            elif kind == "done":
                yield "data: [DONE]\n\n"
                break
            elif kind == "error":
                err = "⚠ All LLM providers are currently rate-limited. Please wait 60 seconds and retry."
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ── Routes: Agentic RAG ────────────────────────────────────────────────────────
#
# The agent now uses MCP for all tool calls.
# AgentService.stream() is an async generator — main.py uses `async for`
# to consume events and forward them to the SSE stream.
#
# Note on agent_stream: still avoids Depends(get_agent_service) for the same
# reason as before — HTTPException before SSE starts causes browser onerror.

@app.post("/agent/query", response_model=AgentResponse, tags=["agent"])
async def agent_query(
    request: AgentRequest,
    agent_svc: Annotated[AgentService, Depends(get_agent_service)],
):
    """
    Run the agentic RAG loop synchronously via MCP tools.

    The agent discovers tools from the MCP server, then runs the ReAct loop:
    search → observe → search again → observe → answer.
    Returns the full reasoning trace alongside the answer.
    """
    try:
        result = await agent_svc.run(request.question, repo_filter=request.repo)
        return AgentResponse(
            answer=result["answer"],
            tool_calls=[AgentToolCall(**tc) for tc in result["tool_calls"]],
            iterations=result["iterations"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


class AgentStreamRequest(BaseModel):
    """Request body for POST /agent/stream — agentic RAG with conversation history."""
    question: str
    repo:     str | None = None
    model_id: str | None = None   # catalog ID from /agent/models; None = auto priority chain
    # Conversation history: prior [{role, content}] turns for follow-up questions.
    history:  list[dict] = []


@app.get("/agent/models", tags=["agent"])
async def agent_models():
    """
    Return the list of available agent models with metadata for the model selector UI.

    Each entry has:
      id:          unique catalog ID sent back as model_id in /agent/stream requests
      name:        display name shown in the UI
      provider:    which API this model is served by
      speed:       "fast" | "slow" — used to show a visual indicator
      speed_label: human-readable time estimate (e.g. "~40s")
      note:        one-sentence tradeoff description shown in the tooltip / expanded row
      available:   whether the required API key is configured on this server
    """
    from backend.config import settings
    result = []
    for m in AGENT_MODELS:
        key_attr = m.get("requires", "")
        available = bool(getattr(settings, key_attr, ""))
        result.append({
            "id":          m["id"],
            "name":        m["name"],
            "provider":    m["provider"],
            "speed":       m["speed"],
            "speed_label": m["speed_label"],
            "note":        m["note"],
            "available":   available,
        })
    return {"models": result}


@app.post("/agent/stream", tags=["agent"])
async def agent_stream(request: AgentStreamRequest):
    """
    Run the agentic RAG loop with real-time SSE streaming.

    Accepts conversation history so the agent has context for follow-up questions.
    Streams tool_call and tool_result events as the agent reasons,
    then streams the final answer token by token.

    NOTE: Does not use Depends(get_agent_service).
    If we raised HTTPException(503) before the SSE stream opened, the browser's
    EventSource would see a non-200 response and fire onerror before any events
    could be received. By handling the None case inside the generator, we always
    return HTTP 200 and send errors as 'event: agent_error' frames.
    """
    import json

    svc      = _agent_service  # may be None if no API key configured
    question = request.question
    repo     = request.repo
    model_id = request.model_id
    history  = request.history[-10:]  # cap at 5 exchanges

    async def event_stream():
        if svc is None:
            msg = "Agent mode requires GROQ_API_KEY or ANTHROPIC_API_KEY in .env"
            yield f"event: agent_error\ndata: {json.dumps({'message': msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            # Keepalive via queue — the safe pattern for long-running SSE streams.
            #
            # Problem with asyncio.wait_for(gen.__anext__(), timeout=N):
            #   wait_for CANCELS the coroutine on timeout, which propagates
            #   CancelledError into the async generator, corrupting its state.
            #   After cancellation, subsequent __anext__() calls hang forever.
            #
            # Solution: run the generator in a separate background task that
            #   pushes events onto a queue. The main loop waits on queue.get()
            #   with a timeout. queue.get() being cancelled is safe — the item
            #   stays in the queue and is picked up on the next iteration.
            #   The producer task is never interrupted.
            queue: asyncio.Queue = asyncio.Queue()

            async def _producer():
                try:
                    async for event in svc.stream(question, repo_filter=repo, history=history, model_id=model_id):
                        await queue.put(("event", event))
                    await queue.put(("done", None))
                except Exception as exc:
                    await queue.put(("error", exc))

            asyncio.create_task(_producer())

            while True:
                try:
                    kind, value = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if kind == "done":
                    break
                if kind == "error":
                    raise value

                event = value
                etype = event["type"]

                if etype == "thought":
                    payload = json.dumps({"text": event["text"]})
                    yield f"event: thought\ndata: {payload}\n\n"

                elif etype == "tool_call":
                    payload = json.dumps({"tool": event["tool"], "input": event["input"]})
                    yield f"event: tool_call\ndata: {payload}\n\n"

                elif etype == "tool_result":
                    payload = json.dumps({"tool": event["tool"], "output": event["output"]})
                    yield f"event: tool_result\ndata: {payload}\n\n"

                elif etype == "token":
                    safe = event["text"].replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

                elif etype == "sources":
                    payload = json.dumps({"sources": event["sources"]})
                    yield f"event: sources\ndata: {payload}\n\n"

                elif etype == "done":
                    payload = json.dumps({"iterations": event["iterations"], "model": event.get("model", "")})
                    yield f"event: done\ndata: {payload}\n\n"
                    yield "data: [DONE]\n\n"

        except Exception as e:
            err_msg = str(e)
            err_lower = err_msg.lower()
            if any(kw in err_lower for kw in (
                "credit", "billing", "quota", "resource_exhausted", "daily limit",
                "rate_limit", "rate limit", "429",
            )):
                err_msg = (
                    "All LLM providers are currently rate-limited or at their daily limit. "
                    "Please wait a minute and try again."
                )
            elif "api_key" in err_lower:
                err_msg = "API key not configured. Check your .env file."
            yield f"event: agent_error\ndata: {json.dumps({'message': err_msg})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    """Simple liveness check."""
    return {"status": "ok"}

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
  GET  /mcp-status               — MCP server info (tools, resources, prompts)
  POST /mcp                      — MCP protocol endpoint (for MCP clients)
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
from backend.mcp_server import mcp, init_services as init_mcp_services
from backend.mcp_client import MCPClient
from retrieval.retrieval import RetrievalService
from ingestion.qdrant_store import QdrantStore


# ── Shared service instances ───────────────────────────────────────────────────

_ingestion_service:  IngestionService  | None = None
_retrieval_service:  RetrievalService  | None = None
_generation_service: GenerationService | None = None
_agent_service:      AgentService      | None = None
_graph_service:      GraphService      | None = None
_mcp_client:         MCPClient         | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load models, connect to Qdrant, wire MCP server + client.
    Shutdown: (add cleanup here if needed).
    """
    global _ingestion_service, _retrieval_service, _generation_service
    global _agent_service, _graph_service, _mcp_client

    print("Starting up — loading models and connecting to Qdrant...")

    # Core services (unchanged)
    _retrieval_service  = RetrievalService()
    _ingestion_service  = IngestionService()
    _graph_service      = GraphService(QdrantStore())
    _generation_service = GenerationService()

    # ── MCP server setup ───────────────────────────────────────────────────────
    # Inject shared service instances into the MCP server's tool functions.
    init_mcp_services(_retrieval_service, QdrantStore())

    # ── MCP client + agent setup ───────────────────────────────────────────────
    if settings.groq_api_key or settings.anthropic_api_key:
        _mcp_client    = MCPClient(server_url="http://localhost:8000/mcp")
        _agent_service = AgentService(_mcp_client)
        print("AgentService ready (MCP-powered agentic RAG enabled).")
    else:
        print("No GROQ_API_KEY or ANTHROPIC_API_KEY — /agent/* endpoints disabled.")

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
    title="GitHub RAG Copilot",
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

def get_graph_service() -> GraphService:
    if _graph_service is None:
        raise RuntimeError("GraphService not initialised")
    return _graph_service

def get_agent_service() -> AgentService:
    if _agent_service is None:
        raise HTTPException(
            status_code=503,
            detail="Agent mode requires GROQ_API_KEY or ANTHROPIC_API_KEY.",
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
):
    """
    Ingest a GitHub repository into the vector index.

    Downloads the repo zip, filters files, chunks by AST boundaries,
    embeds each chunk, and upserts to Qdrant. Safe to re-run (idempotent).
    Set force=true to delete and re-index from scratch.
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

    Returns nodes (functions/classes) and edges (call relationships)
    in D3-compatible format. Node size is proportional to caller_count.
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

    Sends a 'meta' event first (sources + query_type), then token events,
    then a final [DONE] sentinel. One SSE connection does everything.
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
        meta = {
            "sources":    [CodeChunk(**r).model_dump() for r in results],
            "query_type": query_type,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
        for token in generation_svc.stream(question, context, query_type):
            safe_token = token.replace("\n", "\\n")
            yield f"data: {safe_token}\n\n"
        yield "data: [DONE]\n\n"

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


@app.get("/agent/stream", tags=["agent"])
async def agent_stream(
    question: Annotated[str, Query(description="Question about the codebase")],
    repo: str | None = None,
):
    """
    Run the agentic RAG loop with real-time SSE streaming.

    Streams tool_call and tool_result events as the agent reasons,
    then streams the final answer token by token.

    NOTE: Does not use Depends(get_agent_service).
    If we raised HTTPException(503) before the SSE stream opened, the browser's
    EventSource would see a non-200 response and fire onerror before any events
    could be received. By handling the None case inside the generator, we always
    return HTTP 200 and send errors as 'event: agent_error' frames.
    """
    import json

    svc = _agent_service  # may be None if no API key configured

    async def event_stream():
        if svc is None:
            msg = "Agent mode requires GROQ_API_KEY or ANTHROPIC_API_KEY in .env"
            yield f"event: agent_error\ndata: {json.dumps({'message': msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            # async for — consumes the async generator from AgentService.stream()
            async for event in svc.stream(question, repo_filter=repo):
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
            err_msg = str(e)
            if "credit" in err_msg.lower() or "billing" in err_msg.lower():
                err_msg = "API credits exhausted. Check your billing dashboard."
            elif "api_key" in err_msg.lower():
                err_msg = "API key not configured. Check your .env file."
            yield f"event: agent_error\ndata: {json.dumps({'message': err_msg})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    """Simple liveness check."""
    return {"status": "ok"}

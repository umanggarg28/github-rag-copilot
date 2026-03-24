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
)
from backend.config import settings
from backend.services.ingestion_service import IngestionService
from backend.services.generation import GenerationService, classify_query
from retrieval.retrieval import RetrievalService


# ── Shared service instances ───────────────────────────────────────────────────
# These are module-level singletons initialised at startup. FastAPI's lifespan
# ensures they're ready before the first request arrives.

_ingestion_service: IngestionService | None = None
_retrieval_service: RetrievalService | None = None
_generation_service: GenerationService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context: runs startup code before yield, shutdown code after.

    Everything before `yield` runs when the server starts.
    Everything after `yield` runs when the server shuts down (Ctrl+C).

    Loading models here (not at import time) means startup errors are visible
    in the server log, not buried in a traceback from a module-level call.
    """
    global _ingestion_service, _retrieval_service, _generation_service
    print("Starting up — loading models and connecting to Qdrant...")
    _ingestion_service = IngestionService()
    _retrieval_service = RetrievalService()
    _generation_service = GenerationService()
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


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    """Simple health check — returns 200 if the server is running."""
    return {"status": "ok"}

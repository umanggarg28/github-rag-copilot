"""
main.py — FastAPI application entry point.

This file is intentionally thin: lifespan (service wiring), app creation,
CORS, MCP mount, router registration, and health check only.

All route logic lives in backend/routers/:
  ingestion.py   — /ingest, /repos
  query.py       — /search, /query, /query/stream
  agent.py       — /agent/*
  diagrams.py    — /repos/{owner}/{name}/tour|diagram
  mcp_routes.py  — /mcp-status, /mcp-prompt

Shared service instances and dependency providers live in backend/dependencies.py.
"""

import os
from contextlib import asynccontextmanager

import posthog as _posthog

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.dependencies import services
from backend.mcp_server import mcp, init_services as init_mcp_services
from backend.mcp_client import MCPClient
from backend.services.ingestion_service import IngestionService
from backend.services.generation import GenerationService
from backend.services.agent import AgentService
from backend.services.diagram_service import DiagramService
from backend.services.repo_map_service import RepoMapService
from backend.services.readme_service import ReadmeService
from retrieval.retrieval import RetrievalService, Reranker
from ingestion.qdrant_store import QdrantStore
from ingestion.embedder import Embedder

from backend.routers import ingestion, query, agent, diagrams, mcp_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise all services and wire MCP.
    Shutdown: flush analytics.

    Services are expensive to create (model loads, connection pools).
    We create them once here and store them on the `services` container
    imported by all routers via backend/dependencies.py.
    """
    if settings.posthog_api_key:
        _posthog.api_key = settings.posthog_api_key
        _posthog.host    = settings.posthog_host

    print("Starting up — loading models and connecting to Qdrant...")

    # One shared Embedder (600MB model) and one shared QdrantStore (connection pool).
    # Both are passed into every service that needs them — no duplicate loads.
    _embedder     = Embedder()
    _qdrant_store = QdrantStore()
    _reranker     = Reranker()

    services.generation = GenerationService()
    services.retrieval  = RetrievalService(
        embedder=_embedder, store=_qdrant_store,
        reranker=_reranker, gen=services.generation,
    )
    services.ingestion  = IngestionService(
        store=_qdrant_store, embedder=_embedder, gen=services.generation,
    )
    services.diagram    = DiagramService(_qdrant_store, services.generation)
    services.repo_map   = RepoMapService(_qdrant_store)
    services.readme     = ReadmeService(services.repo_map, services.generation)

    init_mcp_services(services.retrieval, _qdrant_store)

    if any([
        settings.cerebras_api_key, settings.groq_api_key,
        settings.gemini_api_key, settings.openrouter_api_key,
        settings.anthropic_api_key,
    ]):
        _port              = int(os.getenv("PORT", "8000"))
        services.mcp_client = MCPClient(server_url=f"http://localhost:{_port}/mcp")
        services.agent      = AgentService(services.mcp_client, repo_map_svc=services.repo_map)
        print("AgentService ready (MCP-powered agentic RAG enabled).")
    else:
        print("No LLM API key found — /agent/* endpoints disabled.")

    print("All services ready.\n")

    async with mcp.session_manager.run():
        yield

    if settings.posthog_api_key:
        _posthog.flush()

    print("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cartographer",
    description="Ask questions about any GitHub repository. Powered by MCP.",
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────

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

# ── MCP sub-app ────────────────────────────────────────────────────────────────
# Mounted before routers so /mcp is handled by FastMCP, not FastAPI.

app.mount("/mcp", mcp.streamable_http_app())

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(ingestion.router)
app.include_router(query.router)
app.include_router(agent.router)
app.include_router(diagrams.router)
app.include_router(mcp_routes.router)


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    """Simple liveness check."""
    return {"status": "ok"}

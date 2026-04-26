"""
dependencies.py — Shared service container + FastAPI dependency providers.

Why this file exists:
  main.py's lifespan() initialises all services once at startup.
  Every router needs access to those same instances.
  Putting them here avoids circular imports (routers can't import main.py
  because main.py imports the routers).

Pattern used: a mutable container object (`services`) whose *attributes* are
set at startup. Routers import `services` and read its attributes at
request-time via Depends(). This works because we mutate the object's
attributes (never rebind the name `services` itself), so all importers
always see the same object.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import posthog as _posthog

from fastapi import Depends, HTTPException, Request

from backend.config import settings

if TYPE_CHECKING:
    from backend.services.ingestion_service import IngestionService
    from backend.services.generation import GenerationService
    from backend.services.agent import AgentService
    from backend.services.diagram_service import DiagramService
    from backend.services.repo_map_service import RepoMapService
    from backend.services.readme_service import ReadmeService
    from backend.mcp_client import MCPClient
    from retrieval.retrieval import RetrievalService


# ── Service container ──────────────────────────────────────────────────────────
# One object; lifespan() sets its attributes; routers read them at request time.

class _Services:
    ingestion:  "IngestionService | None"  = None
    retrieval:  "RetrievalService | None"  = None
    generation: "GenerationService | None" = None
    agent:      "AgentService | None"      = None
    diagram:    "DiagramService | None"    = None
    repo_map:   "RepoMapService | None"    = None
    readme:     "ReadmeService | None"     = None
    mcp_client: "MCPClient | None"         = None

services = _Services()

# In-memory ingestion timestamps — repo_slug → ISO timestamp.
# Survives within a process lifetime; not persisted across restarts.
repo_indexed_at:     dict[str, str] = {}
repo_contextual_at:  dict[str, str] = {}


# ── Dependency providers ───────────────────────────────────────────────────────

def get_ingestion_service() -> "IngestionService":
    if services.ingestion is None:
        raise RuntimeError("IngestionService not initialised")
    return services.ingestion

def get_retrieval_service() -> "RetrievalService":
    if services.retrieval is None:
        raise RuntimeError("RetrievalService not initialised")
    return services.retrieval

def get_generation_service() -> "GenerationService":
    if services.generation is None:
        raise RuntimeError("GenerationService not initialised")
    return services.generation

def get_diagram_service() -> "DiagramService":
    if services.diagram is None:
        raise RuntimeError("DiagramService not initialised")
    return services.diagram

def get_readme_service() -> "ReadmeService":
    if services.readme is None:
        raise RuntimeError("ReadmeService not initialised")
    return services.readme

def get_agent_service() -> "AgentService":
    if services.agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent mode requires GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY.",
        )
    return services.agent

def get_qdrant_store():
    """Direct access to the shared QdrantStore for routes that don't go
    through a service (sessions sidecar, simple CRUD). Pulled off the
    ingestion service since both share the same connection-pooled instance
    initialised once at startup."""
    if services.ingestion is None:
        raise RuntimeError("QdrantStore not initialised")
    return services.ingestion.store


# ── Rate limiter ───────────────────────────────────────────────────────────────
# Sliding window per IP — no external dependency needed for a single process.

_rate_windows: dict[str, deque] = defaultdict(deque)

def check_rate_limit(request: Request) -> None:
    """Raise 429 if the caller has exceeded INGEST_RATE_LIMIT requests/minute."""
    limit = settings.ingest_rate_limit
    if limit <= 0:
        return

    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[-1].strip() if forwarded else ""
    ip = ip or (request.client.host if request.client else "unknown")
    now = time.monotonic()

    window = _rate_windows[ip]
    while window and window[0] < now - 60:
        window.popleft()

    if len(window) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {limit} requests per minute.",
        )
    window.append(now)


# ── PostHog helpers ────────────────────────────────────────────────────────────

def get_distinct_id(request: Request) -> str:
    """Extract the PostHog distinct ID sent by the frontend."""
    return request.headers.get("X-POSTHOG-DISTINCT-ID", "anonymous")

def ph_capture(distinct_id: str, event: str, properties: dict | None = None) -> None:
    """Fire a PostHog event only when analytics are configured."""
    if settings.posthog_api_key:
        _posthog.capture(event, distinct_id=distinct_id, properties=properties or {})

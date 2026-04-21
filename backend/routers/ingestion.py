"""
routers/ingestion.py — Repo ingestion, listing, and deletion endpoints.

Routes:
  POST   /ingest               — ingest a repo (sync)
  GET    /ingest/stream        — ingest with SSE progress
  GET    /repos                — list all indexed repos
  DELETE /repos/{owner}/{name} — delete a repo
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.dependencies import (
    services,
    repo_indexed_at, repo_contextual_at,
    get_ingestion_service, check_rate_limit,
    get_distinct_id, ph_capture,
)
from backend.models.schemas import IngestRequest, IngestResponse, ReposResponse, RepoInfo
from backend.services.ingestion_service import IngestionService
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["ingestion"])


def _safe_error_detail(error: Exception) -> str:
    """Redact provider credentials from errors before returning them to clients."""
    text = str(error)
    text = re.sub(r"([?&](?:key|api_key|token)=)[^&\s\"']+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "AIza[REDACTED]", text)
    return text


@router.post("/ingest", response_model=IngestResponse)
async def ingest_repo(
    request: IngestRequest,
    http_request: Request,
    svc: Annotated[IngestionService, Depends(get_ingestion_service)],
    _: None = Depends(check_rate_limit),
):
    """
    Ingest a GitHub repository into the vector index.

    Downloads the repo zip, filters files, chunks by AST boundaries,
    embeds each chunk, and upserts to Qdrant. Safe to re-run (idempotent).
    Set force=true to delete and re-index from scratch.
    """
    distinct_id = get_distinct_id(http_request)
    try:
        result = await asyncio.to_thread(svc.ingest, request.repo_url, request.force)
        if result.get("repo"):
            if services.diagram:
                services.diagram.invalidate(result["repo"])
            if services.repo_map:
                services.repo_map.invalidate(result["repo"])
            now = datetime.now(timezone.utc).isoformat()
            repo_indexed_at[result["repo"]] = now
            if request.force:
                repo_contextual_at[result["repo"]] = now
        ph_capture(distinct_id, "repo_ingested", {
            "repo": result.get("repo", ""),
            "chunks_indexed": result.get("chunks", 0),
            "force_reindex": request.force,
        })
        return IngestResponse(**result)
    except ValueError as e:
        ph_capture(distinct_id, "repo_ingestion_failed", {
            "repo_url": request.repo_url, "error_type": "validation",
        })
        raise HTTPException(status_code=400, detail=_safe_error_detail(e))
    except Exception as e:
        ph_capture(distinct_id, "repo_ingestion_failed", {
            "repo_url": request.repo_url, "error_type": "server",
        })
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {_safe_error_detail(e)}")


@router.get("/ingest/stream")
async def ingest_stream(repo: str, request: Request, force: bool = False):
    """
    Stream ingestion progress as Server-Sent Events (SSE).

    Each event: { "step": "fetching|filtering|chunking|embedding|storing|done|error",
                  "detail": "human-readable message" }

    force=true deletes and re-ingests from scratch (used by the re-index button).
    """
    check_rate_limit(request)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _progress(step: str, detail: str):
        loop.call_soon_threadsafe(queue.put_nowait, {"step": step, "detail": detail})

    async def _run():
        try:
            result = await asyncio.to_thread(services.ingestion.ingest, repo, force, _progress)
            repo_slug = result.get("repo", repo)
            if services.diagram:
                services.diagram.invalidate(repo_slug)
            if services.repo_map:
                services.repo_map.invalidate(repo_slug)
            now = datetime.now(timezone.utc).isoformat()
            repo_indexed_at[repo_slug] = now
            if force:
                repo_contextual_at[repo_slug] = now
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"step": "error", "detail": _safe_error_detail(e)},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    asyncio.create_task(_run())

    async def _event_stream():
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/repos", response_model=ReposResponse)
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
                indexed_at=repo_indexed_at.get(r["slug"]),
                contextual_at=(
                    svc.store.get_contextual_at(r["slug"])
                    or repo_contextual_at.get(r["slug"])
                ),
            )
            for r in repos
        ],
        total_chunks=total,
    )


@router.delete("/repos/{owner}/{name}")
async def delete_repo(
    owner: str,
    name: str,
    http_request: Request,
    svc: Annotated[IngestionService, Depends(get_ingestion_service)],
):
    """Delete all chunks for a repository from the index."""
    slug = f"{owner}/{name}"
    deleted = svc.delete_repo(slug)
    ph_capture(get_distinct_id(http_request), "repo_deleted", {
        "repo": slug, "chunks_deleted": deleted,
    })
    return {"repo": slug, "chunks_deleted": deleted}

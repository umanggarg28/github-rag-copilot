"""
routers/diagrams.py — Codebase tour and architecture diagram endpoints.

Routes:
  GET /repos/{owner}/{name}/tour           — concept tour (sync)
  GET /repos/{owner}/{name}/tour/stream    — concept tour with SSE progress
  GET /repos/{owner}/{name}/diagram        — architecture/class diagram (sync)
  GET /repos/{owner}/{name}/diagram/stream — diagram with SSE progress
"""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.dependencies import get_diagram_service, get_distinct_id, ph_capture
from backend.services.diagram_service import DiagramService

router = APIRouter(tags=["diagram"])


@router.get("/repos/{owner}/{name}/tour")
async def get_tour(
    owner:       str,
    name:        str,
    diagram_svc: Annotated[DiagramService, Depends(get_diagram_service)],
):
    """Generate (or return cached) a codebase concept tour."""
    repo = f"{owner}/{name}"
    try:
        return await asyncio.to_thread(diagram_svc.build_tour, repo)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tour generation failed: {e}")


@router.get("/repos/{owner}/{name}/tour/stream")
async def stream_tour(
    owner:        str,
    name:         str,
    http_request: Request,
    diagram_svc:  Annotated[DiagramService, Depends(get_diagram_service)],
    force:        bool = Query(False, description="Bypass cache and regenerate"),
):
    """
    Stream concept tour generation as SSE.

    Events: { "stage": "loading|analysing|generating|done|error",
              "progress": 0.0-1.0, "message": "..." }
    Final:  { "stage": "done", "progress": 1.0, ...tour_data }
    """
    repo        = f"{owner}/{name}"
    distinct_id = get_distinct_id(http_request)

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
            if event.get("stage") == "done":
                ph_capture(distinct_id, "tour_generated", {
                    "repo": repo, "from_cache": not force,
                })
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/repos/{owner}/{name}/diagram")
async def get_diagram(
    owner:        str,
    name:         str,
    diagram_svc:  Annotated[DiagramService, Depends(get_diagram_service)],
    type:         str = Query("architecture", description="architecture | class"),
):
    """Generate (or return cached) an architecture or class diagram."""
    repo = f"{owner}/{name}"
    try:
        return await asyncio.to_thread(diagram_svc.build_diagram, repo, type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagram generation failed: {e}")


@router.get("/repos/{owner}/{name}/diagram/stream")
async def stream_diagram(
    owner:        str,
    name:         str,
    http_request: Request,
    diagram_svc:  Annotated[DiagramService, Depends(get_diagram_service)],
    type:         str  = Query("architecture", description="architecture | class"),
    force:        bool = Query(False, description="Bypass cache and regenerate"),
):
    """
    Stream diagram generation as SSE.

    Stages: loading (0.10) → building (0.40) → enriching (0.70) → done (1.00)
    Final event: { "stage": "done", "diagram": {...}, "type": "architecture" }
    """
    repo        = f"{owner}/{name}"
    distinct_id = get_distinct_id(http_request)

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
            if event.get("stage") == "done":
                ph_capture(distinct_id, "diagram_generated", {
                    "repo": repo, "diagram_type": type, "from_cache": not force,
                })
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

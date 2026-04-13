"""
routers/agent.py — Agentic RAG endpoints (MCP-powered ReAct loop).

Routes:
  GET  /agent/models  — list available agent models
  POST /agent/query   — run agent loop (sync)
  POST /agent/stream  — run agent loop with SSE streaming
"""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.dependencies import (
    services, get_agent_service,
    get_distinct_id, ph_capture,
)
from backend.models.schemas import AgentRequest, AgentResponse, AgentToolCall
from backend.services.agent import AgentService, AGENT_MODELS
from backend.config import settings

router = APIRouter(tags=["agent"])


@router.get("/agent/models")
async def agent_models():
    """Return available agent models with metadata for the model selector UI."""
    result = []
    for m in AGENT_MODELS:
        key_attr  = m.get("requires", "")
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


@router.post("/agent/query", response_model=AgentResponse)
async def agent_query(
    request: AgentRequest,
    agent_svc: Annotated[AgentService, Depends(get_agent_service)],
):
    """Run the agentic RAG loop synchronously via MCP tools."""
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
    model_id: str | None = None
    history:  list[dict] = []


@router.post("/agent/stream")
async def agent_stream(request: AgentStreamRequest, http_request: Request):
    """
    Run the agentic RAG loop with real-time SSE streaming.

    NOTE: Does not use Depends(get_agent_service) — if we raised HTTPException(503)
    before the SSE stream opened, the browser's EventSource would see a non-200
    response and fire onerror before any events could be received. Errors are
    sent as 'event: agent_error' frames instead.
    """
    svc         = services.agent
    question    = request.question
    repo        = request.repo
    model_id    = request.model_id
    history     = request.history[-10:]
    distinct_id = get_distinct_id(http_request)

    async def event_stream():
        if svc is None:
            msg = "Agent mode requires GROQ_API_KEY or ANTHROPIC_API_KEY in .env"
            yield f"event: agent_error\ndata: {json.dumps({'message': msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            # Queue-based keepalive pattern — see main.py comment for full explanation.
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
                    yield f"event: thought\ndata: {json.dumps({'text': event['text']})}\n\n"
                elif etype == "tool_call":
                    yield f"event: tool_call\ndata: {json.dumps({'tool': event['tool'], 'input': event['input']})}\n\n"
                elif etype == "tool_result":
                    yield f"event: tool_result\ndata: {json.dumps({'tool': event['tool'], 'output': event['output']})}\n\n"
                elif etype == "token":
                    yield f"data: {event['text'].replace(chr(10), chr(92) + 'n')}\n\n"
                elif etype == "sources":
                    yield f"event: sources\ndata: {json.dumps({'sources': event['sources']})}\n\n"
                elif etype == "done":
                    ph_capture(distinct_id, "agent_query_completed", {
                        "repo": repo or "",
                        "iterations": event["iterations"],
                        "model": event.get("model", ""),
                        "has_history": len(history) > 0,
                    })
                    payload = json.dumps({"iterations": event["iterations"], "model": event.get("model", "")})
                    yield f"event: done\ndata: {payload}\n\n"
                    yield "data: [DONE]\n\n"

        except Exception as e:
            err_msg   = str(e)
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

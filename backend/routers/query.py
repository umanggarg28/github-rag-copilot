"""
routers/query.py — Search and RAG query endpoints.

Routes:
  POST /search        — retrieve chunks, no generation
  POST /query         — RAG: retrieve + generate (sync)
  POST /query/stream  — RAG with SSE streaming + conversation history
"""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.dependencies import (
    get_retrieval_service, get_generation_service,
    get_distinct_id, ph_capture,
)
from backend.models.schemas import (
    SearchRequest, SearchResponse, CodeChunk,
    QueryRequest, QueryResponse,
)
from backend.services.generation import GenerationService, classify_query
from retrieval.retrieval import RetrievalService

router = APIRouter(tags=["query"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    http_request: Request,
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
    ph_capture(get_distinct_id(http_request), "code_search_performed", {
        "repo": request.repo or "", "mode": request.mode, "result_count": len(results),
    })
    return SearchResponse(
        query=request.query,
        results=[CodeChunk(**r) for r in results],
        total=len(results),
    )


@router.post("/query", response_model=QueryResponse)
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
            sources=[], query_type=query_type,
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
    history:             list[dict]  = []


@router.post("/query/stream")
async def query_stream(
    request: QueryStreamRequest,
    http_request: Request,
    retrieval_svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
    generation_svc: Annotated[GenerationService, Depends(get_generation_service)],
):
    """
    Streaming RAG endpoint using Server-Sent Events (SSE).

    Sends a 'meta' event first (sources + query_type), then token events,
    then a final [DONE] sentinel.
    """
    question    = request.question
    history     = request.history[-10:]
    query_type  = classify_query(question)
    distinct_id = get_distinct_id(http_request)

    async def token_stream():
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
        meta = {
            "sources":    [CodeChunk(**r).model_dump() for r in results],
            "query_type": query_type,
            "pipeline":   pipeline_info,
            "model":      generation_svc._model,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

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
                ph_capture(distinct_id, "rag_query_completed", {
                    "repo": request.repo or "",
                    "mode": request.mode,
                    "query_type": query_type,
                    "source_count": len(results) if results else 0,
                    "has_history": len(history) > 0,
                })
                yield "data: [DONE]\n\n"
                break
            elif kind == "error":
                err = "⚠ All LLM providers are currently rate-limited. Please wait 60 seconds and retry."
                yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(token_stream(), media_type="text/event-stream")

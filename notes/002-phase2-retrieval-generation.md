# 002 — Phase 2: Retrieval & Generation

## What was built

Five files complete the backend:

| File | Purpose |
|------|---------|
| `retrieval/retrieval.py` | Hybrid, semantic, keyword search over Qdrant |
| `backend/models/schemas.py` | Pydantic models for all API request/response types |
| `backend/services/generation.py` | LLM wrapper (Groq/Anthropic), query classifier, streaming |
| `backend/services/ingestion_service.py` | Pipeline orchestrator — glues Phase 1 modules together |
| `backend/main.py` | FastAPI app — lifespan startup, routes, CORS, SSE streaming |

## Key decisions

**Qdrant native hybrid search** — `Prefetch` + `FusionQuery(Fusion.RRF)` in one network call.
Alternative (manual BM25 in Python + separate round trips) is 3x slower and less accurate
because Qdrant's fusion has access to the full index (IDF weighting uses corpus statistics).

**Weighted signal classifier** — `classify_query()` uses weighted keyword dicts instead of
an ML classifier. Zero training data needed, fully interpretable, trivially tunable.
Technical wins ties — precision is more valuable than creativity for code questions.

**Two LLM providers** — Groq (primary) → Anthropic (fallback). Checked at startup,
same interface (`answer()` / `stream()`). Router is provider-agnostic.

**Lifespan for model loading** — FastAPI's `@asynccontextmanager` lifespan ensures the
600MB embedding model loads once at startup, not per request. Services are shared via
`Depends()` injection.

**SSE streaming** — GET endpoint (not POST) because browser `EventSource` only supports GET.
Tokens are newline-escaped to prevent breaking the `data: ...\n\n` SSE event format.

## How to test

```bash
# Assumes karpathy/micrograd is already ingested (run demo_ingestion.py first)
python demo_query.py

# Or start the API server and hit /docs
uvicorn backend.main:app --reload
# open http://localhost:8000/docs
```

## What's next

Phase 3: React UI
- Repo URL input → triggers /ingest
- Chat interface → calls /query/stream with SSE
- Source citations with filepath + line numbers
- Repo selector dropdown (populated from /repos)
- Syntax highlighting for code chunks

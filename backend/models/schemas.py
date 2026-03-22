"""
schemas.py — Pydantic request and response models for the FastAPI API.

Why Pydantic?
  FastAPI uses Pydantic models to:
    1. Parse and validate incoming JSON request bodies
    2. Serialize outgoing Python objects to JSON responses
    3. Auto-generate OpenAPI/Swagger docs at /docs

  If a required field is missing or the wrong type, FastAPI returns a 422
  Unprocessable Entity error automatically — no manual validation needed.

  Pydantic v2 (used here) is significantly faster than v1 — it compiles
  validators to Rust via pydantic-core.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request body for POST /ingest — kick off repo ingestion."""
    repo_url: str = Field(
        ...,
        description="GitHub repository URL (e.g. https://github.com/owner/repo)",
        examples=["https://github.com/karpathy/micrograd"],
    )
    force: bool = Field(
        default=False,
        description="If true, delete existing chunks for this repo before re-ingesting",
    )


class IngestResponse(BaseModel):
    """Response after ingestion completes."""
    repo: str           # "owner/name" slug
    files_indexed: int  # number of files processed
    chunks_stored: int  # number of chunks upserted to Qdrant
    message: str        # human-readable summary


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request body for POST /search — retrieve chunks without generating an answer."""
    query: str = Field(..., description="Natural language or code snippet")
    repo: Optional[str] = Field(
        default=None,
        description="Restrict to a specific repo slug (e.g. 'karpathy/micrograd')",
    )
    language: Optional[str] = Field(
        default=None,
        description="Restrict to a programming language (e.g. 'python')",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of results (defaults to TOP_K env var)",
    )
    mode: Literal["semantic", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="Search strategy: semantic (dense), keyword (BM25), or hybrid (both)",
    )


class CodeChunk(BaseModel):
    """A single retrieved code chunk — one item in search results."""
    text: str
    repo: str
    filepath: str
    language: str
    chunk_type: str   # "function", "class", "module", "text"
    name: str         # function/class name (empty for module-level chunks)
    start_line: int
    end_line: int
    score: float


class SearchResponse(BaseModel):
    """Response from POST /search."""
    query: str
    results: list[CodeChunk]
    total: int


# ── Query (RAG) ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query — full RAG: retrieve + generate answer."""
    question: str = Field(..., description="Question about the codebase")
    repo: Optional[str] = Field(
        default=None,
        description="Restrict context to a specific repo slug",
    )
    language: Optional[str] = Field(
        default=None,
        description="Restrict context to a programming language",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of context chunks to retrieve",
    )
    mode: Literal["semantic", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval strategy",
    )
    relevance_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval score — queries below this return a 'not found' message",
    )


class QueryResponse(BaseModel):
    """Response from POST /query."""
    question: str
    answer: str
    sources: list[CodeChunk]
    query_type: str   # "technical" or "creative" — which LLM profile was used


# ── Repos ─────────────────────────────────────────────────────────────────────

class RepoInfo(BaseModel):
    """Metadata for a single indexed repo."""
    slug: str   # "owner/name"
    chunks: int # number of chunks in the index


class ReposResponse(BaseModel):
    """Response from GET /repos — list all indexed repos."""
    repos: list[RepoInfo]
    total_chunks: int

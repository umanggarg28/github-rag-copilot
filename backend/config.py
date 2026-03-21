"""
config.py — Central configuration via environment variables.

All settings live here. Components import from this file rather than
reading os.environ directly — one place to change, one place to document.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str       = os.getenv("GROQ_API_KEY", "")
    anthropic_api_key: str  = os.getenv("ANTHROPIC_API_KEY", "")

    # ── Vector DB ─────────────────────────────────────────────────────────────
    qdrant_url: str         = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str     = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str  = os.getenv("QDRANT_COLLECTION", "github_repos")

    # ── GitHub ────────────────────────────────────────────────────────────────
    # Optional — without it you get 60 API req/hr; with it 5,000 req/hr
    github_token: str       = os.getenv("GITHUB_TOKEN", "")

    # ── Embedding model ───────────────────────────────────────────────────────
    # Default: all-MiniLM-L6-v2 (384-dim, general text, small/fast, already cached).
    # Upgrade to nomic-ai/nomic-embed-code (768-dim, code-optimised) when disk
    # space allows — set EMBEDDING_MODEL in .env and update QDRANT_COLLECTION
    # to a fresh collection (different dim = incompatible with existing points).
    embedding_model: str    = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int      = int(os.getenv("EMBEDDING_DIM", "384"))

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Used as fallback for non-Python files (markdown, config, plain text).
    # AST chunking is used for Python and ignores these values.
    chunk_size: int         = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int      = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int              = int(os.getenv("TOP_K", "8"))


settings = Settings()

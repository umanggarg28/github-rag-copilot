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
    # nomic-embed-code is fine-tuned on code and produces 768-dim vectors.
    # It uses a prefix convention for passages vs queries (see embedder.py).
    embedding_model: str    = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-code")
    embedding_dim: int      = 768  # nomic-embed-code output dimension

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Used as fallback for non-Python files (markdown, config, plain text).
    # AST chunking is used for Python and ignores these values.
    chunk_size: int         = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int      = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int              = int(os.getenv("TOP_K", "8"))


settings = Settings()

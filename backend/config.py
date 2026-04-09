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
    cerebras_api_key: str     = os.getenv("CEREBRAS_API_KEY", "")     # free: https://cloud.cerebras.ai (1M tok/day, 2600 tok/s)
    groq_api_key: str         = os.getenv("GROQ_API_KEY", "")         # free: https://console.groq.com
    gemini_api_key: str       = os.getenv("GEMINI_API_KEY", "")       # free: https://aistudio.google.com
    openrouter_api_key: str   = os.getenv("OPENROUTER_API_KEY", "")   # free: https://openrouter.ai
    anthropic_api_key: str    = os.getenv("ANTHROPIC_API_KEY", "")    # paid fallback

    # ── Vector DB ─────────────────────────────────────────────────────────────
    qdrant_url: str         = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str     = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str  = os.getenv("QDRANT_COLLECTION", "github_repos")

    # ── GitHub ────────────────────────────────────────────────────────────────
    # Optional — without it you get 60 API req/hr; with it 5,000 req/hr
    github_token: str       = os.getenv("GITHUB_TOKEN", "")

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Two embedding providers, selected at startup:
    #
    # 1. Voyage AI (VOYAGE_API_KEY set + EMBEDDING_MODEL=voyage-code-3)
    #    voyage-code-3: code-optimised, 1024-dim, 200M tokens/month free.
    #    ⚠️  Requires EMBEDDING_DIM=1024 and a NEW Qdrant collection — dims
    #    are incompatible with nomic (768-dim) collections.
    #
    # 2. Nomic API (default, NOMIC_API_KEY required)
    #    nomic-embed-text-v1.5: general text, 768-dim, generous free tier.
    #    Free at https://atlas.nomic.ai (no credit card needed).
    #
    # EMBEDDING_DIM must match the chosen model exactly.
    nomic_api_key: str      = os.getenv("NOMIC_API_KEY", "")
    voyage_api_key: str     = os.getenv("VOYAGE_API_KEY", "")
    embedding_model: str    = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
    embedding_dim: int      = int(os.getenv("EMBEDDING_DIM", "768"))

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Used as fallback for non-Python files (markdown, config, plain text).
    # AST chunking is used for Python and ignores these values.
    chunk_size: int         = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int      = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int              = int(os.getenv("TOP_K", "12"))

    # ── Quality: Reranking ────────────────────────────────────────────────────
    # Cohere Rerank API — cross-attention reranker, significantly better than
    # local ms-marco cross-encoder. 1000 calls/month free at https://cohere.com
    # Falls back to local cross-encoder (80MB) if COHERE_API_KEY is not set.
    cohere_api_key: str     = os.getenv("COHERE_API_KEY", "")

    # ── Quality: Query Enhancement ────────────────────────────────────────────
    # HyDE (Hypothetical Document Embeddings):
    #   Generate a hypothetical code snippet that would answer the query, then
    #   embed THAT instead of the raw question. Bridges the semantic gap between
    #   natural language questions and code. Adds ~1-2s (one LLM call).
    #   Requires a configured LLM key (Groq, Gemini, etc.).
    use_hyde: bool          = os.getenv("USE_HYDE", "true").lower() == "true"

    # Query Expansion:
    #   Generate 2-3 rephrased variants of the query, search all, merge with RRF.
    #   Captures synonyms, different identifiers, and alternative phrasings.
    #   Adds ~1-2s (one LLM call). Requires a configured LLM key.
    expand_queries: bool    = os.getenv("EXPAND_QUERIES", "true").lower() == "true"

    # ── Quality: Contextual Retrieval ─────────────────────────────────────────
    # How many chunks to enrich with LLM-generated context descriptions on
    # force re-index. Each enriched chunk gets a context sentence prepended
    # before embedding so the vector captures "architectural role" not just syntax.
    # 0 = all chunks (best quality, slower for large repos)
    # 50 = original conservative limit (fast, only enriches top chunks)
    contextual_top_n: int   = int(os.getenv("CONTEXTUAL_TOP_N", "0"))

    # ── Deployment ────────────────────────────────────────────────────────────
    # Set FRONTEND_URL in HuggingFace Spaces / Render to your Vercel URL
    # so CORS allows the deployed frontend to call the backend.
    frontend_url: str       = os.getenv("FRONTEND_URL", "")

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # Max /ingest requests per IP per minute. Each ingestion downloads a repo,
    # runs the embedding model, and writes to Qdrant — it's expensive.
    # Set to 0 to disable rate limiting (e.g. in local dev).
    ingest_rate_limit: int  = int(os.getenv("INGEST_RATE_LIMIT", "5"))

    # ── Analytics ─────────────────────────────────────────────────────────────
    posthog_api_key: str    = os.getenv("POSTHOG_API_KEY", "")
    posthog_host: str       = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")


settings = Settings()

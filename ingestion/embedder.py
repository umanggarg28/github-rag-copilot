"""
embedder.py — Embed code chunks via Voyage AI or Nomic API.

WHY API-BASED EMBEDDINGS
─────────────────────────
The local sentence-transformers model (nomic-embed-code) is ~600MB RAM.
That kills free-tier hosting (HF Spaces, Render: 512MB–1GB RAM limit).
Both APIs use the same underlying model — vectors are equivalent quality.
Zero RAM cost on our server, just network latency (~200ms/batch).

TWO PROVIDERS, ONE INTERFACE
──────────────────────────────
Provider selection happens at init time, based on env vars:

  VOYAGE_API_KEY set + EMBEDDING_MODEL=voyage-code-3
    → Voyage AI: code-optimised, 1024-dim, 200M tokens/month free.
      voyage-code-3 is specifically trained on code and outperforms
      general-purpose embedders on code retrieval benchmarks.
      ⚠️  Requires new Qdrant collection (dim mismatch with 768-dim).

  NOMIC_API_KEY set (default)
    → Nomic API: nomic-embed-text-v1.5, 768-dim, generous free tier.

TASK TYPES
───────────
Both APIs distinguish between document and query roles:
  - "search_document" / "document": used when indexing chunks
  - "search_query"   / "query":     used when embedding user queries

This produces a better inner-product space than treating both the same.

BATCHING
─────────
Both APIs accept up to 256-512 texts per call. We batch in groups of 96
(conservative) to avoid timeout on large text chunks over free-tier networks.
"""

import time
from pathlib import Path
import sys

import requests as http

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings


_NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
_BATCH_SIZE    = 96   # conservative batch — avoids timeout on large chunks


class Embedder:
    """
    Unified embedding client for Voyage AI and Nomic APIs.

    Provider selection at init:
      - Voyage AI if VOYAGE_API_KEY is set and model contains "voyage"
      - Nomic otherwise (requires NOMIC_API_KEY)

    Both providers use the same public interface:
      embed_chunks(chunks) → list of 1024-dim or 768-dim vectors
      embed_query(query)   → single vector of same dim
    """

    def __init__(self, model_name: str = None):
        self.model_name    = model_name or settings.embedding_model
        self.embedding_dim = settings.embedding_dim

        # Select provider based on available keys + model name
        if settings.voyage_api_key and "voyage" in self.model_name.lower():
            self._provider = "voyage"
            self._init_voyage()
        else:
            self._provider = "nomic"
            self._init_nomic()

    def _init_voyage(self):
        """Initialise Voyage AI client. voyage-code-3 is code-optimised 1024-dim."""
        try:
            import voyageai
            self._voyage = voyageai.Client(api_key=settings.voyage_api_key)
        except ImportError:
            raise ImportError(
                "voyageai package not installed. Run: pip install voyageai"
            )
        print(
            f"Embedder: using Voyage AI ({self.model_name}, {self.embedding_dim}-dim). "
            "No local model loaded."
        )

    def _init_nomic(self):
        """Initialise Nomic API client. nomic-embed-text-v1.5 is 768-dim."""
        if not settings.nomic_api_key:
            raise RuntimeError(
                "No embedding provider configured. "
                "Set NOMIC_API_KEY (free at https://atlas.nomic.ai) or "
                "VOYAGE_API_KEY + EMBEDDING_MODEL=voyage-code-3."
            )
        self._nomic_key = settings.nomic_api_key
        print(
            f"Embedder: using Nomic API ({self.model_name}, {self.embedding_dim}-dim). "
            "No local model loaded."
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """
        Embed a list of chunk dicts for indexing (document role).

        task_type="search_document" / input_type="document" tells the model
        these are passages being stored — it applies a different projection
        than for queries, which improves retrieval precision.
        """
        texts = [c["text"] for c in chunks]
        if self._provider == "voyage":
            return self._voyage_embed(texts, input_type="document")
        return self._nomic_embed(texts, task_type="search_document")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query for retrieval (query role).

        The "query" projection is the counterpart to "document" — vectors
        from both live in the same space but with optimised projections.
        """
        if self._provider == "voyage":
            return self._voyage_embed([query], input_type="query")[0]
        return self._nomic_embed([query], task_type="search_query")[0]

    # ── Voyage AI implementation ───────────────────────────────────────────────

    def _voyage_embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """
        Call Voyage AI API with batching.

        voyage-code-3 is specifically trained on (code, docstring) pairs
        and GitHub issues, giving it much better code retrieval than
        general-purpose text embedders.

        Batching: Voyage API accepts up to 128 texts per call on free tier.
        We use 96 to leave headroom for large chunks.
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch  = texts[i : i + _BATCH_SIZE]
            result = self._voyage_call_api(batch, input_type)
            all_embeddings.extend(result)
        return all_embeddings

    def _voyage_call_api(
        self,
        texts: list[str],
        input_type: str,
        retries: int = 2,
    ) -> list[list[float]]:
        """Single Voyage API call with retry on rate limit."""
        for attempt in range(retries + 1):
            try:
                result = self._voyage.embed(
                    texts,
                    model=self.model_name,
                    input_type=input_type,
                )
                return result.embeddings
            except Exception as e:
                msg = str(e).lower()
                if ("rate" in msg or "429" in msg or "503" in msg) and attempt < retries:
                    wait = 10
                    print(f"Voyage API rate limit. Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Voyage API call failed after retries")

    # ── Nomic API implementation ───────────────────────────────────────────────

    def _nomic_embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Call Nomic Atlas API with batching. Returns list of 768-dim vectors."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch      = texts[i : i + _BATCH_SIZE]
            embeddings = self._nomic_call_api(batch, task_type)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _nomic_call_api(
        self,
        texts: list[str],
        task_type: str,
        retries: int = 2,
    ) -> list[list[float]]:
        """
        Single Nomic API call with retry on rate limit (429) or service error (503).

        The Nomic API returns:
          { "embeddings": [[float, ...], ...], "usage": {...} }
        """
        headers = {
            "Authorization": f"Bearer {self._nomic_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "texts":     texts,
            "model":     self.model_name,
            "task_type": task_type,
        }

        for attempt in range(retries + 1):
            response = http.post(_NOMIC_API_URL, headers=headers, json=payload, timeout=60)

            if response.status_code in (429, 503) and attempt < retries:
                wait = int(response.headers.get("Retry-After", "10"))
                print(f"Nomic API {response.status_code}. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()["embeddings"]

        raise RuntimeError("Nomic API call failed after retries")

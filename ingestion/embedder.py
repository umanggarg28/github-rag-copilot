"""
embedder.py — Embed code chunks via a hosted embedding API.

WHY API-BASED EMBEDDINGS
─────────────────────────
Local sentence-transformers models are ~600MB RAM — enough to kill
free-tier hosting (HF Spaces, Render: 512MB–1GB RAM limit). Hosted
APIs give us zero RAM cost and equivalent quality, at the price of
~200ms of network latency per batch.

THREE PROVIDERS, ONE INTERFACE
──────────────────────────────
Provider is selected from EMBEDDING_MODEL at init:

  EMBEDDING_MODEL contains "voyage" + VOYAGE_API_KEY set  (default)
    → Voyage AI: code-optimised, 1024-dim, 200M tokens/month free.
      voyage-code-3 is specifically trained on code and outperforms
      general-purpose embedders on code retrieval benchmarks.
      Requires EMBEDDING_DIM=1024 and a new Qdrant collection.

  EMBEDDING_MODEL contains "gemini" + GEMINI_API_KEY set
    → Google Gemini: gemini-embedding-001, 768-dim output (configurable
      via MRL), generous free tier. Re-uses the same GEMINI_API_KEY we
      use for the LLM, but free-tier limits are tight for huge repos.

  EMBEDDING_MODEL contains "nomic" + NOMIC_API_KEY set  (legacy fallback)
    → Nomic API: nomic-embed-text-v1.5, 768-dim. Free quota is 10M
      tokens total — easy to exhaust across a few large repo indexes.

TASK TYPES
───────────
Every provider distinguishes document and query roles. A document
projection and a query projection live in the same embedding space
but are optimised for their direction of the inner product:
  - document:  used when indexing chunks
  - query:     used when embedding the user's question

BATCHING
─────────
All three APIs accept batched input. We use groups of 32 to stay
well under request-body size limits on large contextually-enriched
chunks (~8KB each) and to keep individual retries cheap.
"""

import time
from pathlib import Path
import sys

import requests as http

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings


_NOMIC_API_URL   = "https://api-atlas.nomic.ai/v1/embedding/text"
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_BATCH_SIZE      = 32    # conservative for all providers: stays under ~10MB body
                         # and keeps each failed batch cheap to retry
_MAX_CHARS       = 8000  # truncate each text before sending — embeddings degrade
                         # gracefully on truncation and models silently clip anyway


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

        # Provider selection is driven by the MODEL NAME, with the available
        # API key gating the choice. This lets an operator flip providers by
        # only changing EMBEDDING_MODEL in .env — no code change needed.
        name = self.model_name.lower()
        if "voyage" in name:
            if not settings.voyage_api_key:
                raise RuntimeError("EMBEDDING_MODEL is Voyage-based but VOYAGE_API_KEY is not set.")
            self._provider = "voyage"
            self._init_voyage()
        elif "gemini" in name:
            if not settings.gemini_api_key:
                raise RuntimeError("EMBEDDING_MODEL is Gemini-based but GEMINI_API_KEY is not set.")
            self._provider = "gemini"
            self._init_gemini()
        elif "nomic" in name and settings.nomic_api_key:
            self._provider = "nomic"
            self._init_nomic()
        else:
            raise RuntimeError(
                f"No embedding provider available for model '{self.model_name}'. "
                "Set GEMINI_API_KEY (default — free at https://aistudio.google.com), "
                "VOYAGE_API_KEY + EMBEDDING_MODEL=voyage-code-3, or "
                "NOMIC_API_KEY + EMBEDDING_MODEL=nomic-embed-text-v1.5."
            )

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
        self._nomic_key = settings.nomic_api_key
        print(
            f"Embedder: using Nomic API ({self.model_name}, {self.embedding_dim}-dim). "
            "No local model loaded."
        )

    def _init_gemini(self):
        """Initialise Gemini embeddings. gemini-embedding-001 supports MRL,
        so we request exactly `embedding_dim` dimensions from the API — that
        way one deployment can reuse an existing Qdrant collection schema
        (768-dim) or scale up to a larger one without code changes."""
        self._gemini_key = settings.gemini_api_key
        self._gemini_last_request_at = 0.0
        print(
            f"Embedder: using Gemini API ({self.model_name}, {self.embedding_dim}-dim). "
            "No local model loaded."
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """
        Embed a list of chunk dicts for indexing (document role).

        task_type="search_document" / input_type="document" tells the model
        these are passages being stored — it applies a different projection
        than for queries, which improves retrieval precision.

        Texts are truncated to _MAX_CHARS before sending. Embedding models have
        a token limit (~8192 tokens) and API gateways have a request body size
        limit (~10MB). Truncation degrades retrieval quality marginally but
        avoids 413 errors on large class definitions or contextually-enriched chunks.
        """
        texts = [c["text"][:_MAX_CHARS] for c in chunks]
        if self._provider == "voyage":
            return self._voyage_embed(texts, input_type="document")
        if self._provider == "gemini":
            return self._gemini_embed(texts, task_type="RETRIEVAL_DOCUMENT")
        return self._nomic_embed(texts, task_type="search_document")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query for retrieval (query role).

        The "query" projection is the counterpart to "document" — vectors
        from both live in the same space but with optimised projections.
        """
        if self._provider == "voyage":
            return self._voyage_embed([query], input_type="query")[0]
        if self._provider == "gemini":
            return self._gemini_embed([query], task_type="RETRIEVAL_QUERY")[0]
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
        """Call Nomic Atlas API with batching. Returns list of 768-dim vectors.

        Each text is truncated to _MAX_CHARS so a batch of 32 stays well under
        the 10MB request body limit even for contextually-enriched chunks.
        """
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch      = [t[:_MAX_CHARS] for t in texts[i : i + _BATCH_SIZE]]
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

    # ── Gemini API implementation ──────────────────────────────────────────────

    def _gemini_embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Call Gemini batchEmbedContents with batching. Returns list of
        `embedding_dim`-dim vectors.

        task_type is the Gemini task enum (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY).
        These produce different projections within the same embedding space —
        the document projection is optimised for being retrieved, the query
        projection for doing the retrieving.
        """
        all_embeddings: list[list[float]] = []
        batch_size = max(1, settings.gemini_embedding_batch_size)
        for i in range(0, len(texts), batch_size):
            batch      = [t[:_MAX_CHARS] for t in texts[i : i + batch_size]]
            embeddings = self._gemini_call_api(batch, task_type)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _gemini_call_api(
        self,
        texts: list[str],
        task_type: str,
        retries: int = None,
    ) -> list[list[float]]:
        """
        Single Gemini batchEmbedContents call with retry on rate limit (429)
        or service error (503). Gemini free tier is RPM-capped, so backoff is
        more aggressive than Nomic (3 retries vs 2, longer default wait).

        Response shape:
          { "embeddings": [{ "values": [float, ...] }, ...] }
        """
        url = (
            f"{_GEMINI_API_BASE}/{self.model_name}:batchEmbedContents"
            f"?key={self._gemini_key}"
        )
        model_id = f"models/{self.model_name}"
        payload = {
            "requests": [
                {
                    "model":                model_id,
                    "content":              {"parts": [{"text": t}]},
                    "taskType":             task_type,
                    "outputDimensionality": self.embedding_dim,
                }
                for t in texts
            ]
        }

        retries = settings.gemini_embedding_retries if retries is None else retries

        for attempt in range(retries + 1):
            min_interval = max(0.0, settings.gemini_embedding_min_interval)
            elapsed = time.monotonic() - self._gemini_last_request_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            response = http.post(url, json=payload, timeout=60)
            self._gemini_last_request_at = time.monotonic()

            if response.status_code in (429, 503) and attempt < retries:
                # Gemini free-tier limits can be RPM/TPM based and the API
                # doesn't always send Retry-After. Wait long enough for the
                # quota window to reset instead of failing the ingestion run.
                wait = int(response.headers.get("Retry-After", min(300, 30 * (2 ** attempt))))
                print(f"Gemini API {response.status_code}. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return [e["values"] for e in response.json()["embeddings"]]

        raise RuntimeError("Gemini API call failed after retries")

"""
retrieval.py — Hybrid search over indexed code chunks using Qdrant.

How hybrid search works in Qdrant (vs what we built manually in PDF RAG):

  PDF RAG approach (manual):
    1. Run semantic search → top-K results with cosine scores
    2. Run BM25 search    → top-K results with BM25 scores
    3. Fuse with RRF manually in Python
    4. Return merged list

  Qdrant native hybrid (this file):
    1. Send one request with TWO prefetch queries:
         - prefetch A: dense vector (semantic)
         - prefetch B: sparse vector (BM25)
    2. Qdrant runs both on the server, fuses with RRF internally
    3. Returns already-fused results

  Why native is better:
    - One network round-trip instead of three
    - Fusion happens on the server with access to full index
    - No need to maintain a separate BM25 index in Python memory

Filtering:
  Qdrant supports pre-filtering before vector search — filter by payload
  (e.g. repo="pytorch/pytorch", language="python") BEFORE scoring vectors.
  This is done at index level (using the keyword indices we created), so
  filtering a large collection to one repo is nearly free.
"""

from pathlib import Path
from typing import Optional, Literal
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    SparseVector,
    FusionQuery,
    Fusion,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings
from ingestion.embedder import Embedder
from ingestion.qdrant_store import _text_to_sparse


class RetrievalService:
    """
    Unified retrieval over Qdrant — supports semantic, keyword, and hybrid search.

    Uses the same Embedder as ingestion so queries live in the same vector space
    as the indexed chunks. Mixing embedding models breaks retrieval entirely —
    vectors from different models are incomparable.
    """

    DENSE_VECTOR_NAME  = "code"
    SPARSE_VECTOR_NAME = "bm25"

    def __init__(self):
        self.embedder = Embedder()
        self.client   = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self.collection = settings.qdrant_collection

    def search(
        self,
        query: str,
        top_k: int = None,
        repo_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        mode: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        relevance_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Search for code chunks relevant to the query.

        Args:
            query:            Natural language or code snippet
            top_k:            Number of results to return
            repo_filter:      Restrict to a specific repo ("owner/name")
            language_filter:  Restrict to a specific language ("python", "typescript")
            mode:             Search strategy
            relevance_threshold: Minimum score (0–1) for results. Queries below
                              this are considered out-of-domain and return [].

        Returns:
            List of result dicts sorted by relevance (best first):
            [{text, repo, filepath, language, chunk_type, name,
              start_line, end_line, score}, ...]
        """
        top_k = top_k or settings.top_k
        qdrant_filter = self._build_filter(repo_filter, language_filter)

        if mode == "semantic":
            results = self._semantic_search(query, top_k, qdrant_filter)
        elif mode == "keyword":
            results = self._keyword_search(query, top_k, qdrant_filter)
        else:
            results = self._hybrid_search(query, top_k, qdrant_filter)

        # Relevance gate — skip when repo_filter is set (user explicitly chose a repo)
        if relevance_threshold > 0 and not repo_filter and results:
            if results[0]["score"] < relevance_threshold:
                return []

        return results

    def _build_filter(
        self,
        repo: Optional[str],
        language: Optional[str],
    ) -> Optional[Filter]:
        """Build a Qdrant filter from optional repo and language constraints."""
        conditions = []
        if repo:
            conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo)))
        if language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        return Filter(must=conditions) if conditions else None

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        qdrant_filter: Optional[Filter],
    ) -> list[dict]:
        """
        Pure dense vector search (cosine similarity).
        Good for: conceptual questions ("how does attention work?")
        """
        query_vector = self.embedder.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=self.DENSE_VECTOR_NAME,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._format_result(r) for r in results.points]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        qdrant_filter: Optional[Filter],
    ) -> list[dict]:
        """
        Pure sparse vector search (BM25 keyword matching).
        Good for: exact identifiers ("reciprocal_rank_fusion", "embed_batch")
        """
        sparse_vector = _text_to_sparse(query)

        results = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using=self.SPARSE_VECTOR_NAME,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._format_result(r) for r in results.points]

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        qdrant_filter: Optional[Filter],
    ) -> list[dict]:
        """
        Hybrid search: dense + sparse fused with RRF on Qdrant's server.

        Prefetch fetches top_k*2 from each system before fusion —
        a result that ranked 6th semantically and 6th by keyword would
        rank 1st by RRF, but would be missed if we only prefetched top_k.
        """
        query_vector  = self.embedder.embed_query(query)
        sparse_vector = _text_to_sparse(query)

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using=self.DENSE_VECTOR_NAME,
                    filter=qdrant_filter,
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=sparse_vector,
                    using=self.SPARSE_VECTOR_NAME,
                    filter=qdrant_filter,
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [self._format_result(r) for r in results.points]

    def _format_result(self, point) -> dict:
        """Convert a Qdrant ScoredPoint into a clean result dict."""
        p = point.payload or {}
        return {
            "text":       p.get("text", ""),
            "repo":       p.get("repo", ""),
            "filepath":   p.get("filepath", ""),
            "language":   p.get("language", ""),
            "chunk_type": p.get("chunk_type", ""),
            "name":       p.get("name", ""),
            "start_line": p.get("start_line", 0),
            "end_line":   p.get("end_line", 0),
            "score":      round(point.score, 4) if point.score else 0.0,
        }

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into an LLM-ready context string.

        Each chunk is numbered and labelled with its file path and line range
        so the LLM can produce traceable citations like:
          "According to Source 3 (src/auth/middleware.py, lines 45–72)..."
        """
        if not results:
            return "No relevant code found in the indexed repositories."

        parts = []
        for i, r in enumerate(results, 1):
            citation = f"{r['filepath']}"
            if r.get("name"):
                citation += f" — {r['name']}()"
            citation += f" | lines {r['start_line']}–{r['end_line']}"

            parts.append(f"[Source {i} | {r['repo']} | {citation}]\n{r['text']}")

        return "\n\n" + "─" * 40 + "\n\n".join(parts)

"""
retrieval.py — Hybrid search over indexed code chunks, with a full quality stack.

RETRIEVAL PIPELINE (what runs per query)
──────────────────────────────────────────
1. HyDE  (optional, USE_HYDE=true)
   Generate a hypothetical code snippet that would answer the query.
   Embed THAT snippet instead of the raw question.
   WHY: Natural language questions ("how does backprop work?") and actual code
   ("def backward(self):") live in different parts of the embedding space.
   Hypothetical code bridges this gap — it looks like code, talks about
   the same concept, and retrieves actual code much better.

2. Query Expansion  (optional, EXPAND_QUERIES=true)
   Ask the LLM to generate 2-3 rephrased versions of the original query.
   Run retrieval on the original + all variants, then merge with RRF.
   WHY: A single query has synonyms the user didn't think of ("forward pass"
   vs "inference", "weight update" vs "gradient step"). Expanding covers
   all of them at the cost of one extra LLM call.

3. Hybrid search  (Qdrant native — always on)
   Send ONE request to Qdrant with two prefetch queries:
     a) Dense vector (cosine similarity, captures semantics)
     b) Sparse vector BM25 (token matching, captures exact identifiers)
   Qdrant fuses the two lists on the server using Reciprocal Rank Fusion.
   WHY: Semantic search finds conceptually related code but misses exact
   function names. BM25 finds exact names but misses conceptual matches.
   Hybrid gets both.

4. Re-ranking  (always on)
   A cross-encoder reads (query, chunk) pairs together in one forward pass.
   Unlike cosine similarity (query and document embedded separately), the
   cross-encoder can see interactions: "this query asks about backward()
   and this chunk calls loss.backward(3 times) — high relevance."
   Cohere rerank-v3.5 is used if COHERE_API_KEY is set; falls back to
   local ms-marco cross-encoder (80MB, CPU-based).

QDRANT HYBRID SEARCH (vs what we built manually earlier)
──────────────────────────────────────────────────────────
  Manual approach (old):
    1. Run semantic search → top-K results
    2. Run BM25 search    → top-K results
    3. Fuse with RRF in Python
    4. Return merged list

  Qdrant native hybrid (this file):
    1. Send ONE request with TWO prefetch queries
    2. Qdrant runs both on the server, fuses with RRF internally
    3. Returns already-fused results
  Benefits: one network round-trip, fusion at full index scale.
"""

import json
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
from ingestion.qdrant_store import QdrantStore, _text_to_sparse


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rrf_merge(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion: merge multiple ranked lists into a single list.

    RRF formula: score(d) = Σ 1/(k + rank(d, list_i)) for each list i.

    WHY k=60?
      k is a smoothing constant. With k=60, the difference between rank 1
      and rank 2 is smaller (1/61 vs 1/62) than between rank 1 and rank 60
      (1/61 vs 1/120). This prevents any single list from dominating.
      k=60 is the standard value from the original RRF paper (Cormack 2009).

    RRF is parameter-free (no weights to tune) and robust to mixing scores
    from different systems (BM25 scores and cosine scores are not comparable
    — but ranks are). That's why Qdrant uses it internally for hybrid search.
    """
    scores:     dict[str, float] = {}
    all_chunks: dict[str, dict]  = {}

    for ranked_list in lists:
        for rank, result in enumerate(ranked_list):
            key = (
                f"{result.get('repo', '')}::"
                f"{result.get('filepath', '')}::"
                f"{result.get('start_line', 0)}"
            )
            scores[key]     = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            all_chunks[key] = result

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [{**all_chunks[k], "score": round(scores[k], 4)} for k in sorted_keys]


# ── Reranker ───────────────────────────────────────────────────────────────────

class Reranker:
    """
    Two-stage re-ranker: Cohere API primary, local cross-encoder fallback.

    WHY RE-RANKING EXISTS
    ─────────────────────
    Hybrid search (BM25 + semantic) is good at RECALL — it casts a wide net
    and finds most relevant chunks somewhere in the top-20. But its PRECISION
    is limited: it ranks by cosine similarity and BM25 scores independently.

    A cross-encoder reads query AND document together in one forward pass.
    It can see: "the query asks about backpropagation, and this chunk calls
    loss.backward() — that's a direct match." Cosine similarity can't do this
    because it embeds query and document separately.

    Result: retrieval fetches top-K×3 (high recall), re-ranker picks top-K
    (high precision). This two-stage pattern is standard in production RAG.

    COHERE vs LOCAL
    ────────────────
    Cohere rerank-v3.5:
      - API-based — no local model, just an HTTP call
      - Trained on multilingual, multi-domain data including code
      - 1000 calls/month free at https://cohere.com
      - ~300ms latency per call

    Local ms-marco-MiniLM-L-6-v2:
      - 80MB, runs on CPU in <200ms for 20 chunks
      - Trained on MS MARCO (120M relevance pairs)
      - No API calls, works offline
      - Loaded lazily on first use
    """

    LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self._local_model = None   # loaded lazily on first fallback use
        self._cohere      = None   # initialised on first cohere call

    def provider(self) -> str:
        """Return which reranker will be used: 'cohere' or 'local'."""
        return "cohere" if settings.cohere_api_key else "local"

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Score each candidate against the query and return the top_k highest."""
        if not candidates:
            return candidates

        if settings.cohere_api_key:
            return self._cohere_rerank(query, candidates, top_k)
        return self._local_rerank(query, candidates, top_k)

    def _cohere_rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Cohere Rerank API: sends all (query, doc) pairs in one call.

        Returns top_k results sorted by relevance_score (0–1, higher = better).
        Falls back to local cross-encoder if the API call fails.
        """
        if self._cohere is None:
            try:
                import cohere
                self._cohere = cohere.ClientV2(api_key=settings.cohere_api_key)
            except ImportError:
                print("cohere package not installed, falling back to local reranker")
                return self._local_rerank(query, candidates, top_k)

        # Truncate texts to avoid exceeding Cohere's per-doc token limit (~4096)
        docs = [c["text"][:1500] for c in candidates]

        try:
            resp = self._cohere.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs,
                top_n=top_k,
            )
            reranked = []
            for r in resp.results:
                c = dict(candidates[r.index])
                c["score"] = round(r.relevance_score, 4)
                reranked.append(c)
            return reranked
        except Exception as e:
            print(f"Cohere rerank failed (falling back to local): {e}")
            return self._local_rerank(query, candidates, top_k)

    def _local_rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Local cross-encoder re-ranking using ms-marco-MiniLM-L-6-v2.

        Scores are raw logits (unbounded floats). We apply sigmoid to map them
        to (0, 1) before returning, so the frontend can display them as percentages.
        Sigmoid preserves rank order: if logit(A) > logit(B) then sigmoid(A) > sigmoid(B).
        """
        import math

        def _sigmoid(x: float) -> float:
            # Clamp to avoid overflow in exp for very large negative values.
            return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

        if self._local_model is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            print(f"Reranker: loading {self.LOCAL_MODEL}...")
            self._local_model = CrossEncoder(self.LOCAL_MODEL)
            print("Reranker: ready")

        pairs  = [(query, c["text"]) for c in candidates]
        scores = self._local_model.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{**c, "score": round(_sigmoid(float(s)), 4)} for s, c in ranked[:top_k]]


# ── RetrievalService ───────────────────────────────────────────────────────────

class RetrievalService:
    """
    Full retrieval pipeline: HyDE → query expansion → hybrid search → re-rank.

    Uses the same Embedder as ingestion so queries live in the same vector
    space as the indexed chunks. Mixing embedding models breaks retrieval
    entirely — vectors from different models are incomparable.

    Why accept embedder, store, reranker, and gen as arguments?
      - Embedder: IngestionService and RetrievalService both need it.
        Instantiating twice wastes RAM. main.py creates one and shares it.
      - QdrantStore: one connection pool, one auth handshake.
      - Reranker: model loads lazily but we want one instance.
      - gen: GenerationService, needed for HyDE and query expansion.
        Passing it avoids a circular import and lets callers control
        whether LLM-based quality features are active.
    """

    DENSE_VECTOR_NAME  = "code"
    SPARSE_VECTOR_NAME = "bm25"

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
        reranker: Reranker | None = None,
        gen=None,   # GenerationService — typed as Any to avoid circular import
    ):
        self.embedder  = embedder or Embedder()
        self.reranker  = reranker or Reranker()
        self._gen      = gen   # None if no LLM configured; HyDE/expansion silently skipped

        if store is not None:
            self.client     = store.client
            self.collection = store.collection
        else:
            self.client     = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            self.collection = settings.qdrant_collection

    # ── Public search interface ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = None,
        repo_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        mode: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        relevance_threshold: float = 0.0,
        return_pipeline: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """
        Full retrieval pipeline: optional HyDE + query expansion → search → rerank.

        Args:
            query:               Natural language or code snippet
            top_k:               Number of results to return
            repo_filter:         Restrict to a specific repo ("owner/name")
            language_filter:     Restrict to a language ("python", "typescript")
            mode:                Search strategy (semantic | keyword | hybrid)
            relevance_threshold: Minimum score to include a result (0 = off)
            return_pipeline:     If True, return (results, pipeline_info) instead
                                 of just results. Used by the streaming endpoint
                                 to include quality feature metadata in the meta event.

        Returns:
            If return_pipeline=False (default): list of result dicts
            If return_pipeline=True: (results, pipeline_info) where
              pipeline_info = {"hyde": bool, "expanded": int, "reranker": str}
        """
        top_k        = top_k or settings.top_k
        qdrant_filter = self._build_filter(repo_filter, language_filter)
        pipeline      = {"hyde": False, "expanded": 0, "reranker": self.reranker.provider()}

        # ── Stage 1a: HyDE — replace query with a hypothetical code answer ─────
        # The hypothetical answer lives in code-embedding space, not question-
        # embedding space, so it retrieves actual code much better.
        # Only runs if gen is available and USE_HYDE is enabled.
        search_query = query
        if self._gen and settings.use_hyde:
            hyde_text = self._hyde_expand(query)
            if hyde_text and hyde_text != query:
                search_query  = hyde_text
                pipeline["hyde"] = True

        # ── Stage 1b: Query expansion — search multiple phrasings ─────────────
        # Generate 2-3 variants, search each, merge all results with RRF.
        # We fetch fewer candidates per variant (candidate_k // 3) so the
        # total pool stays manageable.
        candidate_k = top_k * 3
        all_result_lists: list[list[dict]] = []

        if mode == "semantic":
            base_results = self._semantic_search(search_query, candidate_k, qdrant_filter)
        elif mode == "keyword":
            base_results = self._keyword_search(query, candidate_k, qdrant_filter)  # no HyDE for keyword
        else:
            base_results = self._hybrid_search(search_query, candidate_k, qdrant_filter)

        all_result_lists.append(base_results)

        # Skip expansion when HyDE already ran — both are LLM calls and running
        # both on the same query burns 2 provider tokens before the main answer.
        # HyDE is the stronger quality signal; expansion adds diminishing returns.
        if self._gen and settings.expand_queries and mode != "keyword" and not pipeline["hyde"]:
            variants = self._expand_query(query)
            for variant in variants:
                var_results = self._hybrid_search(variant, candidate_k // 2, qdrant_filter)
                all_result_lists.append(var_results)
            if variants:
                pipeline["expanded"] = len(variants)

        # Merge all result lists with RRF if we have more than one
        if len(all_result_lists) > 1:
            candidates = _rrf_merge(all_result_lists)[:candidate_k]
        else:
            candidates = base_results

        # Per-result relevance gate — filter before re-ranking
        if relevance_threshold > 0 and not repo_filter:
            candidates = [r for r in candidates if r["score"] >= relevance_threshold]

        # ── Stage 2: Re-rank with cross-encoder ───────────────────────────────
        if len(candidates) > top_k:
            results = self.reranker.rerank(query, candidates, top_k)
        else:
            results = candidates[:top_k]

        if return_pipeline:
            return results, pipeline
        return results

    # ── LLM-powered query enhancement ─────────────────────────────────────────

    def _hyde_expand(self, query: str) -> str:
        """
        HyDE: generate a hypothetical code snippet that would answer the query.

        Hypothetical Document Embeddings (Gao et al. 2022):
          Instead of embedding "how does backprop work?", embed the answer
          to that question — a realistic code snippet showing backprop.
          The snippet lives in the same embedding space as actual code,
          so retrieval finds the actual implementation much more reliably.

        The LLM is instructed to write SHORT, realistic code without
        explanation — we want something that embeds like real code.
        If the LLM fails (rate limit, etc.), we fall back to the original query.
        """
        system = (
            "You are a code search assistant. Given a question about code, write a short "
            "realistic Python code snippet (10-25 lines) that would directly answer or demonstrate "
            "the concept. Include brief inline comments. Output ONLY the code snippet, no prose."
        )
        prompt = f"Question: {query}\n\nCode snippet:"
        try:
            return self._gen.generate(system, prompt, temperature=0.1)
        except Exception as e:
            print(f"HyDE expansion failed (using original query): {e}")
            return query

    def _expand_query(self, query: str) -> list[str]:
        """
        Query Expansion: generate 2-3 alternative search queries for the same intent.

        WHY THIS HELPS
        ──────────────
        A developer might ask "how does the optimizer step work?" but the
        relevant code uses the identifiers "Adam", "weight_update", "grad.zero_".
        Expanding the query to ["Adam optimizer step implementation",
        "weight_update gradient descent", "zero_grad backward step"] retrieves
        all three patterns with one extra LLM call.

        We use JSON mode to get a clean list with no parsing ambiguity.
        Failed expansion silently returns empty list — the original query still runs.
        """
        system = (
            "You are a code search query optimizer. Generate 2-3 alternative search queries "
            "that capture different angles, synonyms, or implementation details of the question. "
            "Return ONLY a JSON array of strings."
        )
        prompt = f"Original query: {query}\n\nAlternative queries (JSON array):"
        try:
            raw = self._gen.generate(system, prompt, temperature=0.1, json_mode=True)
            variants = json.loads(raw)
            if isinstance(variants, list):
                return [str(v) for v in variants[:3] if v != query]
        except Exception as e:
            print(f"Query expansion failed (using only original query): {e}")
        return []

    # ── Qdrant search methods ──────────────────────────────────────────────────

    def _build_filter(self, repo: Optional[str], language: Optional[str]) -> Optional[Filter]:
        conditions = []
        if repo:
            conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo)))
        if language:
            conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        return Filter(must=conditions) if conditions else None

    def _semantic_search(self, query: str, top_k: int, qdrant_filter: Optional[Filter]) -> list[dict]:
        """Pure dense vector search (cosine similarity). Best for: conceptual questions."""
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

    def _keyword_search(self, query: str, top_k: int, qdrant_filter: Optional[Filter]) -> list[dict]:
        """Pure sparse vector search (BM25). Best for: exact identifiers and function names."""
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

    def _hybrid_search(self, query: str, top_k: int, qdrant_filter: Optional[Filter]) -> list[dict]:
        """
        Hybrid: dense + sparse fused with RRF on Qdrant's server.

        Prefetch fetches top_k×2 from each system before fusion — a result
        that ranked 6th semantically AND 6th by keyword would rank 1st by
        RRF but would be missed if we only prefetched top_k.
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

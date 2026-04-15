"""
qdrant_store.py — Store and retrieve code chunks in Qdrant Cloud.

What is Qdrant?
  Qdrant is a vector database — it stores vectors (embeddings) alongside
  metadata (payload) and lets you search by vector similarity.

  Unlike ChromaDB (local file on disk), Qdrant Cloud is a hosted service:
    - Your vectors live in the cloud, not on your laptop
    - The app can be deployed anywhere without worrying about disk persistence
    - Free tier: 1GB storage, 1 collection — enough for ~50 repos

Why Qdrant specifically?
  Two features make it ideal for this project:

  1. Native hybrid search: Qdrant supports both dense vectors (semantic
     similarity) and sparse vectors (BM25 keyword matching) in one query.
     No need to maintain a separate BM25 index or implement RRF manually.

  2. Rich filtering: filter by metadata before scoring (e.g. "only search
     Python files in the auth/ directory"). This is done at the index level,
     not by post-filtering results — much more efficient.

Collection setup:
  A Qdrant "collection" is like a database table.
  Ours has:
    - Dense vectors: 768-dim, cosine distance (nomic-embed-code output)
    - Sparse vectors: BM25 (Qdrant computes this from text automatically)
    - Payload (metadata per point): repo, filepath, language, function_name,
      class_name, chunk_type, start_line, end_line, text

Point = one chunk. Each point has:
  - id: MD5 hash of "repo::filepath::start_line" (stable across re-ingestion)
  - vector: 768-dim float list
  - payload: all metadata + the chunk text itself
"""

import hashlib
from pathlib import Path
from typing import Optional
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings


class QdrantStore:
    """
    Manages the Qdrant collection and provides upsert + search operations.

    The client connects to Qdrant Cloud on init. The collection is created
    if it doesn't already exist (idempotent).
    """

    DENSE_VECTOR_NAME  = "code"    # name for the dense vector field
    SPARSE_VECTOR_NAME = "bm25"    # name for the sparse vector field

    def __init__(self):
        if not settings.qdrant_url:
            raise ValueError(
                "QDRANT_URL not set. Get a free cluster at https://cloud.qdrant.io\n"
                "Then add QDRANT_URL and QDRANT_API_KEY to your .env file."
            )

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=60,  # default is 5s — too short for batch upserts over the network
        )
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Create the collection if it doesn't exist.

        Qdrant collections must declare their vector configuration upfront.
        We configure:
          - Dense vectors: cosine distance, 768 dims (nomic-embed-code)
          - Sparse vectors: BM25 (no dimension needed — sparse by definition)

        Why cosine distance?
          Cosine similarity measures the angle between vectors, ignoring
          magnitude. This is ideal for embeddings that have been L2-normalized
          (which our embedder does). Two vectors pointing in the same direction
          have cosine similarity 1.0 regardless of their length.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            print(f"Creating collection '{self.collection}'...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    self.DENSE_VECTOR_NAME: VectorParams(
                        size=settings.embedding_dim,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
        else:
            print(f"Collection '{self.collection}' already exists")

        # Ensure payload indices exist — called even for existing collections so
        # newly added fields (like text_hash) get indexed after a code update.
        # create_payload_index is idempotent for most Qdrant versions; we wrap
        # in try/except for safety in case a specific version raises on duplicate.
        self._ensure_payload_indices()

    def _ensure_payload_indices(self):
        """Create payload indices for all filterable fields (idempotent)."""
        for field in ["repo", "filepath", "language", "chunk_type", "name", "text_hash", "calls", "imports", "base_classes"]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # index already exists — safe to ignore
        print(f"  → Payload indices ensured")

    def get_contextual_at(self, repo: str) -> str | None:
        """
        Return the contextual_at timestamp for a repo, or None if not run.

        During contextual re-ingestion, we stamp every chunk with the ISO
        timestamp of when contextual retrieval ran. Reading it back here
        avoids keeping this state in an in-memory dict that resets on restart.

        We just need the first chunk that has contextual_at set — they all
        have the same timestamp for a given re-ingest.
        """
        filt = Filter(must=[
            FieldCondition(key="repo", match=MatchValue(value=repo)),
        ])
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=filt,
            limit=50,   # check a batch — not all chunks have contextual_at
            offset=None,
            with_payload=["contextual_at"],
            with_vectors=False,
        )
        for p in results:
            ts = (p.payload or {}).get("contextual_at")
            if ts:
                return ts
        return None

    def upsert_chunks(self, chunks: list[dict], dense_vectors: list[list[float]]) -> list[str]:
        """
        Store chunks and their embeddings in Qdrant. Returns the list of point IDs written.

        Upsert = insert or update. If a point with the same ID already exists,
        it's overwritten. This means re-ingesting a repo is safe — no duplicates.

        The ID is a hash of "repo::filepath::start_line". This is stable:
        the same chunk from the same file always gets the same ID, regardless
        of when it was ingested.

        Sparse vectors (BM25) are computed from the text on Qdrant's side.
        We pass the tokens (words) and their term frequencies.
        Qdrant uses these to build the sparse vector internally.

        We return the written IDs so the caller can delete stale points
        (chunks that existed before re-index but no longer appear in the source).
        """
        if len(chunks) != len(dense_vectors):
            raise ValueError(f"Chunks ({len(chunks)}) and vectors ({len(dense_vectors)}) must match")

        points = []
        written_ids: list[str] = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            point_id = _stable_id(chunk)
            written_ids.append(point_id)

            # Build sparse vector (BM25) from the chunk text
            sparse_vec = _text_to_sparse(chunk["text"])

            points.append(PointStruct(
                id=point_id,
                vector={
                    self.DENSE_VECTOR_NAME: dense_vec,
                    self.SPARSE_VECTOR_NAME: sparse_vec,
                },
                payload={
                    "text":        chunk["text"],
                    "repo":        chunk.get("repo", ""),
                    "filepath":    chunk.get("filepath", ""),
                    "language":    chunk.get("language", ""),
                    "chunk_type":  chunk.get("chunk_type", "text"),
                    "name":        chunk.get("name", ""),
                    "start_line":  chunk.get("start_line", 0),
                    "end_line":    chunk.get("end_line", 0),
                    # Call graph: names of functions/methods called by this chunk.
                    # Extracted by _CallExtractor during AST chunking.
                    # Empty list for non-Python or window chunks.
                    "calls":        chunk.get("calls", []),
                    # Import graph: module names imported at the top of the file.
                    # Populated only for module-level chunks; empty for functions/classes.
                    # Used to build file-level dependency edges in the Architecture diagram.
                    "imports":      chunk.get("imports", []),
                    # Inheritance graph: base class names from "class Foo(Bar):".
                    # Populated only for class chunks; empty for functions/modules.
                    # Used to build real inheritance edges in the Class Hierarchy diagram.
                    "base_classes": chunk.get("base_classes", []),
                    # SHA-256 of the chunk text — used for cross-repo deduplication.
                    # If two repos contain identical code, the second ingestion can
                    # reuse the existing embedding vector instead of re-calling the model.
                    "text_hash":   chunk.get("text_hash", ""),
                    # ISO timestamp of when contextual retrieval ran for this chunk.
                    # Persisted in Qdrant so it survives server restarts — the /repos
                    # endpoint reads this instead of an in-memory dict.
                    # None for chunks that haven't been contextualised.
                    "contextual_at": chunk.get("contextual_at"),
                },
            ))

        # Smaller batches reduce per-request payload and avoid cloud write timeouts
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)

        print(f"  Upserted {len(points)} points to Qdrant")
        return written_ids

    def count(self, repo: Optional[str] = None) -> int:
        """Return total number of stored chunks, optionally filtered by repo."""
        count_filter = None
        if repo:
            count_filter = Filter(must=[
                FieldCondition(key="repo", match=MatchValue(value=repo))
            ])
        result = self.client.count(
            collection_name=self.collection,
            count_filter=count_filter,
            exact=True,
        )
        return result.count

    def list_repos(self) -> list[str]:
        """
        Return all unique repo names stored in the collection.
        Used by the UI to populate the repo selector.
        """
        # Scroll through all points and collect unique repo values
        repos = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=offset,
                with_payload=["repo"],
                with_vectors=False,
            )
            for point in results:
                repos.add(point.payload.get("repo", ""))
            if offset is None:
                break
        return sorted(repos - {""})

    def scroll_repo(self, repo: str, with_payload: list[str] | None = None) -> list[dict]:
        """
        Fetch all points for a repo as plain dicts.

        Used by the graph service to build the call graph without going through
        the retrieval/embedding pipeline. We scroll (paginate) through all
        points because Qdrant limits a single query to 10,000 results.

        Args:
            repo:         "owner/name" slug
            with_payload: list of payload field names to include (None = all)
        """
        results = []
        offset = None
        filt = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))])
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filt,
                limit=1000,
                offset=offset,
                with_payload=with_payload or True,
                with_vectors=False,
            )
            for p in points:
                results.append(p.payload)
            if offset is None:
                break
        return results

    def find_callers(self, function_name: str, repo: Optional[str] = None) -> list[dict]:
        """
        Find all chunks that call a specific function by searching the 'calls' payload.

        During AST chunking, _CallExtractor records every function/method call made
        within each chunk and stores the list in the 'calls' payload field.
        This lets us do an exact structural lookup instead of fuzzy text search —
        "find all functions that call backward()" is a filter, not a search.

        Args:
            function_name: Exact function name to look for in callers
            repo:          Optional 'owner/name' to restrict scope

        Returns:
            List of payload dicts for chunks that contain a call to function_name
        """
        conditions = [
            FieldCondition(key="calls", match=MatchValue(value=function_name))
        ]
        if repo:
            conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo)))

        filt = Filter(must=conditions)
        results = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filt,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                results.append(p.payload)
            if offset is None:
                break
        return results

    def find_symbol(self, symbol_name: str, repo: Optional[str] = None) -> list[dict]:
        """
        Find chunks whose 'name' field exactly matches a symbol (function or class).

        Unlike search_code (which embeds the query and does nearest-neighbour search),
        this is a Qdrant filter query — no vectors involved at all.
        It's like a WHERE name = 'foo' in SQL.

        Why is this different from search_code("foo")?
        - search_code embeds "foo" and returns semantically similar code.
          If the function is named 'foo' but the query embeds to a different region
          of the vector space, it might not be top-ranked.
        - find_symbol("foo") is an exact key lookup — always returns the definition
          of 'foo' if it was indexed, regardless of vector proximity.

        Use case: the agent knows the exact name from a previous search result
        and wants to jump straight to the definition.

        Args:
            symbol_name: Exact function or class name as it appears in source
            repo:        Optional 'owner/name' to restrict scope
        """
        conditions = [FieldCondition(key="name", match=MatchValue(value=symbol_name))]
        if repo:
            conditions.append(FieldCondition(key="repo", match=MatchValue(value=repo)))

        filt = Filter(must=conditions)
        results = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filt,
                limit=20,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                results.append(p.payload)
            if offset is None:
                break
        return results

    def find_vectors_by_hash(self, hashes: list[str]) -> dict[str, list[float]]:
        """
        Look up existing dense embedding vectors by text hash.

        Used by the ingestion pipeline to avoid re-embedding identical code
        that already exists in the index (possibly from a different repo).

        Example: if micrograd/engine.py and nanoGPT/utils.py both contain
        "def relu(x): return max(0, x)", the second ingestion finds the hash
        already in Qdrant and reuses the vector — no embedding call needed.

        Args:
            hashes: list of SHA-256 hex strings (from chunk["text_hash"])

        Returns:
            {hash: dense_vector_list} for every hash that already exists.
            Hashes with no existing vector are simply absent from the result.
        """
        if not hashes:
            return {}

        from qdrant_client.models import MatchAny

        found: dict[str, list[float]] = {}
        offset = None
        filt = Filter(must=[FieldCondition(key="text_hash", match=MatchAny(any=hashes))])

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filt,
                limit=500,
                offset=offset,
                with_payload=["text_hash"],
                # Only fetch the dense vector — sparse is not needed for reuse
                with_vectors=[self.DENSE_VECTOR_NAME],
            )
            for p in points:
                h = (p.payload or {}).get("text_hash", "")
                if not h or h in found:
                    continue
                # p.vector is a dict when the collection has named vectors
                vec = p.vector
                if isinstance(vec, dict):
                    vec = vec.get(self.DENSE_VECTOR_NAME)
                if vec is not None:
                    found[h] = list(vec)
            if offset is None:
                break

        return found

    def delete_stale_chunks(self, repo: str, keep_ids: set[str]) -> int:
        """
        Delete chunks for a repo whose IDs are NOT in keep_ids.

        Called after a force re-index completes. Because we now upsert first and
        delete stale chunks last (blue-green strategy), the repo stays visible with
        old (stale) data throughout the entire re-index, and only disappears briefly
        during this final cleanup step — not for the 5-10 minutes of contextual retrieval.

        Why not just delete everything upfront?
          With free-tier LLM APIs, contextual enrichment can take 5-10 minutes.
          If we delete old chunks first, the repo has 0 chunks for that entire window —
          it disappears from the sidebar and any queries during that time return nothing.
          Upserting first means old chunks remain queryable until the new ones are ready.

        Returns the number of stale points deleted.
        """
        from qdrant_client.models import PointIdsList

        # Scroll to collect all current IDs for this repo
        old_ids: list[str] = []
        offset = None
        filt = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))])
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filt,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for p in points:
                old_ids.append(str(p.id))
            if offset is None:
                break

        # Normalize both sides before comparing.
        # _stable_id() returns a 32-char MD5 hex string (no dashes).
        # Qdrant stores UUIDs internally and returns them with dashes when scrolled
        # (e.g. "a1b2c3d4-e5f6-a7b8-c9d0-e1f2a3b4c5d6"). Strip dashes on both
        # sides so the comparison works regardless of formatting.
        normalized_keep = {k.replace("-", "") for k in keep_ids}
        stale = [id_ for id_ in old_ids if id_.replace("-", "") not in normalized_keep]
        if not stale:
            return 0

        # Delete in batches — Qdrant recommends ≤500 IDs per request
        batch_size = 500
        for i in range(0, len(stale), batch_size):
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=stale[i : i + batch_size]),
            )

        print(f"  Deleted {len(stale)} stale points for {repo}")
        return len(stale)

    def delete_repo(self, repo: str) -> int:
        """Delete all chunks for a repo. Returns number of points deleted."""
        before = self.count(repo=repo)
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="repo", match=MatchValue(value=repo))
            ]),
        )
        return before

    # ── Tour concept-name feedback store ─────────────────────────────────────
    # When the evaluator corrects an artifact concept name (e.g. "health" → "Health
    # Check Bypass"), that correction is stored here so future tour builds can
    # avoid the same artifact name. Uses the same sidecar pattern as agent notes.
    #
    # Storage: one point per repo, ID = hash(repo + ":feedback"), payload stores
    # the full {bad_name: good_name} dict as a JSON string. One point per repo
    # makes load/save O(1) and keeps the collection tiny.

    @property
    def _feedback_collection(self) -> str:
        return f"{self.collection}_feedback"

    def _ensure_feedback_collection(self) -> None:
        """Create the feedback sidecar collection if it doesn't exist (idempotent)."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self._feedback_collection not in existing:
            self.client.create_collection(
                collection_name=self._feedback_collection,
                vectors_config=VectorParams(size=1, distance=Distance.DOT),
            )
            try:
                self.client.create_payload_index(
                    collection_name=self._feedback_collection,
                    field_name="repo",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def save_tour_feedback(self, repo: str, corrections: dict) -> None:
        """
        Persist bad→good concept name corrections for a repo.

        Merges new corrections into any previously stored ones — the feedback
        store only grows, never shrinks, so the model's avoid-list gets richer.
        """
        self._ensure_feedback_collection()
        existing = self.load_tour_feedback(repo)
        existing.update(corrections)
        import json as _json
        feedback_id = hashlib.md5(f"{repo}::feedback".encode()).hexdigest()
        self.client.upsert(
            collection_name=self._feedback_collection,
            points=[PointStruct(
                id=feedback_id,
                vector=[0.0],   # dummy — retrieved by filter, not similarity
                payload={
                    "repo":     repo,
                    "feedback": _json.dumps(existing),
                },
            )],
        )

    def load_tour_feedback(self, repo: str) -> dict:
        """
        Load persisted concept-name corrections for a repo.
        Returns {bad_name: good_name} dict, or {} if no feedback stored yet.
        """
        self._ensure_feedback_collection()
        import json as _json
        results, _ = self.client.scroll(
            collection_name=self._feedback_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="repo", match=MatchValue(value=repo))
            ]),
            limit=1,
            with_payload=["feedback"],
            with_vectors=False,
        )
        for p in results:
            raw = (p.payload or {}).get("feedback", "")
            if raw:
                try:
                    return _json.loads(raw)
                except Exception:
                    return {}
        return {}

    # ── Persistent agent notes ────────────────────────────────────────────────
    # Agent notes are stored in a sidecar collection ("<collection>_notes").
    # Each note is a point whose ID is a deterministic hash of (repo, key),
    # so upserting the same key simply overwrites the previous value.
    #
    # Why a separate collection?
    #   Notes don't need vectors — they're looked up by exact (repo, key) match,
    #   not by similarity. A sidecar collection keeps them out of the code-search
    #   index so they can never pollute retrieval results.
    #   We store a dummy 1-dim zero vector to satisfy Qdrant's schema requirement.

    @property
    def _notes_collection(self) -> str:
        return f"{self.collection}_notes"

    def _ensure_notes_collection(self) -> None:
        """Create the notes sidecar collection if it doesn't exist (idempotent)."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self._notes_collection not in existing:
            self.client.create_collection(
                collection_name=self._notes_collection,
                vectors_config=VectorParams(size=1, distance=Distance.DOT),
            )
            # Index repo + key for efficient filtering
            for field in ["repo", "key"]:
                try:
                    self.client.create_payload_index(
                        collection_name=self._notes_collection,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

    def save_note(self, repo: str, key: str, value: str) -> None:
        """
        Persist an agent note for a repo (upsert by key).

        The point ID is a deterministic hash of (repo, key) so re-saving
        the same key overwrites the previous value without creating duplicates.
        """
        self._ensure_notes_collection()
        note_id = hashlib.md5(f"{repo}::{key}".encode()).hexdigest()
        import datetime
        self.client.upsert(
            collection_name=self._notes_collection,
            points=[PointStruct(
                id=note_id,
                vector=[0.0],  # dummy — notes are retrieved by filter, not similarity
                payload={
                    "repo":       repo,
                    "key":        key,
                    "value":      value,
                    "updated_at": datetime.datetime.utcnow().isoformat(),
                },
            )],
        )

    def load_notes(self, repo: str) -> dict[str, str]:
        """
        Load all persisted notes for a repo. Returns {key: value} dict.

        Called at the start of each agent run to pre-populate the in-memory
        working memory, so the agent can recall facts from previous sessions.
        """
        self._ensure_notes_collection()
        results, _ = self.client.scroll(
            collection_name=self._notes_collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="repo", match=MatchValue(value=repo))
            ]),
            limit=200,
            with_payload=["key", "value"],
            with_vectors=False,
        )
        return {
            p.payload["key"]: p.payload["value"]
            for p in results
            if p.payload and "key" in p.payload and "value" in p.payload
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stable_id(chunk: dict) -> str:
    """
    Generate a stable, deterministic ID for a chunk.

    MD5 of "repo::filepath::start_line" → same chunk always gets same ID.
    Qdrant requires point IDs to be UUID strings or unsigned integers.
    We use the hex digest of MD5 as a UUID-like string.
    """
    key = f"{chunk.get('repo', '')}::{chunk.get('filepath', '')}::{chunk.get('start_line', 0)}"
    return hashlib.md5(key.encode()).hexdigest()


def _text_to_sparse(text: str) -> SparseVector:
    """
    Build a sparse BM25 vector from text.

    Sparse vectors represent term frequencies: which "dimensions" (token IDs)
    are present and how often. Unlike dense vectors (768 floats where all
    positions have values), sparse vectors only store non-zero positions.

    Example:
      text = "def embed_text(self, text):"
      tokens = {"def": 1, "embed_text": 1, "self": 1, "text": 2}
      → indices = [md5("def") % 1M, md5("embed_text") % 1M, ...]
        values  = [1.0, 1.0, 1.0, 2.0, ...]

    IMPORTANT — this is TF (term frequency) only, NOT full BM25.
    True BM25 requires IDF (inverse document frequency), which weights rare
    terms higher than common ones (e.g. "backward" > "def"). Qdrant can apply
    IDF automatically only if you use its built-in FastEmbed sparse vectorizer
    (which builds a vocabulary from your corpus). When you supply raw sparse
    vectors manually (as we do here), Qdrant treats them as-is — no IDF,
    no document length normalisation.

    Practical effect: exact identifier lookups still work well (TF alone finds
    the chunk containing "backward" 5 times better than one mentioning it once).
    But stop-word-heavy queries ("how does the") may score higher than they
    should. For a code search use case this is a reasonable trade-off since
    identifiers are the dominant signal and they're naturally rare.

    Upgrade path: switch to Qdrant's FastEmbed sparse vectorizer for true BM25.

    WHY NOT hash(token)?
      Python's built-in hash() is randomised per process (PYTHONHASHSEED).
      The same token gets a different integer in each run, so query vectors
      and stored vectors would map to completely different dimensions —
      keyword search would return random noise. hashlib.md5 is stable.
    """
    from collections import Counter
    import re

    # Tokenise: lowercase, split on non-alphanumeric, filter short tokens
    tokens = re.findall(r"[a-zA-Z_]\w*", text.lower())
    token_counts = Counter(tokens)

    # Map tokens to stable integer indices using MD5 (process-invariant).
    # Using the first 8 hex chars = 32-bit integer, then mod 2^20 dimensions.
    # Two different tokens can collide to the same index (hash collision).
    # Qdrant requires unique indices per sparse vector, so we sum the values
    # for any colliding tokens rather than emitting duplicate indices.
    index_map: dict[int, float] = {}
    for token, count in token_counts.items():
        idx = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % (2 ** 20)
        index_map[idx] = index_map.get(idx, 0.0) + float(count)

    return SparseVector(indices=list(index_map.keys()), values=list(index_map.values()))

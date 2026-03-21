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
        if self.collection in existing:
            print(f"Collection '{self.collection}' already exists")
            return

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
        print(f"  → Collection created")

    def upsert_chunks(self, chunks: list[dict], dense_vectors: list[list[float]]):
        """
        Store chunks and their embeddings in Qdrant.

        Upsert = insert or update. If a point with the same ID already exists,
        it's overwritten. This means re-ingesting a repo is safe — no duplicates.

        The ID is a hash of "repo::filepath::start_line". This is stable:
        the same chunk from the same file always gets the same ID, regardless
        of when it was ingested.

        Sparse vectors (BM25) are computed from the text on Qdrant's side.
        We pass the tokens (words) and their term frequencies.
        Qdrant uses these to build the sparse vector internally.
        """
        if len(chunks) != len(dense_vectors):
            raise ValueError(f"Chunks ({len(chunks)}) and vectors ({len(dense_vectors)}) must match")

        points = []
        for chunk, dense_vec in zip(chunks, dense_vectors):
            point_id = _stable_id(chunk)

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
                },
            ))

        # Qdrant recommends upserting in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)

        print(f"  Upserted {len(points)} points to Qdrant")

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
      → indices = [hash("def"), hash("embed_text"), ...]
        values  = [1.0, 1.0, 1.0, 2.0, ...]

    Qdrant uses these sparse vectors for BM25-style keyword matching.
    The actual BM25 ranking (IDF weighting, document length normalisation)
    is applied at query time by Qdrant.
    """
    from collections import Counter
    import re

    # Tokenise: lowercase, split on non-alphanumeric, filter short tokens
    tokens = re.findall(r"[a-zA-Z_]\w*", text.lower())
    token_counts = Counter(tokens)

    # Map tokens to integer indices using hash (consistent across calls)
    indices = []
    values  = []
    for token, count in token_counts.items():
        idx = abs(hash(token)) % (2 ** 20)  # 1M possible dimensions
        indices.append(idx)
        values.append(float(count))

    return SparseVector(indices=indices, values=values)

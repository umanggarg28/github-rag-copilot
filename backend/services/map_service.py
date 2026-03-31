"""
map_service.py — Semantic Code Map: project code embeddings to 2D.

═══════════════════════════════════════════════════════════════
WHAT IS THE SEMANTIC CODE MAP?
═══════════════════════════════════════════════════════════════

Each code chunk (function/class) is stored in Qdrant with a 768-dim
embedding. These embeddings encode semantic meaning — two functions
that do similar things will have similar vectors, even if they have
different names or live in different files.

The Semantic Code Map makes this invisible structure visible:
  - Fetches all chunk embeddings for a repo
  - Projects from 768-dim → 2-dim using UMAP (or PCA fallback)
  - Assigns each chunk to a semantic cluster via K-means
  - Returns (x, y, cluster_id) + metadata for every chunk

In the resulting map, semantically similar code clusters together:
  - All neural network math operations → one cluster
  - All I/O and parsing utilities → another cluster
  - All test helpers → another cluster

═══════════════════════════════════════════════════════════════
WHY UMAP (vs PCA/t-SNE)?
═══════════════════════════════════════════════════════════════

UMAP (Uniform Manifold Approximation and Projection):
  + Preserves BOTH local cluster structure AND global topology
  + O(n log n) — handles 2000 chunks in ~1–3s on CPU
  + Supports transform() — new points can be projected onto the
    existing layout without recomputing everything. This is critical
    for the query overlay: when a user asks a question, we embed the
    query and project it onto the map to show WHERE the question lands.
  + metric="cosine" matches our stored vectors exactly

PCA fallback is used if umap-learn is not installed:
  + Deterministic, near-instant
  - Preserves variance (global structure) but not local clusters —
    nearby PCA points are not necessarily semantically near

═══════════════════════════════════════════════════════════════
THE RETRIEVAL OVERLAY
═══════════════════════════════════════════════════════════════

After a RAG query, the frontend highlights which chunks were retrieved.
The query embedding is projected onto the map, showing WHERE in semantic
space the question landed — making the invisible retrieval step visible.
"""

import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from ingestion.qdrant_store import QdrantStore

# 8 visually distinct cluster colors (colorblind-safe-ish, chosen for dark bg)
CLUSTER_COLORS = [
    "#7C3AED",  # violet
    "#2DD4BF",  # teal
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#10B981",  # emerald
    "#6366F1",  # indigo
    "#EC4899",  # pink
    "#84CC16",  # lime
]


class MapService:
    """
    Builds and caches the 2D semantic map for indexed repos.

    The map is computed once per repo on first request and cached in memory.
    Call invalidate(repo) after re-ingestion to force a rebuild.
    """

    def __init__(self, store: QdrantStore):
        self._store = store
        # Cache: repo_slug → {"nodes": [...], "projector": fitted reducer, ...}
        self._cache: dict = {}

    def build_map(self, repo: str) -> dict:
        """
        Build (or return cached) the 2D semantic map for a repo.

        Returns:
            {
              "nodes": [
                {
                  "id":         str,   # Qdrant point ID
                  "x":          float, # 2D coordinate (0–1000)
                  "y":          float,
                  "name":       str,
                  "filepath":   str,
                  "chunk_type": str,
                  "start_line": int,
                  "end_line":   int,
                  "cluster_id": int,
                }
              ],
              "clusters": [{"id": int, "label": str, "color": str}],
              "stats": {"node_count": int}
            }
        """
        if repo in self._cache:
            c = self._cache[repo]
            return {"nodes": c["nodes"], "clusters": c["clusters"], "stats": c["stats"]}

        raw_points = self._scroll_with_vectors(repo)
        if not raw_points:
            return {"nodes": [], "clusters": [], "stats": {"node_count": 0}}

        ids        = [p["id"]      for p in raw_points]
        embeddings = np.array([p["vector"]  for p in raw_points], dtype=np.float32)
        payloads   = [p["payload"] for p in raw_points]
        n          = len(embeddings)

        # ── 2D projection ──────────────────────────────────────────────────────
        coords, projector = self._project(embeddings)

        # Normalise to [50, 950]
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0

        # ── K-means clustering ─────────────────────────────────────────────────
        # Cluster in the 2D UMAP space (not the 768-dim space) so cluster IDs
        # correspond visually to what the user sees on the map.
        k = max(2, min(8, n // 20))  # 2–8 clusters depending on corpus size
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        cluster_ids = kmeans.fit_predict(coords).tolist()

        # ── Cluster labels: most common directory prefix ───────────────────────
        cluster_paths: dict[int, list[str]] = {i: [] for i in range(k)}
        for cid, payload in zip(cluster_ids, payloads):
            fp = payload.get("filepath", "")
            # Take the first path segment (directory name) as the label candidate
            parts = fp.replace("\\", "/").split("/")
            cluster_paths[cid].append(parts[0] if len(parts) > 1 else fp)

        clusters = []
        for cid in range(k):
            most_common = Counter(cluster_paths[cid]).most_common(1)
            label = most_common[0][0] if most_common else f"cluster {cid}"
            clusters.append({
                "id":    cid,
                "label": label,
                "color": CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            })

        # ── Build node list ────────────────────────────────────────────────────
        nodes = []
        for i, (pid, payload, cid) in enumerate(zip(ids, payloads, cluster_ids)):
            x = 50 + (coords[i, 0] - x_min) / x_range * 900
            y = 50 + (coords[i, 1] - y_min) / y_range * 900
            nodes.append({
                "id":         str(pid),
                "x":          float(x),
                "y":          float(y),
                "name":       payload.get("name") or payload.get("function_name") or "",
                "filepath":   payload.get("filepath", ""),
                "chunk_type": payload.get("chunk_type", "function"),
                "start_line": payload.get("start_line", 0),
                "end_line":   payload.get("end_line", 0),
                "cluster_id": int(cid),
            })

        stats = {"node_count": n}
        self._cache[repo] = {
            "nodes": nodes, "clusters": clusters, "stats": stats,
            "projector": projector,
            "x_min": x_min, "x_range": x_range,
            "y_min": y_min, "y_range": y_range,
        }
        return {"nodes": nodes, "clusters": clusters, "stats": stats}

    def project_embedding(self, repo: str, embedding: list[float]) -> tuple[float, float] | None:
        """
        Project a single new embedding (query vector) onto the cached 2D map.

        Returns (x, y) in the same coordinate space as the map nodes, or None
        if the map hasn't been built for this repo yet.

        This is what makes the retrieval overlay work: we embed the user's
        question and project it to show WHERE it lands in semantic space.
        """
        if repo not in self._cache:
            return None
        cached   = self._cache[repo]
        vec      = np.array([embedding], dtype=np.float32)
        coord    = cached["projector"].transform(vec)[0]
        x = 50 + (coord[0] - cached["x_min"]) / cached["x_range"] * 900
        y = 50 + (coord[1] - cached["y_min"]) / cached["y_range"] * 900
        return float(x), float(y)

    def invalidate(self, repo: str):
        """Remove a repo from the cache — call after re-ingestion."""
        self._cache.pop(repo, None)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _project(self, embeddings: np.ndarray) -> tuple[np.ndarray, object]:
        """
        Project embeddings to 2D. Uses UMAP if available, falls back to PCA.
        Returns (coords array shape (N,2), fitted projector).
        """
        n = len(embeddings)
        try:
            import umap
            # n_neighbors must be < n; 15 is a good default for code chunks
            n_neighbors = min(15, max(2, n - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,   # deterministic layout
            )
            coords = reducer.fit_transform(embeddings)
            return coords, reducer
        except ImportError:
            # umap-learn not installed — use PCA as fallback
            n_components = min(2, n)
            pca = PCA(n_components=n_components, random_state=42)
            coords = pca.fit_transform(embeddings)
            if n_components == 1:
                coords = np.column_stack([coords, np.zeros(n)])
            return coords, pca

    def _scroll_with_vectors(self, repo: str) -> list[dict]:
        """Fetch all Qdrant points for a repo WITH their dense vectors."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = []
        offset  = None
        store   = self._store
        filt    = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))])

        while True:
            points, offset = store.client.scroll(
                collection_name=store.collection,
                scroll_filter=filt,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                vec = p.vector
                if isinstance(vec, dict):
                    vec = vec.get("code") or next(iter(vec.values()))
                if vec is not None:
                    results.append({"id": p.id, "vector": vec, "payload": p.payload})
            if offset is None:
                break

        return results

"""
ingestion_service.py — Orchestrates the full ingestion pipeline.

Pipeline steps:
  1. Fetch   — download the repo as a zip from GitHub API
  2. Filter  — keep only indexable files (by extension, not excluded dirs)
  3. Chunk   — split files into function/class chunks (AST) or windows (fallback)
  4. Embed   — convert chunk text to dense vectors (sentence-transformers)
  5. Store   — upsert chunks + vectors + BM25 sparse vectors to Qdrant

This file is the glue between the five ingestion modules. Each module
handles one concern; this service wires them together into a single call:

  result = IngestionService().ingest("https://github.com/karpathy/micrograd")

Why a service class instead of a function?
  The Embedder loads a 600MB model on init. If this were a function, it would
  reload the model on every request. As a class instantiated once (at app
  startup via FastAPI lifespan), the model stays in memory between requests.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.repo_fetcher import parse_github_url, fetch_repo_files
from ingestion.file_filter import should_index, language_from_path
from ingestion.code_chunker import chunk_files
from ingestion.embedder import Embedder
from ingestion.qdrant_store import QdrantStore
from backend.config import settings


class IngestionService:
    """
    End-to-end pipeline: GitHub URL → Qdrant points.

    Shared state:
      - self.embedder  — kept alive so the model isn't reloaded per request
      - self.store     — keeps the Qdrant client open (HTTP connection pooling)

    Why accept store as an argument?
      main.py creates one QdrantStore and shares it across IngestionService,
      GraphService, and the MCP server. A single client means one connection
      pool, one auth handshake, and consistent state across all services.
    """

    def __init__(self, store: QdrantStore | None = None, embedder=None, gen=None):
        self.embedder = embedder or Embedder()
        self.store    = store or QdrantStore()
        self._gen     = gen   # optional GenerationService for contextual retrieval

    def ingest(self, repo_url: str, force: bool = False, progress: callable = None) -> dict:
        """
        Run the full ingestion pipeline for one repository.

        Args:
            repo_url: GitHub URL (https://github.com/owner/repo)
            force:    If True, delete all existing chunks for this repo first.
                      Use this to re-index after the repo has changed.
            progress: Optional callback(step: str, detail: str) called at key
                      milestones. Used by the SSE streaming endpoint to push
                      real-time updates to the UI without blocking the event loop.
                      Defaults to None so existing callers don't break.

        Returns:
            dict with keys: repo, files_indexed, chunks_stored, message
        """
        def _emit(step: str, detail: str) -> None:
            # Only call the callback if one was provided — safe to skip otherwise.
            if progress is not None:
                progress(step, detail)

        # ── Step 1: Parse URL & optionally clear old data ─────────────────────
        owner, name = parse_github_url(repo_url)
        repo_slug   = f"{owner}/{name}"
        print(f"\n=== Ingesting {repo_slug} ===")

        if force:
            deleted = self.store.delete_repo(repo_slug)
            print(f"  Deleted {deleted} existing chunks for {repo_slug}")

        # ── Step 2: Download repo (filtering happens inside fetch_repo_files) ────
        _emit("fetching", "Fetching file list from GitHub...")
        print("Fetching repo files from GitHub...")
        raw_files = fetch_repo_files(repo_url, should_index)
        # raw_files is a list of dicts: [{"path": "...", "content": "...", "size": N}, ...]
        print(f"  Downloaded {len(raw_files)} indexable files")
        _emit("filtering", f"Found {len(raw_files)} files, filtering...")

        if not raw_files:
            _emit("done", "No indexable files found in this repository.")
            return {
                "repo":          repo_slug,
                "files_indexed": 0,
                "chunks_stored": 0,
                "message":       "No indexable files found in this repository.",
            }

        # ── Step 3: Build file dicts with metadata ─────────────────────────────
        # chunk_files() expects dicts with keys: filepath, content, language, repo
        file_dicts = [
            {
                "filepath": f["path"],
                "content":  f["content"],
                "language": language_from_path(f["path"]),
                "repo":     repo_slug,
            }
            for f in raw_files
        ]

        # ── Step 5: Chunk ─────────────────────────────────────────────────────
        _emit("chunking", f"Chunking {len(file_dicts)} files...")
        print("Chunking files...")
        chunks = chunk_files(file_dicts)
        print(f"  Produced {len(chunks)} chunks from {len(file_dicts)} files")

        if not chunks:
            _emit("done", "Files found but no chunks produced (files may be empty).")
            return {
                "repo":          repo_slug,
                "files_indexed": len(raw_files),
                "chunks_stored": 0,
                "message":       "Files found but no chunks produced (files may be empty).",
            }

        # ── Step 5b: Contextual Retrieval (optional) ──────────────────────────
        # Anthropic's "Contextual Retrieval" technique: before embedding each
        # chunk, prepend a short LLM-generated description of that chunk's role
        # within its file. This dramatically improves search quality because the
        # embedding encodes "what this chunk does" rather than just its raw text.
        #
        # Without context:  "def forward(self, x): ..."  (embedding = math operations)
        # With context:     "This is the main forward pass of the Transformer model,
        #                    calling attention + FFN + residual connections. def forward..."
        #                   (embedding = architectural role)
        #
        # Free-tier friendly: we only contextualise the top 50 most important chunks
        # (classes + key files) to stay within free-tier rate limits. The top chunks
        # benefit most because they're retrieved most often.
        # Only runs on force=True re-ingestion to avoid slowing down first-time indexing.
        if force and hasattr(self, '_gen') and self._gen is not None:
            top_n = settings.contextual_top_n  # 0 = all chunks
            limit_str = f"all {len(chunks)}" if top_n == 0 else f"top {top_n}"
            _emit("contextualizing",
                  f"Contextual retrieval: adding AI-generated descriptions to {limit_str} chunks…")
            print(f"Contextual retrieval: enriching {limit_str} chunks with context...")
            now = datetime.now(timezone.utc).isoformat()
            chunks = _add_context(chunks, file_dicts, self._gen, top_n=top_n, contextual_at=now)
            n_enriched = sum(1 for c in chunks if c.get('_contextualised'))
            print(f"  Context added to {n_enriched} chunks")

        # ── Step 6: Deduplicate, then embed only new chunks ────────────────────
        # Compute a content hash for each chunk so we can detect identical code
        # that already exists in the index (possibly from a different repo).
        # The hash is stored in the payload so future ingestions can look it up.
        for chunk in chunks:
            chunk["text_hash"] = hashlib.sha256(chunk["text"].encode()).hexdigest()

        # Ask Qdrant which hashes already have vectors. For a first ingest this
        # returns {} (nothing cached). On re-ingest or cross-repo ingest, it
        # avoids re-embedding any chunk whose text hasn't changed.
        _emit("embedding", f"Checking for reusable embeddings...")
        existing_vectors = self.store.find_vectors_by_hash(
            [c["text_hash"] for c in chunks]
        )
        reused_count = sum(1 for c in chunks if c["text_hash"] in existing_vectors)
        new_chunks    = [c for c in chunks if c["text_hash"] not in existing_vectors]

        if reused_count:
            print(f"  Deduplication: {reused_count}/{len(chunks)} chunks reuse existing embeddings")
            _emit("embedding", f"Embedding {len(new_chunks)} new chunks ({reused_count} reused from index)...")
        else:
            _emit("embedding", f"Embedding {len(chunks)} chunks...")

        print("Embedding chunks...")
        new_vectors = self.embedder.embed_chunks(new_chunks) if new_chunks else []
        if new_vectors:
            print(f"  Produced {len(new_vectors)} vectors ({len(new_vectors[0])}-dim each)")

        # Reconstruct the full vectors list in original chunk order.
        # Chunks with existing vectors use the stored vector; new ones use the
        # freshly computed one. This preserves the 1-to-1 chunks↔vectors pairing
        # that upsert_chunks requires.
        new_hash_to_vec = {c["text_hash"]: v for c, v in zip(new_chunks, new_vectors)}
        vectors = [
            existing_vectors.get(c["text_hash"]) or new_hash_to_vec[c["text_hash"]]
            for c in chunks
        ]

        # ── Step 7: Store ─────────────────────────────────────────────────────
        _emit("storing", f"Storing {len(chunks)} chunks in Qdrant...")
        print("Storing in Qdrant...")
        self.store.upsert_chunks(chunks, vectors)

        total_stored = self.store.count(repo=repo_slug)
        message = (
            f"Ingested {repo_slug}: "
            f"{len(raw_files)} files → {len(chunks)} chunks → {total_stored} total stored"
        )
        print(f"\n✓ {message}")
        _emit("done", f"Indexed {len(chunks)} chunks from {len(raw_files)} files")

        return {
            "repo":          repo_slug,
            "files_indexed": len(raw_files),
            "chunks_stored": len(chunks),
            "message":       message,
        }

    def list_repos(self) -> list[dict]:
        """
        Return all indexed repos with their chunk counts.

        Used by the UI to populate the repo selector dropdown.
        Returns a list of dicts: [{slug, chunks}, ...]
        """
        slugs = self.store.list_repos()
        return [
            {"slug": slug, "chunks": self.store.count(repo=slug)}
            for slug in slugs
        ]

    def delete_repo(self, repo_slug: str) -> int:
        """Delete all chunks for a repo. Returns the number deleted."""
        return self.store.delete_repo(repo_slug)


# ── Contextual Retrieval ───────────────────────────────────────────────────────

_CONTEXT_SYSTEM = (
    "You are a precise technical writer. Write a single sentence that situates a "
    "code chunk within its file. Output ONLY that one sentence — no preamble, no quotes."
)

# Importance scoring mirrors diagram_service's chunk ranking.
# Higher score = more important = more benefit from contextual embedding.
_ENTRY_FILES   = {"main.py", "app.py", "server.py", "__init__.py", "agent.py",
                  "train.py", "model.py", "pipeline.py", "engine.py"}


def _chunk_importance(c: dict) -> int:
    score = 0
    if c.get("chunk_type") in ("class", "module"):
        score += 10
    fname = (c.get("filepath", "") or "").split("/")[-1]
    if fname in _ENTRY_FILES:
        score += 8
    if c.get("base_classes"):
        score += 2
    return score


def _add_context(
    chunks: list[dict],
    file_dicts: list[dict],
    gen,
    top_n: int = 0,
    contextual_at: str = None,
) -> list[dict]:
    """
    Contextual Retrieval: prepend a short LLM-generated context sentence to
    each high-importance chunk before embedding.

    WHY THIS WORKS
    ──────────────
    When you embed a short function like:
        def _relu(x): return max(0, x)
    the resulting vector captures "arithmetic on a scalar" — it matches queries
    about math, not about neural network activations.

    With a prepended context sentence:
        "ReLU is the activation function used in every layer to introduce
         non-linearity. def _relu(x): return max(0, x)"
    the vector now captures "neural network activation" — it correctly retrieves
    this chunk for "how does the network add non-linearity?".

    COVERAGE
    ────────
    top_n=0 means ALL chunks are contextualised — best quality but slow for
    large repos (each chunk = 1 LLM call, Groq free tier: 30 req/min).
    top_n=50 is the conservative limit: only enrich the most important chunks.

    PERSISTENCE
    ───────────
    contextual_at is stamped on ALL chunks (not just enriched ones) so the
    /repos endpoint can read it from Qdrant on restart, rather than relying on
    an in-memory dict that resets with each server restart.

    IF USING ANTHROPIC
    ──────────────────
    Anthropic supports prompt caching: mark the <document> block with
    cache_control={"type": "ephemeral"} and Anthropic caches the KV state once
    per file, reusing it for all chunks in that file — ~50x cheaper.
    """
    # Build filepath → content map
    file_content_map: dict[str, str] = {
        fd["filepath"]: fd["content"] for fd in file_dicts
    }

    # Rank by importance and determine how many to enrich
    ranked = sorted(enumerate(chunks), key=lambda ic: _chunk_importance(ic[1]), reverse=True)
    limit  = top_n if top_n > 0 else len(ranked)   # 0 = all

    result = list(chunks)   # copy; we mutate by index

    # Stamp contextual_at on ALL chunks upfront so the timestamp is persisted
    # in Qdrant regardless of which chunks actually got enriched.
    if contextual_at:
        for i in range(len(result)):
            result[i] = dict(result[i])
            result[i]["contextual_at"] = contextual_at

    for idx, _chunk in ranked[:limit]:
        # Use result[idx] (which already has contextual_at stamped) as the base
        chunk      = result[idx]
        filepath   = chunk.get("filepath", "")
        chunk_text = chunk.get("text", "")
        doc_text   = file_content_map.get(filepath, "")[:6000]   # truncate large files

        if not chunk_text or not doc_text:
            continue

        prompt = (
            f"File: {filepath}\n\n"
            f"File content (truncated):\n{doc_text}\n\n"
            f"Chunk to describe:\n{chunk_text[:800]}\n\n"
            "Write ONE sentence describing what this chunk does and its role in the file."
        )
        try:
            context_sentence = gen.generate(_CONTEXT_SYSTEM, prompt, temperature=0.0).strip()
            updated                    = dict(chunk)
            updated["raw_text"]        = chunk_text   # original code kept for display
            updated["text"]            = f"{context_sentence}\n\n{chunk_text}"
            updated["_contextualised"] = True
            result[idx] = updated
        except Exception as e:
            print(f"  Context skipped for {filepath}:{chunk.get('name', '?')} — {e}")
            # Leave result[idx] unchanged — graceful fallback to raw chunk

    return result

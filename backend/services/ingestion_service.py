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
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.repo_fetcher import parse_github_url, fetch_repo_files
from ingestion.file_filter import should_index, language_from_path
from ingestion.code_chunker import chunk_files
from ingestion.embedder import Embedder
from ingestion.qdrant_store import QdrantStore


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

    def __init__(self, store: QdrantStore | None = None, embedder=None):
        self.embedder = embedder or Embedder()
        self.store    = store or QdrantStore()

    def ingest(self, repo_url: str, force: bool = False) -> dict:
        """
        Run the full ingestion pipeline for one repository.

        Args:
            repo_url: GitHub URL (https://github.com/owner/repo)
            force:    If True, delete all existing chunks for this repo first.
                      Use this to re-index after the repo has changed.

        Returns:
            dict with keys: repo, files_indexed, chunks_stored, message
        """
        # ── Step 1: Parse URL & optionally clear old data ─────────────────────
        owner, name = parse_github_url(repo_url)
        repo_slug   = f"{owner}/{name}"
        print(f"\n=== Ingesting {repo_slug} ===")

        if force:
            deleted = self.store.delete_repo(repo_slug)
            print(f"  Deleted {deleted} existing chunks for {repo_slug}")

        # ── Step 2: Download repo (filtering happens inside fetch_repo_files) ────
        print("Fetching repo files from GitHub...")
        raw_files = fetch_repo_files(repo_url, should_index)
        # raw_files is a list of dicts: [{"path": "...", "content": "...", "size": N}, ...]
        print(f"  Downloaded {len(raw_files)} indexable files")

        if not raw_files:
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
        print("Chunking files...")
        chunks = chunk_files(file_dicts)
        print(f"  Produced {len(chunks)} chunks from {len(file_dicts)} files")

        if not chunks:
            return {
                "repo":          repo_slug,
                "files_indexed": len(raw_files),
                "chunks_stored": 0,
                "message":       "Files found but no chunks produced (files may be empty).",
            }

        # ── Step 6: Embed ─────────────────────────────────────────────────────
        print("Embedding chunks...")
        vectors = self.embedder.embed_chunks(chunks)
        print(f"  Produced {len(vectors)} vectors ({len(vectors[0])}-dim each)")

        # ── Step 7: Store ─────────────────────────────────────────────────────
        print("Storing in Qdrant...")
        self.store.upsert_chunks(chunks, vectors)

        total_stored = self.store.count(repo=repo_slug)
        message = (
            f"Ingested {repo_slug}: "
            f"{len(raw_files)} files → {len(chunks)} chunks → {total_stored} total stored"
        )
        print(f"\n✓ {message}")

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

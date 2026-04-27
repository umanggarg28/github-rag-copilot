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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        # ── Step 1: Parse URL ─────────────────────────────────────────────────
        # We no longer delete upfront on force=True. Instead we upsert new chunks
        # first and delete stale ones afterwards (blue-green strategy). This keeps
        # the repo visible in the sidebar throughout the re-index — critical when
        # contextual enrichment takes 5-10 minutes on free-tier LLM APIs.
        owner, name = parse_github_url(repo_url)
        repo_slug   = f"{owner}/{name}"
        print(f"\n=== Ingesting {repo_slug} (force={force}) ===")

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
            total_ctx = len(chunks) if top_n == 0 else min(top_n, len(chunks))
            _emit("contextualizing",
                  f"Adding context to chunks… 0 / {total_ctx}")
            print(f"Contextual retrieval: enriching {total_ctx} chunks with context...")
            now = datetime.now(timezone.utc).isoformat()

            # Emit progress every 20 chunks so the UI bar advances visibly
            def _ctx_progress(done: int, total: int) -> None:
                _emit("contextualizing", f"Adding context to chunks… {done} / {total}")

            chunks = _add_context(
                chunks, file_dicts, self._gen,
                top_n=top_n, contextual_at=now,
                progress=_ctx_progress,
            )
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

        # ── Step 7: Embed + upsert in checkpointed groups ─────────────────────
        # Stream embed→upsert in groups so a crash mid-run leaves earlier
        # chunks safely in Qdrant. Retry then skips them via the existing
        # find_vectors_by_hash dedup path above. Without checkpoints, a 15-min
        # ingest that dies at chunk 13000/13594 loses 100% of the work.
        #
        # Group size 500: big enough that Qdrant upsert overhead amortises,
        # small enough that a crash loses at most ~500 re-embeddings on retry.
        CHECKPOINT_SIZE = 500
        total_new = len(new_chunks)
        new_done  = 0
        written_ids: list = []

        # Progress callback for the embedder — maps batch-level progress
        # within a checkpoint group to an overall "chunks embedded / total"
        # count. `new_done` snapshots the running total across groups.
        def _embed_progress(batch_done: int, batch_total: int) -> None:
            overall_done = new_done + batch_done
            _emit("embedding", f"Embedded {overall_done}/{total_new} chunks...")

        for group_start in range(0, len(chunks), CHECKPOINT_SIZE):
            group = chunks[group_start : group_start + CHECKPOINT_SIZE]

            # Within the group, split reused vs new. Only the new ones hit
            # the embedding API; reused chunks pull from `existing_vectors`.
            group_new_chunks = [c for c in group if c["text_hash"] not in existing_vectors]

            if group_new_chunks:
                group_new_vectors = self.embedder.embed_chunks(
                    group_new_chunks, progress=_embed_progress,
                )
                new_done += len(group_new_chunks)
            else:
                group_new_vectors = []

            # Stitch back into group order so each chunk lines up with its vector.
            group_hash_to_vec = {
                c["text_hash"]: v for c, v in zip(group_new_chunks, group_new_vectors)
            }
            group_vectors = [
                existing_vectors.get(c["text_hash"]) or group_hash_to_vec[c["text_hash"]]
                for c in group
            ]

            # Upsert this group before touching the next — that's the actual
            # checkpoint. If the next group's embedding call dies, everything
            # up to here is already in Qdrant.
            #
            # Deliberately NOT emitting a "storing" event here: the UI phase
            # would flicker embedding → storing → embedding every group,
            # yanking the progress bar between its embed-range and its
            # storing-range. The user perceives one continuous "embedding"
            # phase, so keep the phase stable and let the per-batch progress
            # emits from the embedder drive the bar smoothly.
            group_ids = self.store.upsert_chunks(group, group_vectors)
            written_ids.extend(group_ids)
            print(f"  Checkpoint {group_start + len(group)}/{len(chunks)} stored")

        # On a force re-index, delete chunks that no longer exist in the source.
        # This handles deleted files and renamed functions — their old IDs won't
        # appear in written_ids, so they get swept up here.
        if force:
            _emit("storing", "Removing stale chunks...")
            stale = self.store.delete_stale_chunks(repo_slug, set(written_ids))
            if stale:
                print(f"  Swept {stale} stale chunks from previous index")

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
    "You are a code indexing assistant. Your output is prepended to a code chunk before "
    "it is embedded in a vector database — it must make the embedding match the queries "
    "developers actually type, not explain the code's failure modes. "
    "Write 1-2 sentences that situate this chunk within the document: "
    "name the function/class, state its role in the file's pipeline, and name the "
    "key identifier(s) a developer would search for to find this code. "
    "NEVER write 'This chunk', 'This code', or failure scenarios. "
    "NEVER invent behaviour not in the source. "
    "Output ONLY the 1-2 sentences — no preamble, no quotes."
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


def _anthropic_contextualise(
    client, model: str, system: str, doc_text: str, chunk_question: str,
    max_tokens: int = 200,
) -> str:
    """
    Call Anthropic with prompt caching on the document block.

    Structure:
      system: _CONTEXT_SYSTEM
      user message:
        block 1 — <document>...</document>  with cache_control=ephemeral
        block 2 — chunk + question          (not cached; varies per chunk)

    The ephemeral cache lives for 5 minutes (Anthropic's TTL for beta caching).
    Within a single contextualisation run all chunks for the same file are
    processed in the same thread pool — close enough together to hit the cache.

    Cache write tokens count at 1.25× normal; cache read tokens at 0.1× normal.
    For a file with 10 chunks: 1 write + 9 reads vs 10 full document reads.
    Break-even at 2 chunks per file; most source files have 5-20+.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<document>\n{doc_text}\n</document>",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "\n\n" + chunk_question,
                },
            ],
        }],
    )
    return resp.content[0].text.strip()


def _add_context(
    chunks: list[dict],
    file_dicts: list[dict],
    gen,
    top_n: int = 0,
    contextual_at: str = None,
    progress: callable = None,
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

    # Detect whether the active provider supports prompt caching.
    # When using Anthropic, the <document> block can be marked ephemeral so
    # the KV state is cached once per file and reused for all chunks in that
    # file — ~50x cheaper than embedding the full document on every call.
    # The cached block must be ≥1024 tokens to qualify; typical source files
    # are 200-6000 tokens, so most will qualify.
    _use_anthropic_cache = getattr(gen, 'provider', None) == 'anthropic'

    # Tunable caps — bumped automatically when gen.premium_mode is on so a
    # premium prebake includes more of each chunk and surrounding file context
    # than the free-tier defaults allow.
    _doc_chars      = gen.cap("context_doc_chars",   6000)
    _chunk_chars    = gen.cap("context_chunk_chars",  800)
    _ctx_max_tokens = gen.cap("context_chunk_tokens", 200)

    # Worker function for a single chunk — called from multiple threads.
    # Returns (idx, updated_chunk) or (idx, None) on failure.
    def _enrich_one(idx: int, chunk: dict) -> tuple[int, dict | None]:
        filepath   = chunk.get("filepath", "")
        chunk_text = chunk.get("text", "")
        doc_text   = file_content_map.get(filepath, "")[:_doc_chars]
        if not chunk_text or not doc_text:
            return idx, None

        chunk_question = (
            f"Here is the chunk we want to situate within the document above:\n"
            f"<chunk>\n{chunk_text[:_chunk_chars]}\n</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall "
            "document for the purpose of improving search retrieval of the chunk. "
            "Name the function/class/block, its role in the file's pipeline, and the key "
            "identifiers a developer would search to find it. "
            "Answer only with the succinct context and nothing else."
        )

        try:
            if _use_anthropic_cache:
                # Anthropic prompt caching: mark the document block as ephemeral.
                # The Anthropic API caches the KV state for the document once per
                # unique document text; subsequent calls with the same document
                # (i.e. other chunks from the same file) reuse the cache at ~10%
                # of the original token cost. This turns O(N_chunks) document
                # processing into O(N_files) full-cost calls.
                sentence = _anthropic_contextualise(
                    gen._client, gen._model, _CONTEXT_SYSTEM,
                    doc_text, chunk_question, max_tokens=_ctx_max_tokens,
                )
            else:
                prompt = (
                    f"<document>\n{doc_text}\n</document>\n\n"
                    + chunk_question
                )
                # fast=True: use the lightweight model tier for enrichment.
                # Contextual enrichment is 1-2 sentences — does not require the
                # strong synthesis model. This preserves quota for tour/diagram calls.
                sentence = gen.generate(_CONTEXT_SYSTEM, prompt, temperature=0.0, fast=True).strip()

            updated              = dict(chunk)
            updated["raw_text"]  = chunk_text
            updated["text"]      = f"{sentence}\n\n{chunk_text}"
            updated["_contextualised"] = True
            return idx, updated
        except Exception as e:
            print(f"  Context skipped for {filepath}:{chunk.get('name', '?')} — {e}")
            return idx, None

    # Run up to 3 LLM calls concurrently. We tried 8 workers but all free-tier
    # providers (Gemini 15 RPM, OpenRouter 8 RPM, Groq TPM) hit limits simultaneously,
    # causing cascading 429s across every provider at once — slower than 3 workers.
    # 3 workers gives ~3x speedup while keeping the per-minute request count
    # low enough that the generation service's retry/fallback logic can breathe.
    _WORKERS     = 3
    _REPORT_EVERY = 20
    to_enrich = [(idx, result[idx]) for idx, _ in ranked[:limit]]

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_enrich_one, idx, chunk): i
                   for i, (idx, chunk) in enumerate(to_enrich)}
        done_count = 0
        for future in as_completed(futures):
            idx, updated = future.result()
            done_count += 1
            if updated is not None:
                result[idx] = updated
            if progress and (done_count % _REPORT_EVERY == 0 or done_count == limit):
                progress(done_count, limit)

    return result

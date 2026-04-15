"""
tour_agent.py — Multi-step agent for building codebase concept tours.

WHY AN AGENT INSTEAD OF ONE-SHOT
──────────────────────────────────
One-shot generation sends 50 ranked code chunks to the LLM and asks it to
simultaneously understand the architecture, trace the pipeline, identify
design decisions, and format JSON. The LLM pattern-matches across 35,000
chars of disconnected snippets — it summarises rather than reasons.

Result: concepts that are individually plausible but miss the system-level
flow. Dependency graphs are guessed, not traced. The foundational concept
is rarely a pipeline overview.

This agent separates understanding from formatting into three focused phases,
modelled directly on how Claude Code's /init command surveys a codebase:

  Phase 0 — README   (= /init Phase 2: "read key files before touching code")
    Read README.md and manifest files from the index.
    This grounds Phase 1 in what the repo CLAIMS to do, not just what the
    code does. The mismatch between stated purpose and implementation is
    itself a signal worth surfacing.

  Phase 1 — MAP   (= architecture_overview)
    "What is the main pipeline and which files own each stage?"
    Input:  README summary + module-level chunks from entry files
    Output: { entry_file, readme_summary, pipeline_stages: [{name, file, key_aspect}] }

  Phase 2 — INVESTIGATE   (= explain_tool × N, "start broad, narrow down")
    WHY / HOW / WHERE / WHAT — plus "what can't the code tell you?"
    Input:  all chunks for one file + related imported files if sparse
    Output: { name, subtitle, insight, key_functions, naive_rejected, gaps }

  Phase 3 — SYNTHESIZE
    "Convert traced understanding into tour JSON."
    Input:  README summary + pipeline map + per-stage insights (no raw code)
    Output: full tour JSON with fan-out (not chain) dependency graph

Inspired by Claude Code source:
  - /init NEW_INIT_PROMPT: reads README, manifests, CI configs before generating
  - MagicDocs: WHY/HOW/WHERE/WHAT framing, terse high-signal documentation
  - Explore agent: start broad, narrow down; parallel search strategies
  - AgentTool prompt: "never delegate understanding", focused context per call

TRADE-OFFS
───────────
One-shot: ~10-20s, 1 LLM call, shallow guessed understanding
Agent:    ~25-50s, 7-10 LLM calls, traced grounded understanding

Phase 2 uses 3 parallel workers (ThreadPoolExecutor) — same approach as
_add_context in ingestion_service.py. 3 workers gives ~3x speedup on typical
5-stage pipelines while staying under free-tier rate limits (15 RPM Gemini).
All-at-once parallelism would hit the rate limiter on 6+ stage pipelines.
"""

from __future__ import annotations

import json as _json
import re as _re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator


# ── Concept-name feedback store ────────────────────────────────────────────────
# When the evaluator corrects a concept name (artifact → technique), we persist
# the bad name so future runs avoid generating it. This is online learning without
# retraining: the system observes its own corrections and injects them as negative
# examples into Phase 3's synthesize prompt.
#
# Feedback is stored per repo (bad names in micrograd are local knowledge, not
# universal). The store is a flat JSON dict: {"bad_name": "corrected_name", ...}.
# We accumulate corrections across runs — the file grows but never shrinks, so the
# model's "avoid list" gets richer over time.

_FEEDBACK_DIR = Path(__file__).parent.parent / "tour_feedback"
_FEEDBACK_DIR.mkdir(exist_ok=True)


def _feedback_path(repo: str) -> Path:
    slug = repo.replace("/", "_")
    return _FEEDBACK_DIR / f"{slug}_feedback.json"


def _load_feedback(repo: str, store=None) -> dict[str, str]:
    """
    Return persisted name corrections for repo: {bad_name: good_name}.

    Tries Qdrant first (durable across HF Space restarts) then falls back
    to the local filesystem. Qdrant is preferred for production deployments
    where container file systems are ephemeral.
    """
    if store is not None:
        try:
            return store.load_tour_feedback(repo)
        except Exception:
            pass  # Qdrant unavailable — fall through to file
    # File-based fallback (local dev / no Qdrant connection)
    p = _feedback_path(repo)
    if p.exists():
        try:
            return _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_feedback(repo: str, corrections: dict[str, str], store=None) -> None:
    """
    Merge new bad→good corrections into the repo's feedback store.

    Saves to Qdrant when available (production), and also mirrors to the local
    file as a backup so local dev sessions accumulate feedback without Qdrant.
    """
    if not corrections:
        return
    if store is not None:
        try:
            store.save_tour_feedback(repo, corrections)
        except Exception as e:
            print(f"TourAgent: Qdrant feedback save failed (non-fatal): {e}")
    # Also write to local file (backup + local dev experience)
    p = _feedback_path(repo)
    existing = _load_feedback(repo)   # read from file for the local merge
    existing.update(corrections)
    try:
        p.write_text(_json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"TourAgent: could not save feedback to file for {repo}: {e}")


def _token_budget(text: str, max_tokens: int) -> str:
    """
    Truncate text to stay within an approximate token budget.

    Uses 4 chars/token as a conservative heuristic — works well for English
    prose mixed with code (actual ratio varies 3-6 chars/token).  We don't
    depend on tiktoken to keep the dependency footprint minimal.

    Called at LLM prompt-building time, not after the fact, so overflow is
    prevented rather than surfaced as a server-side 400 error.
    """
    limit = max_tokens * 4
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    # Snap to the last newline so we don't cut mid-line
    last_nl = truncated.rfind("\n")
    if last_nl > limit // 2:   # only snap if we're not losing too much
        truncated = truncated[:last_nl]
    return truncated + f"\n\n[... truncated at ~{max_tokens} tokens ...]"


def _synthesize_negative_block(repo: str, store=None) -> str:
    """
    Return a prompt fragment listing previously bad concept names for this repo.

    Injected into Phase 3's NAME RULE section. When the evaluator has corrected
    names in prior runs, those artifact names are forbidden so the model won't
    generate them again — saving one evaluator round per run.

    Returns an empty string if no feedback exists yet.
    """
    feedback = _load_feedback(repo, store)
    if not feedback:
        return ""
    lines = [
        f'  PREVIOUSLY REJECTED for this repo (do NOT use these names — they were flagged as artifacts):'
    ]
    for bad, good in list(feedback.items())[:20]:   # cap at 20 to stay token-light
        lines.append(f'    "{bad}" → was corrected to "{good}"')
    return "\n".join(lines) + "\n"


# Pipeline entry files — scripts that run the core data transformation.
# EXCLUDED from this list: main.py, app.py, server.py — these are web-framework
# bootstrap files that import ALL features (routers, services, middleware) equally.
# Their import graph says "everything is equally important", which is the opposite
# of what Phase 1 needs. The pipeline lives in service/library files, not the
# bootstrap. (This is the same principle as claude-code's get_architecture tool —
# it provides a curated structural overview, not the raw import graph of main.py.)
_ENTRY_NAMES = {
    "run.py", "train.py", "pipeline.py", "index.py", "engine.py",
}

# ── System prompts ─────────────────────────────────────────────────────────────

_MAP_SYSTEM = (
    "You are a senior engineer mapping an unfamiliar codebase for the first time. "
    "You have the repo's README (what it claims to do) AND its entry-file imports "
    "(what the code actually does). Use both. "
    "Trace the main user-facing pipeline — the sequence of files that execute "
    "when the system does its primary job. "
    "NEVER list every file. Focus on the critical path only. "
    "Return ONLY valid JSON, no markdown, no explanation."
)

_INVESTIGATE_SYSTEM = (
    "You are a senior engineer doing a deep-dive into one component of a codebase. "
    "You know exactly where this component fits in the larger system. "
    "Answer four questions grounded in the code: "
    "WHY does this component exist (what breaks without it?), "
    "HOW does it connect to adjacent components, "
    "WHERE is the entry point a reader should start, "
    "WHAT non-obvious pattern or design decision makes this work. "
    "Also flag what the code CANNOT tell you — rationale, tradeoffs, why a library "
    "was chosen — these are things a new engineer must discover elsewhere. "
    "Class names, function names, file names are ENCOURAGED when they clarify design. "
    "Return ONLY valid JSON, no markdown, no explanation."
)

_SYNTHESIZE_SYSTEM = (
    "You are a senior engineer writing the guided tour you wished existed before "
    "reading this codebase. You have traced the full pipeline and investigated each stage. "
    "DEPENDENCY RULE: depends_on means 'cannot understand B without A' — NOT execution order. "
    "Most concepts are independent of each other; they share only concept 0 as a prerequisite. "
    "A chain A→B→C→D→E is almost always wrong. A fan-out from concept 0 is almost always right. "
    "NAME RULE: concept names MUST be technique/decision names — never artifacts. "
    "A BAD name identifies an artifact: any filename, any class name, any function name. "
    "A GOOD name identifies a decision or mechanism that exists regardless of what it is called: "
    "'Lazy Initialisation', 'Request-Response Lifecycle', 'Two-Phase Commit', 'Optimistic Locking'. "
    "Return ONLY valid JSON, no markdown, no explanation."
)

_VALIDATE_SYSTEM = (
    "You are a strict quality reviewer for codebase concept tours. "
    "Your ONLY job: check whether concept names are technique/decision names or artifact names. "
    "ARTIFACT (BAD): any file path, class name, function name, or identifier from the code. "
    "Signs of artifacts: ends in .py/.ts/.js, contains underscores, is CamelCase matching a class, "
    "matches a module name, or reads like a variable (e.g. 'health', 'config', 'utils', 'main'). "
    "TECHNIQUE (GOOD): a name that describes a design decision or mechanism — it could exist "
    "in any codebase with a similar purpose: 'Two-Phase Execution', 'Lazy Initialisation', "
    "'Connection Pooling', 'Request Fan-Out', 'Idempotent Retry'. "
    "Return ONLY valid JSON."
)


class TourAgent:
    """
    Four-phase agent: README → Map → Investigate × N → Synthesize.

    Each phase is a focused LLM call with tight context scope.
    Phase 0 (README) has no LLM call — it's pure data extraction from the index.
    """

    def __init__(self, store, gen):
        self._store = store
        self._gen   = gen
        self._chunk_cache: dict[str, list[dict]] = {}

    # ── Data access ────────────────────────────────────────────────────────────

    def _all_chunks(self, repo: str) -> list[dict]:
        """Load and cache all indexed chunks via Qdrant scroll (no vectors)."""
        if repo not in self._chunk_cache:
            raw = self._store.scroll_repo(repo)
            normalised = []
            for p in raw:
                normalised.append({
                    "name":         p.get("name") or p.get("function_name") or "",
                    "chunk_type":   p.get("chunk_type", "function"),
                    "filepath":     p.get("filepath", ""),
                    "text":         p.get("text", ""),
                    "calls":        p.get("calls", []),
                    "imports":      p.get("imports", []),
                    "base_classes": p.get("base_classes", []),
                    "start_line":   p.get("start_line", "?"),
                    "end_line":     p.get("end_line", "?"),
                })
            self._chunk_cache[repo] = normalised
        return self._chunk_cache[repo]

    def _file_chunks(self, repo: str, filepath: str) -> list[dict]:
        """All chunks for a specific file (flexible path matching)."""
        all_c = self._all_chunks(repo)
        fp_lower = filepath.lower()
        return [
            c for c in all_c
            if (c["filepath"].lower() == fp_lower
                or c["filepath"].lower().endswith("/" + fp_lower)
                or fp_lower in c["filepath"].lower())
        ]

    def _readme_chunks(self, repo: str) -> list[dict]:
        """
        README and top-level documentation chunks.

        Mirrors /init Phase 2: "read README before touching code."
        The README is the repo's authoritative statement of purpose — it tells
        us what the system is FOR, not just how it's implemented. Including it
        in Phase 1 prevents the LLM from confusing an internal utility file
        with the main user-facing pipeline.
        """
        all_c = self._all_chunks(repo)
        readme = []
        for c in all_c:
            fp = c["filepath"].lower()
            fname = fp.split("/")[-1]
            # README.md, readme.txt, README, docs/index.md, etc.
            if (fname.startswith("readme") or fname in ("index.md", "overview.md")
                    or "/docs/" in fp and fname.endswith(".md")):
                readme.append(c)
        # Prefer root-level README over nested ones; cap to avoid overloading context
        readme.sort(key=lambda c: (c["filepath"].count("/"), -len(c["text"])))
        return readme[:4]

    def _entry_module_chunks(self, repo: str) -> list[dict]:
        """
        Module-level chunks from likely entry-point files.

        Module chunks contain imports + file-level calls, revealing the call
        graph without exposing implementation details.

        Priority order: named entry files first (most likely top of the call stack),
        then other module chunks. This ensures Phase 1 sees the app's spine before
        peripheral files.
        """
        all_c  = self._all_chunks(repo)
        entries    = []
        non_entries = []
        seen_files: set[str] = set()
        for c in all_c:
            fp    = c["filepath"]
            fname = fp.split("/")[-1]
            is_entry  = fname in _ENTRY_NAMES
            is_module = c["chunk_type"] == "module"
            if (is_entry or is_module) and fp not in seen_files:
                seen_files.add(fp)
                if is_entry:
                    entries.append(c)
                else:
                    non_entries.append(c)
        # Entry files first so the LLM sees the true top of the call stack before
        # peripheral module chunks that may look like pipeline stages but aren't.
        return entries + non_entries

    def _related_chunks(self, repo: str, stage_file: str, stage_name: str,
                        primary_chunks: list[dict]) -> list[dict]:
        """
        "Start broad, narrow down" — from Explore agent pattern.

        When a file has only sparse chunks (<3), we expand the search:
        1. Files whose path contains keywords from the stage name
        2. Files that appear in the imports of the primary chunks

        This mirrors how a real engineer reads code: when one file is too thin,
        they follow the imports to adjacent files.
        Returns supplementary chunks (not including primary_chunks).
        """
        if len(primary_chunks) >= 3:
            return []   # Primary file is rich enough — no expansion needed

        all_c     = self._all_chunks(repo)
        primary_fps = {c["filepath"] for c in primary_chunks}

        # Strategy 1: keyword match in filepath
        keywords = [w.lower() for w in stage_name.split() if len(w) > 3]
        related: list[dict] = []
        seen: set[str] = set(primary_fps)

        for c in all_c:
            fp = c["filepath"].lower()
            if fp in seen:
                continue
            if any(kw in fp for kw in keywords):
                related.append(c)
                seen.add(fp)

        # Strategy 2: follow imports from primary chunks
        imported_names: set[str] = set()
        for c in primary_chunks:
            for imp in c.get("imports", []):
                # "from retrieval.store import X" → "store" as a keyword
                parts = imp.replace("from ", "").replace("import ", "").split(".")
                imported_names.update(p.strip() for p in parts if len(p) > 3)

        if imported_names:
            for c in all_c:
                fp = c["filepath"].lower()
                if fp in seen:
                    continue
                fname = fp.split("/")[-1].replace(".py", "").replace(".ts", "")
                if fname in imported_names or any(n in fp for n in imported_names):
                    related.append(c)
                    seen.add(fp)

        return related[:8]  # Cap to avoid bloating context

    def _list_file_paths(self, repo: str) -> list[str]:
        """Unique file paths in the index."""
        all_c = self._all_chunks(repo)
        seen: set[str] = set()
        paths = []
        for c in all_c:
            fp = c["filepath"]
            if fp and fp not in seen:
                seen.add(fp)
                paths.append(fp)
        return sorted(paths)

    # ── Formatters ────────────────────────────────────────────────────────────

    def _fmt_files_by_directory(self, all_files: list[str]) -> str:
        """
        Group files by directory for Phase 1's structural overview.

        A flat list of 80 files gives no hint about the repo's layer architecture.
        Grouped by directory, the LLM can immediately see "ingestion/ holds the
        data intake logic, retrieval/ holds search, services/ holds orchestration"
        — the same kind of structural signal that claude-code's get_architecture
        and list_directory tools provide.
        """
        from collections import defaultdict
        dirs: dict[str, list[str]] = defaultdict(list)
        for fp in all_files:
            if "/" in fp:
                parent, fname = fp.rsplit("/", 1)
            else:
                parent, fname = "(root)", fp
            dirs[parent].append(fname)

        lines = []
        for d in sorted(dirs):
            files_str = ", ".join(sorted(dirs[d]))
            lines.append(f"  {d}/: {files_str}")
        return "\n".join(lines)

    def _fmt_module_chunk(self, c: dict, max_len: int = 500) -> str:
        fp    = c["filepath"]
        imps  = c.get("imports", [])
        calls = c.get("calls", [])
        text  = c["text"].strip()[:max_len]
        lines = [f"── {fp}"]
        if imps:
            lines.append(f"   imports: {', '.join(imps[:8])}")
        if calls:
            lines.append(f"   calls:   {', '.join(calls[:8])}")
        if text:
            lines.append(f"   source:\n{text}")
        return "\n".join(lines)

    def _fmt_file_for_investigation(self, chunks: list[dict], max_len: int = 700,
                                     label: str = "") -> str:
        """Format chunks for the investigation phase, with optional section label."""
        parts = []
        if label:
            parts.append(f"=== {label} ===")
        for c in chunks[:12]:
            name   = c["name"] or "?"
            ctype  = c["chunk_type"]
            sl, el = c["start_line"], c["end_line"]
            text   = c["text"].strip()[:max_len]
            header = f"### {name} ({ctype}) — lines {sl}–{el}"
            parts.append(f"{header}\n{text}" if text else header)
        return "\n\n".join(parts)

    # ── Phase 1: Map ──────────────────────────────────────────────────────────

    def _phase_map(self, repo: str, readme_text: str) -> dict:
        """
        Identify the main pipeline and its stages.

        Strategy: README + directory structure as primary signals, service/library
        module chunks as secondary signal. Deliberately avoids bootstrap files
        (main.py, app.py) — they import ALL features equally and reveal nothing
        about the core pipeline.

        This mirrors the claude-code get_architecture approach: provide a curated
        structural overview (directory layers) rather than a raw import graph.
        """
        module_chunks = self._entry_module_chunks(repo)[:12]
        all_files     = self._list_file_paths(repo)

        # Directory-grouped file list — structural signal the LLM needs to see
        # which files belong to which layer (ingestion, retrieval, services, routers).
        # A flat list of 80 files gives no architectural signal.
        dir_text = self._fmt_files_by_directory(all_files)

        chunk_text = _token_budget(
            "\n\n".join(self._fmt_module_chunk(c) for c in module_chunks),
            max_tokens=1800,
        )

        readme_section = (
            f"What the repo claims to do (from README):\n{readme_text}\n\n"
            if readme_text else ""
        )

        prompt = f"""Repository: {repo}

{readme_section}Repository structure (files grouped by directory — shows the layer architecture):
{dir_text}

Module-level code from service/library files (imports and calls — secondary signal):
{chunk_text}

Your task: identify the CORE DATA TRANSFORMATION pipeline this repo implements.

Step 1 — Read the README summary above. It tells you WHAT the system does for a user.
The pipeline stages are the steps that make that happen — each one takes data in one
form and produces it in another.

Step 2 — Read the directory structure. Each directory name reveals its role:
  - Directories whose names suggest domain operations (e.g. ingestion, parsing, embedding,
    retrieval, inference, processing, generation, search) — these contain pipeline stages.
  - Directories whose names suggest wiring (e.g. routers, routes, middleware, handlers,
    controllers, api, endpoints, tests, utils, config) — these contain dispatch and
    infrastructure, not pipeline stages.

Step 3 — For each stage the README describes, find the FILE in a domain-operation directory
that implements it. Prefer files whose names match the operation (fetcher, chunker, embedder,
retrieval, generation, etc.) over files in routing or infrastructure directories.

Return ONLY this JSON:
{{
  "entry_file": "the service or library file that orchestrates the core pipeline (not a router, not a bootstrap file)",
  "readme_summary": "1-2 sentences: what the README says this repo does for the user",
  "pipeline_stages": [
    {{
      "name": "Short stage name (2-4 words, names the DATA TRANSFORMATION operation)",
      "file": "filepath/to/stage.py (must appear in the directory listing above)",
      "key_aspect": "One sentence: what input this stage receives and what output it produces"
    }}
  ]
}}

Rules:
- 3-6 stages covering the COMPLETE core pipeline, ordered by data flow
- Every stage file must be in a domain-operation directory, never in a routing/dispatch directory
- A stage transforms data: raw source → structured form, text → vectors, query → ranked results
- NEVER pick a router, HTTP handler, or application bootstrap file as a stage
- stage name must name the OPERATION (good: "AST Code Chunking"; bad: "chunker.py", "ChunkerClass")
- Every technique the README highlights as a key capability must appear as a stage
"""
        raw = self._gen.generate(_MAP_SYSTEM, prompt, temperature=0.0,
                                  json_mode=True, max_tokens=1024)
        try:
            result = _parse_json(raw)
            if "pipeline_stages" not in result or not result["pipeline_stages"]:
                raise ValueError("no stages")
            return result
        except Exception as e:
            print(f"TourAgent._phase_map failed: {e}  raw={raw[:300]}")
            fallback_stages = [
                {"name": c["name"] or c["filepath"].split("/")[-1],
                 "file": c["filepath"],
                 "key_aspect": ""}
                for c in module_chunks[:5]
            ]
            return {
                "entry_file": module_chunks[0]["filepath"] if module_chunks else "",
                "readme_summary": "",
                "pipeline_stages": fallback_stages,
            }

    # ── Phase 2: Investigate ──────────────────────────────────────────────────

    def _phase_investigate(self, repo: str, stage: dict, pipeline_context: str) -> dict:
        """
        Deep-dive into one pipeline stage using WHY/HOW/WHERE/WHAT framing.

        "Start broad, narrow down" (from Explore agent): when the primary file
        has sparse chunks, we expand to related files found via keyword matching
        and import following. This prevents thin investigations on files that
        delegate heavily to helpers.

        Also asks what the code CANNOT tell you — rationale and tradeoffs that
        live in commit history, design docs, or the author's head. Surfacing
        these gaps is a key insight from /init: "note what you could NOT figure
        out from code alone — these become interview questions."
        """
        stage_file   = stage.get("file", "")
        stage_name   = stage.get("name", "")
        stage_aspect = stage.get("key_aspect", "")

        # Primary: all chunks for the stage file
        primary = self._file_chunks(repo, stage_file)

        # Fallback if nothing found by filepath: search by function/class name
        if not primary:
            lname = stage_name.lower()
            all_c = self._all_chunks(repo)
            primary = [c for c in all_c if lname in c["name"].lower()][:6]

        # "Start broad, narrow down": expand to related files when primary is sparse
        related  = self._related_chunks(repo, stage_file, stage_name, primary)

        code_text = self._fmt_file_for_investigation(primary, label=stage_file)
        if related:
            related_text = self._fmt_file_for_investigation(
                related, label="Related files (imports / keyword match)")
            code_text = code_text + "\n\n" + related_text

        if not code_text.strip():
            return {
                "name": stage_name, "subtitle": stage_aspect,
                "insight": stage_aspect, "key_functions": [],
                "naive_rejected": "", "gaps": "",
            }

        # Guard: 12 primary × 700 chars + 8 related × 700 = up to 14 000 chars.
        # Cap at ~3 000 tokens (12 000 chars) so we stay within context budgets
        # on free-tier models with 8K context windows.
        code_text = _token_budget(code_text, max_tokens=3000)

        prompt = f"""Repository: {repo}
Stage: {stage_name}
Role in pipeline: {stage_aspect}

Full pipeline context:
{pipeline_context}

Code for this stage:
{code_text}

Answer five questions. Ground every answer in the code above.

1. WHY does this component exist? What breaks or degrades without it?
2. HOW does it connect to the rest of the pipeline? Input → transformation → output.
3. WHERE should a new engineer start reading? Name the entry-point class or function.
4. WHAT is the non-obvious pattern or decision? Name the technique and what it replaces.
5. GAPS: What important design rationale CANNOT be determined from this code alone?
   (e.g. why a particular library was chosen, what alternatives were considered,
   performance tradeoffs that aren't visible in the code)

Return ONLY this JSON:
{{
  "name": "3-5 words — the KEY TECHNIQUE or DESIGN DECISION (NEVER a file name, class name, or function name)",
  "subtitle": "One sentence: WHY this technique exists — the specific failure it prevents",
  "insight": "2-3 sentences: HOW it works, WHAT makes it non-obvious, naive alternative and its failure mode",
  "key_functions": ["entry_class_or_function", "other_key_function"],
  "naive_rejected": "One sentence: the simpler approach that would fail and why",
  "gaps": "One sentence: what important context is NOT visible in this code (or 'None' if fully self-explaining)"
}}

Rules:
- key_functions must be actual names visible in the code above
- insight must name a concrete failure mode with the naive approach
- Use actual class/function names when they clarify design
"""
        raw = self._gen.generate(_INVESTIGATE_SYSTEM, prompt, temperature=0.0,
                                  json_mode=True, max_tokens=900)
        try:
            result = _parse_json(raw)
            result.setdefault("name",          stage_name)
            result.setdefault("subtitle",      stage_aspect)
            result.setdefault("insight",       "")
            result.setdefault("key_functions", [])
            result.setdefault("naive_rejected","")
            result.setdefault("gaps",          "")
            return result
        except Exception as e:
            print(f"TourAgent._phase_investigate failed for {stage_name}: {e}")
            return {
                "name": stage_name, "subtitle": stage_aspect,
                "insight": stage_aspect, "key_functions": [],
                "naive_rejected": "", "gaps": "",
            }

    # ── Evaluator pass ───────────────────────────────────────────────────────

    def _validate_concepts(self, tour: dict, repo: str = "") -> dict:
        """
        Evaluator-optimizer loop: catch concept names that are artifacts.

        Implements the full evaluator-optimizer pattern:
          generate (Phase 3) → evaluate → if PASS return
                                         → if FAIL apply corrections, re-evaluate
        Max two evaluation rounds — enough to handle edge cases where the first
        correction round replaces one artifact name with another.

        Each round feeds previous corrections back into the prompt so the model
        knows what was already tried and rejected — this is the "memory
        accumulation" property from the cookbook's loop() implementation.

        After validation, persists bad→good corrections to disk (M4: negative
        example feedback). On the next run for the same repo, these previously
        bad names are injected into Phase 3's prompt so the model avoids
        generating them from the start.

        Non-fatal: any exception returns the unmodified tour.
        """
        # Dynamic round count: repos with a history of many artifact names get
        # an extra pass; fresh repos with no history need only one.
        # 0 past corrections → 1 round (no known problem patterns yet)
        # 1-5 corrections    → 2 rounds (some artifact tendency, default)
        # >5 corrections     → 3 rounds (repo consistently produces bad names)
        past_correction_count = len(_load_feedback(repo, self._store)) if repo else 0
        if past_correction_count == 0:
            MAX_ROUNDS = 1
        elif past_correction_count > 5:
            MAX_ROUNDS = 3
        else:
            MAX_ROUNDS = 2

        concepts = tour.get("concepts", [])
        if not concepts:
            return tour

        prior_corrections: list[str] = []   # accumulates across rounds for context
        # Collect bad→good pairs across all rounds for end-of-run persistence
        all_corrections: dict[str, str] = {}

        for round_num in range(MAX_ROUNDS):
            names_block = "\n".join(
                f'  id={c["id"]}: name="{c.get("name","")}" | subtitle="{c.get("subtitle","")}"'
                for c in tour.get("concepts", [])
            )

            # On round 2+, include what was already tried so the model doesn't
            # repeat the same correction that already failed evaluation.
            prior_context = ""
            if prior_corrections:
                prior_context = (
                    "\nPrevious correction attempts that were still flagged:\n"
                    + "\n".join(f"  - {p}" for p in prior_corrections)
                    + "\nDo NOT repeat these — find a different technique name.\n"
                )

            prompt = f"""You are evaluating concept names in a codebase tour.
A good concept name describes a TECHNIQUE or DESIGN DECISION that required tradeoffs.
{prior_context}
Concepts to review:
{names_block}

For each concept, apply these three tests:

TEST 1 — Is it a technique or decision name?
  PASS: names a mechanism, pattern, or tradeoff (e.g. "Two-Phase Execution", "Lazy Initialisation", "Connection Pooling")
  FAIL: is a file, class, function, or identifier (has underscores, ends in .py/.ts, matches a class/function name)

TEST 2 — Does it reveal an architectural insight?
  Ask: "Would a senior engineer need to understand this to work on the system's core value?"
  PASS: yes — removing this concept would leave a gap in understanding the system's key behaviour
  FAIL: no — it handles an edge case, manages a single external dependency, or is a standard utility found in every codebase

TEST 3 — Is the subtitle evidence of a real design decision?
  PASS: subtitle explains what breaks or degrades without this technique
  FAIL: subtitle describes loading config, excluding files, handling a single API's quota, or other narrow infrastructure

A concept that fails TEST 1 should be RENAMED.
A concept that fails TEST 2 or TEST 3 should be REMOVED — it is trivial infrastructure.

If ALL concepts pass all three tests, return: {{"status": "ok"}}

Otherwise return:
{{
  "status": "fixed",
  "concepts": [
    {{"id": 0, "action": "keep"}},
    {{"id": 1, "action": "rename", "name": "better technique name", "subtitle": "revised subtitle explaining the failure mode"}},
    {{"id": 2, "action": "remove"}}
  ]
}}

Include EVERY concept id. Use "action": "keep" for concepts that pass all tests.
"""
            try:
                raw = self._gen.generate(_VALIDATE_SYSTEM, prompt, temperature=0.0,
                                          json_mode=True, max_tokens=1200)
                result = _parse_json(raw)
            except Exception as e:
                print(f"TourAgent._validate_concepts round {round_num+1} failed (non-fatal): {e}")
                break

            if result.get("status") == "ok":
                break   # All names passed — no further rounds needed

            if result.get("status") == "fixed" and result.get("concepts"):
                corrections = {c["id"]: c for c in result["concepts"]}

                # Record renames — for context in the next round AND for
                # end-of-run persistence as negative examples.
                for c in tour.get("concepts", []):
                    corr = corrections.get(c["id"])
                    if corr and corr.get("action") == "rename" and corr.get("name"):
                        bad_name  = c.get("name", "")
                        good_name = corr["name"]
                        if good_name != bad_name:
                            prior_corrections.append(
                                f'id={c["id"]}: "{bad_name}" → "{good_name}"'
                            )
                            all_corrections[bad_name] = good_name

                # Build new concept list: keep or rename; omit those marked remove.
                # The evaluator now explicitly sets action="remove" for trivial
                # infrastructure, so removal intent is unambiguous — no longer
                # inferred from missing IDs which could silently swallow concepts
                # the evaluator simply forgot to include.
                new_concepts = []
                for c in tour["concepts"]:
                    corr   = corrections.get(c["id"], {})
                    action = corr.get("action", "keep")  # default keep if LLM omits
                    if action == "remove":
                        continue
                    if action == "rename":
                        if corr.get("name"):
                            c = {**c, "name": corr["name"]}
                        if corr.get("subtitle"):
                            c = {**c, "subtitle": corr["subtitle"]}
                    new_concepts.append(c)

                # Re-number ids and fix depends_on references after any removals.
                # When a concept that others depended on is removed (e.g. a health-
                # check trivial concept), its dependents would otherwise have a
                # dangling reference. We reroute those to concept 0 (pipeline
                # overview) — the safe universal prerequisite — rather than
                # silently dropping the dependency and leaving concepts with an
                # empty depends_on (which signals "no prerequisites needed").
                if len(new_concepts) != len(tour["concepts"]):
                    id_map    = {old["id"]: new_i for new_i, old in enumerate(new_concepts)}
                    new_id_0  = 0   # after re-numbering, concept 0 stays 0
                    for c in new_concepts:
                        old_id = c["id"]
                        c["id"] = id_map[old_id]
                        remapped = []
                        for d in c.get("depends_on", []):
                            if d in id_map:
                                new_d = id_map[d]
                                if new_d != c["id"]:
                                    remapped.append(new_d)
                            else:
                                # Dep was removed — reroute to concept 0 (the pipeline
                                # overview) so the concept still has a prerequisite.
                                if new_id_0 != c["id"] and new_id_0 not in remapped:
                                    remapped.append(new_id_0)
                        c["depends_on"] = remapped
                    if new_concepts:
                        new_concepts[0]["depends_on"] = []

                # Concept 0 (pipeline overview) must always be a root node —
                # it has no prerequisites by definition. Clear any deps that
                # the LLM may have set (or that survived the correction round).
                if new_concepts:
                    new_concepts[0]["depends_on"] = []

                tour = {**tour, "concepts": new_concepts}
                # Continue to next round to verify corrections passed

        # Persist bad→good corrections so future runs inject them as negative
        # examples into Phase 3 — the model won't generate these names again.
        if repo and all_corrections:
            _save_feedback(repo, all_corrections, self._store)
            print(f"TourAgent: saved {len(all_corrections)} negative example(s) for {repo}")

        return tour

    # ── Phase 3: Synthesize ───────────────────────────────────────────────────

    def _phase_synthesize(self, repo: str, pipeline_map: dict,
                          insights: list[dict]) -> dict:
        """
        Convert traced understanding to tour JSON.

        The LLM receives only structured findings — no raw code. Phase 3's
        only job is to assign the right type, file, and dependency relationships.
        Separation of concerns: no call does two hard things simultaneously.

        The README summary from Phase 1 is included so the tour's introductory
        summary aligns with the repo's stated purpose, not just its code structure.

        Past corrections (M4): if the evaluator has previously flagged bad concept
        names for this repo, they are injected here as a FORBIDDEN list so the model
        avoids generating the same artifact names before the evaluator even runs.
        """
        stages       = pipeline_map.get("pipeline_stages", [])
        entry        = pipeline_map.get("entry_file", "")
        readme_summ  = pipeline_map.get("readme_summary", "")

        stages_text = "\n".join(
            f"  {i+1}. {s['name']} — {s['file']}: {s.get('key_aspect','')}"
            for i, s in enumerate(stages)
        )
        insights_text = "\n\n".join(
            f"Stage {i+1} — {ins.get('name', '?')}\n"
            f"  subtitle:       {ins.get('subtitle', '')}\n"
            f"  insight:        {ins.get('insight', '')}\n"
            f"  key_functions:  {', '.join(ins.get('key_functions', []))}\n"
            f"  naive_rejected: {ins.get('naive_rejected', '')}\n"
            f"  gaps:           {ins.get('gaps', '')}"
            for i, ins in enumerate(insights)
        )

        readme_section = (
            f"What the README says this repo does:\n{readme_summ}\n\n"
            if readme_summ else ""
        )

        prompt = f"""Repository: {repo}
Entry file: {entry}

{readme_section}Pipeline traced in order:
{stages_text}

Per-stage findings (use these verbatim — do NOT paraphrase):
{insights_text}

Convert this into a concept tour JSON.

═══ DEPENDENCY RULE (CRITICAL) ═══
depends_on = "cannot understand B without first understanding A" — NOT execution order.

Test for each concept: "Can an engineer understand this WITHOUT knowing the others?"
  Yes → depends_on: [0]     (pipeline overview is the only prerequisite)
  No  → depends_on: [specific concept id they must know first]

WRONG (chain):   1→2→3→4→5→6   (almost never true for concepts in a codebase)
RIGHT (fan-out): 1,2,3,4,5 each depend on 0 only — they are parallel concepts

Typical structure for 7 concepts:
  0: pipeline overview (no deps)
  1,2,3,4,5: core independent concepts (each depends_on: [0])
  6: one concept with a genuine prerequisite (depends_on: [1] or [2])

═══ FORMAT ═══
Return ONLY this JSON:
{{
  "summary": "2 sentences: (1) what the user can DO with this repo and the key technique that enables it. (2) the single architectural decision that shapes everything else. Ground this in the README summary above.",
  "entry_point": "{entry}",
  "concepts": [
    {{
      "id": 0,
      "name": "End-to-end pipeline name (3-5 words)",
      "subtitle": "What this pipeline does for the user",
      "file": "{entry}",
      "type": "module",
      "description": "2-3 sentences: what enters, how each stage transforms it, what the user gets. Name the key files.",
      "key_items": ["entry_function", "other_function"],
      "depends_on": [],
      "reading_order": 1,
      "ask": "Where does the data flow change if one of the pipeline stages is removed?"
    }},
    {{
      "id": 1,
      "name": "Use exact 'name' from stage 1 findings",
      "subtitle": "Use exact 'subtitle' from stage 1 findings",
      "file": "file from stage 1",
      "type": "class|function|module|algorithm",
      "description": "Use exact 'insight' from stage 1 findings",
      "key_items": ["use exact key_functions from stage 1 findings"],
      "depends_on": [0],
      "reading_order": 2,
      "ask": "A specific question naming THIS concept's mechanism — answerable only by someone who understands this specific technique, e.g. 'What would fail if this component processed items sequentially instead of in parallel?' or 'Why does this layer need to complete before the next stage can begin?'"
    }}
  ]
}}

Rules:
- 6-8 concepts total (id=0 is pipeline overview, id=1..N are stage insights)
- Use the EXACT name/subtitle/insight/key_functions from findings — do not paraphrase
- All concepts except id=0 must have depends_on non-empty
- Most should have depends_on: [0] — add deeper deps only when genuinely required
- reading_order: sequential integers from 1
- type: exactly one of class, function, module, algorithm
- NAME RULE (CRITICAL): concept names MUST describe a technique or design decision.
  FORBIDDEN: any filename, any class name, any function name, any identifier with underscores
  A valid name could exist in any codebase with similar purpose — it names a MECHANISM, not an artifact
{_synthesize_negative_block(repo, self._store)}- ask RULE: each ask must be a SPECIFIC question about THIS concept's key mechanism and failure mode.
  BAD: "Why was the naive approach rejected?" (could apply to any concept anywhere — zero information)
  GOOD: name the mechanism, name the thing that would break: "What breaks if you remove X from Y?"
  Each ask must be answerable ONLY by someone who has read this specific concept — never vague
"""
        raw = self._gen.generate(_SYNTHESIZE_SYSTEM, prompt, temperature=0.0,
                                  json_mode=True, max_tokens=8192)
        try:
            tour = _parse_json(raw)
        except Exception as e:
            print(f"TourAgent._phase_synthesize failed: {e}\nRaw: {raw[:600]}")
            raise ValueError(f"Synthesis failed: {e}")

        # Sanitise depends_on — remove self-references and invalid IDs
        if "concepts" in tour:
            valid_ids = {c["id"] for c in tour["concepts"]}
            for c in tour["concepts"]:
                c["depends_on"] = [
                    d for d in c.get("depends_on", [])
                    if d in valid_ids and d != c["id"]
                ]
        return tour

    # ── Main entry point ──────────────────────────────────────────────────────

    def build(self, repo: str) -> Generator[dict, None, None]:
        """
        Build a concept tour yielding SSE-compatible progress events.

        Phase 0 (README) has no LLM call — pure data extraction.
        Phases 1-3 each make one focused LLM call.
        """
        yield {"stage": "mapping", "progress": 0.05, "message": "Loading repository…"}

        all_chunks = self._all_chunks(repo)
        if not all_chunks:
            yield {"stage": "error", "progress": 1.0,
                   "error": "No indexed chunks found. Try re-ingesting."}
            return

        n_files  = len({c["filepath"] for c in all_chunks if c["filepath"]})
        n_chunks = len(all_chunks)

        yield {
            "stage": "mapping", "progress": 0.08,
            "message": f"Found {n_chunks} chunks across {n_files} files",
            "trace": {"type": "info",
                      "text": f"{n_chunks} chunks across {n_files} files"},
        }

        # ── Phase 0: README ───────────────────────────────────────────────────
        # Read the repo's stated purpose before touching code.
        # Mirrors /init Phase 2: "read README before mapping."
        readme_chunks = self._readme_chunks(repo)
        readme_text   = ""
        if readme_chunks:
            # Take the first 1500 chars of README — enough for the overview,
            # not so much that it drowns the entry-file signal in Phase 1.
            readme_text = "\n\n".join(c["text"][:600] for c in readme_chunks[:3]).strip()[:1500]
            yield {
                "stage": "mapping", "progress": 0.12,
                "message": "Read README…",
                "trace": {"type": "file",
                          "text": readme_chunks[0]["filepath"],
                          "step": 0, "total": 0},
            }
        else:
            yield {
                "stage": "mapping", "progress": 0.12,
                "message": "No README found — mapping from code only",
                "trace": {"type": "info", "text": "No README in index"},
            }

        # ── Phase 1: Map ──────────────────────────────────────────────────────
        yield {
            "stage": "mapping", "progress": 0.15,
            "message": "Mapping pipeline from README + imports…",
            "trace": {"type": "thinking",
                      "text": "Combining stated purpose with actual call graph…"},
        }

        try:
            pipeline_map = self._phase_map(repo, readme_text)
        except Exception as e:
            yield {"stage": "error", "progress": 1.0,
                   "error": f"Pipeline mapping failed: {e}"}
            return

        stages = pipeline_map.get("pipeline_stages", [])
        entry  = pipeline_map.get("entry_file", "")

        yield {
            "stage": "mapping", "progress": 0.25,
            "message": f"Found {len(stages)}-stage pipeline",
            "trace": {"type": "found",
                      "text": f"Entry: {entry}",
                      "stages": [s["name"] for s in stages]},
        }

        # ── Phase 2: Investigate all stages in parallel ───────────────────────
        # All stage investigations are independent — each reads its own file
        # chunks and uses only the shared pipeline_context string.
        # 3 workers: ~3x speedup while staying within free-tier rate limits.
        # (All-at-once parallelism hits 15 RPM Gemini on 6+ stage pipelines.)
        pipeline_context = "\n".join(
            f"  {i+1}. {s['name']}: {s.get('key_aspect', '')}"
            for i, s in enumerate(stages)
        )
        n_stages   = len(stages)
        base_prog  = 0.25
        stage_step = 0.55 / max(n_stages, 1)
        _INV_WORKERS = 3

        # Emit "start" events for all stages before the pool runs — the pool
        # runs synchronously from the generator's perspective, so we can't
        # yield from inside the threads. Emit start events now, results after.
        for i, stage in enumerate(stages):
            primary_count = len(self._file_chunks(repo, stage.get("file", "")))
            search_mode   = "broad (expanding to related files)" if primary_count < 3 else "deep"
            yield {
                "stage": "investigating", "progress": base_prog + i * stage_step * 0.3,
                "message": f"Queued: {stage['name']} ({i+1}/{n_stages})",
                "trace": {"type": "file",
                          "text": f"{stage.get('file', '')} — {search_mode}",
                          "step": i + 1,
                          "total": n_stages},
            }

        yield {
            "stage": "investigating", "progress": base_prog + 0.05,
            "message": f"Investigating {n_stages} stages in parallel…",
            "trace": {"type": "thinking",
                      "text": f"Running {n_stages} investigations with {_INV_WORKERS} workers…"},
        }

        # Run investigations in parallel; collect results keyed by stage index.
        insights: list[dict | None] = [None] * n_stages
        with ThreadPoolExecutor(max_workers=_INV_WORKERS) as pool:
            future_to_idx = {
                pool.submit(self._phase_investigate, repo, stage, pipeline_context): i
                for i, stage in enumerate(stages)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    insights[i] = future.result()
                except Exception as e:
                    print(f"TourAgent: investigation failed for stage {i}: {e}")
                    insights[i] = {
                        "name": stages[i].get("name", ""),
                        "subtitle": stages[i].get("key_aspect", ""),
                        "insight": "", "key_functions": [],
                        "naive_rejected": "", "gaps": "",
                    }

        # Emit "found" events in pipeline order after all are complete
        for i, (stage, insight) in enumerate(zip(stages, insights)):
            prog      = base_prog + (i + 1) * stage_step
            gaps_text  = insight.get("gaps", "")
            trace_text = (insight.get("insight") or "")[:120]
            if gaps_text and gaps_text.lower() != "none":
                trace_text += f" | gap: {gaps_text[:60]}"
            yield {
                "stage": "investigating", "progress": prog,
                "message": f"Found: {insight.get('name', stage['name'])}",
                "trace": {"type": "finding",
                          "name": insight.get("name", ""),
                          "text": trace_text},
            }

        # ── Phase 3: Synthesize ───────────────────────────────────────────────
        yield {
            "stage": "synthesizing", "progress": 0.82,
            "message": "Synthesizing concept tour…",
            "trace": {"type": "thinking",
                      "text": "Building fan-out dependency graph from traced insights…"},
        }

        try:
            tour = self._phase_synthesize(repo, pipeline_map, insights)
        except Exception as e:
            yield {"stage": "error", "progress": 1.0,
                   "error": f"Synthesis failed: {e}"}
            return

        # ── Evaluator pass ────────────────────────────────────────────────────
        # A targeted quality check: are concept names techniques or artifacts?
        # Runs one cheap LLM call on just the names — far less context than a
        # full re-synthesis. Non-fatal: if it fails, the original tour is used.
        yield {
            "stage": "synthesizing", "progress": 0.93,
            "message": "Validating concept names…",
            "trace": {"type": "thinking",
                      "text": "Checking concepts are technique names, not file/class names…"},
        }
        tour = self._validate_concepts(tour, repo=repo)

        yield {"stage": "done", "progress": 1.0, **tour}


# ── JSON parser ────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Parse JSON from an LLM response, tolerating markdown fences and trailing text."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        obj, _ = _json.JSONDecoder().raw_decode(cleaned)
        return obj
    except _json.JSONDecodeError:
        pass
    match = _re.search(r'\{.*\}', cleaned, _re.DOTALL)
    if match:
        try:
            obj, _ = _json.JSONDecoder().raw_decode(match.group(0))
            return obj
        except _json.JSONDecodeError:
            pass
    raise ValueError(f"No JSON in response: {raw[:200]}")

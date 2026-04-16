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

# ── Phase 1 stage-name validation ─────────────────────────────────────────────
# Phase 1 asks the LLM for "design decisions" but the LLM sometimes returns code
# identifiers (class names, method names, filenames) as stage names. These are
# artifacts, not techniques. We filter them out in Python before Phase 2 runs —
# no LLM call should be wasted investigating "AgentService._build_initial_messages".
#
# Rules derived from what artifacts look like syntactically:
import re as _re

_ARTIFACT_EXTENSION = _re.compile(r'\.\w{1,5}$')          # e.g. "diagrams.py"
_ARTIFACT_SNAKE     = _re.compile(r'[a-z]_[a-z]')         # snake_case: "build_initial"
_ARTIFACT_DOTREF    = _re.compile(r'\w\.\w')               # dot-reference: "Agent.method"
_ARTIFACT_ALLCAPS   = _re.compile(r'^[A-Z_]{3,}$')         # ALL_CAPS constant
# Single generic words that describe infrastructure, not design decisions
_GENERIC_WORDS = {
    'health', 'config', 'utils', 'main', 'app', 'server', 'index', 'api',
    'init', 'setup', 'core', 'base', 'test', 'run', 'build', 'client',
    'router', 'handler', 'middleware', 'logger', 'helper', 'model', 'schema',
}

def _is_artifact_stage_name(name: str) -> bool:
    """Return True if the stage name looks like a code artifact, not a technique.

    WHY: Phase 1 identifies design decisions but the LLM sometimes copies code
    identifiers (method names, filenames, class references) from the module chunks.
    These produce empty Phase 2 results and hollow concept cards. Better to skip
    them before Phase 2 wastes a LLM call than to catch them later.
    """
    n = name.strip()
    if not n:
        return True
    if _ARTIFACT_EXTENSION.search(n): return True  # "diagrams.py", "agent.go"
    if _ARTIFACT_SNAKE.search(n):     return True  # "build_initial_messages"
    if _ARTIFACT_DOTREF.search(n):    return True  # "AgentService._build"
    if _ARTIFACT_ALLCAPS.match(n):    return True  # "MAX_TOKENS"
    # Single generic word (infrastructure, not a decision)
    words = n.split()
    if len(words) == 1 and n.lower() in _GENERIC_WORDS:
        return True
    return False


# ── System prompts ─────────────────────────────────────────────────────────────

_MAP_SYSTEM = (
    "You are a senior engineer mapping an unfamiliar codebase for the first time. "
    "You have the repo's README (what it claims to do) AND the module-level imports "
    "and signatures of every non-bootstrap file (what the code actually does). Use both. "
    "Trace the main pipeline — the sequence of files that execute the system's primary job. "
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
    "ASK RULE: each 'ask' MUST name a specific function from key_items and describe a concrete "
    "failure mode. Generic asks ('Why was X rejected?', 'What would happen without this?') are "
    "forbidden — they convey no information and waste the reader's time. "
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

    def _manifest_chunks(self, repo: str) -> list[dict]:
        """
        Project manifest files — the most information-dense source for Phase 1.

        Manifest files (package.json, Cargo.toml, pyproject.toml, go.mod, etc.)
        declare dependencies and entry points. They reveal the tech stack and
        project type for ANY repo without relying on directory name heuristics:
          - fastapi + qdrant-client → web API + vector search
          - torch + transformers    → ML training pipeline
          - llvm-sys                → compiler
          - no framework deps       → pure library (like micrograd)

        This is the same principle as claude-code /init Phase 2: read manifest
        files before touching code.
        """
        # Filenames that are universally recognised project manifests
        MANIFEST_NAMES = {
            "package.json", "cargo.toml", "pyproject.toml", "setup.py",
            "setup.cfg", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts",
            "gemfile", "composer.json", "mix.exs", "project.clj", "dune-project",
            "requirements.txt", "pipfile", "makefile",
        }
        all_c = self._all_chunks(repo)
        manifests = []
        seen: set[str] = set()
        for c in all_c:
            fname = c["filepath"].split("/")[-1].lower()
            # Root-level manifests only (depth 0 or 1) — nested ones are
            # subpackage manifests and would confuse the pipeline mapping.
            depth = c["filepath"].count("/")
            if fname in MANIFEST_NAMES and depth <= 1 and c["filepath"] not in seen:
                seen.add(c["filepath"])
                manifests.append(c)
        # Prefer root-level; within same depth prefer richer content
        manifests.sort(key=lambda c: (c["filepath"].count("/"), -len(c["text"])))
        return manifests[:6]

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

    # ── Phase 1 (Agentic): ReAct exploration loop ─────────────────────────────

    # WHY AGENTIC: a one-shot static snapshot gives the LLM ~14 random module
    # chunks and asks it to identify design decisions. The LLM often picks code
    # identifiers (method names, filenames) because those are the most prominent
    # things visible in the chunks — not because they're design decisions.
    #
    # An agentic loop gives the LLM TOOLS and lets it decide what to read.
    # Like a developer joining a new project, it starts at the top (README +
    # directory listing), narrows in on interesting files, and stops when it
    # understands the architecture — not when a fixed token budget runs out.
    #
    # The generator yields trace events so the UI can show the ReAct loop live:
    #   THINK → TOOL → RESULT → THINK → TOOL → RESULT → ... → DONE
    # This doubles as an educational demonstration of how agentic AI works.

    # Tools exposed to the ReAct agent — same capabilities as our MCP server
    # (backend/mcp_server.py) but called directly rather than over the wire.
    # The MCP server already defines: list_files, read_file, search_symbol,
    # find_callers, trace_calls, search_code.  We reuse that same logic here
    # so the Phase 1 agent has exactly the same power as a Claude Code session
    # connected to our MCP server.

    _AGENTIC_MAP_SYSTEM = (
        "You are a senior engineer exploring an unfamiliar codebase to identify its key "
        "ARCHITECTURAL DECISIONS — the non-obvious choices where a simpler alternative "
        "was deliberately rejected.\n\n"
        "TOOLS — call exactly one per turn:\n"
        "  list_files(path)            list files/dirs at a path (\"\" = repo root)\n"
        "  read_file(filepath)         read a source file (imports, classes, functions)\n"
        "  search_symbol(name)         find a class or function definition by exact name\n"
        "  find_callers(name)          find all call sites of a function/class\n"
        "  trace_calls(name)           trace the call graph from an entry point\n\n"
        "FORMAT — output exactly two lines per turn:\n"
        "  THINK: [one sentence: what you learned and what to investigate next]\n"
        "  TOOL: tool_name(\"argument\")\n\n"
        "  OR when you have identified 4-6 decisions:\n"
        "  THINK: [why you have enough information now]\n"
        "  DONE: {\"entry_file\":\"...\",\"readme_summary\":\"...\","
        "\"pipeline_stages\":[{\"name\":\"...\",\"file\":\"...\",\"key_aspect\":\"...\"}]}\n\n"
        "EXPLORATION STRATEGY:\n"
        "  1. list_files(\"\") — see top-level repo structure, identify ALL major directories\n"
        "  2. read_file() key manifests (package.json, pyproject.toml, go.mod, Cargo.toml)\n"
        "  3. read_file() at least one file from each major directory you found in step 1\n"
        "  4. search_symbol() / find_callers() to trace how key components connect\n"
        "  5. DONE only after reading files from multiple directories — never after one\n\n"
        "DIVERSITY RULE (critical):\n"
        "  Each stage in pipeline_stages MUST come from a DIFFERENT file.\n"
        "  If you find 2+ interesting things in one file, pick only the most important one —\n"
        "  then keep exploring OTHER directories before calling DONE.\n"
        "  A DONE with multiple stages pointing to the same file will be rejected.\n\n"
        "STAGE NAME RULES (critical — every name is checked):\n"
        "  GOOD: names a technique, algorithm, or tradeoff (e.g. 'Lazy Evaluation Cache',\n"
        "        'Hybrid Sparse-Dense Retrieval', 'Progressive Context Expansion')\n"
        "  BAD: any filename, class name, function name, or identifier with underscores\n"
        "  4-6 stages only — core decisions, skip infrastructure (routing, config, health)\n"
        "  key_aspect: what simpler approach this replaces and the concrete cost of that\n\n"
        "Return ONLY valid JSON in the DONE: line — no markdown fences, no explanation."
    )

    def _agentic_list_files(self, repo: str, path: str) -> str:
        """GitHub API directory listing — same as mcp_server.list_files."""
        import requests as _req
        from backend.config import settings
        path = path.strip("/").strip()
        owner, name = repo.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{name}/contents/{path}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if settings.github_token:
            headers["Authorization"] = f"token {settings.github_token}"
        try:
            resp = _req.get(url, headers=headers, timeout=15)
            if resp.status_code == 404:
                return f"Path not found: '{path}' in {repo}"
            resp.raise_for_status()
        except Exception as e:
            return f"GitHub fetch failed: {e}"
        entries = resp.json()
        if not isinstance(entries, list):
            return f"'{path}' is a file — use read_file to read it."
        dirs  = sorted([e["name"] + "/" for e in entries if e["type"] == "dir"])
        files = sorted([
            f"{e['name']}  ({e.get('size',0)//1024}KB)" if e.get("size",0)>=1024
            else f"{e['name']}  ({e.get('size',0)}B)"
            for e in entries if e["type"] == "file"
        ])
        return f"# {repo}/{path or ''}\n" + "\n".join(dirs + files)

    def _agentic_read_file(self, repo: str, filepath: str) -> str:
        """GitHub API file read, truncated to ~600 tokens — same as mcp_server.read_file."""
        import requests as _req
        from backend.config import settings
        filepath = filepath.strip()
        owner, name = repo.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{name}/contents/{filepath}"
        headers = {"Accept": "application/vnd.github.v3.raw"}
        if settings.github_token:
            headers["Authorization"] = f"token {settings.github_token}"
        try:
            resp = _req.get(url, headers=headers, timeout=15)
            if resp.status_code == 404:
                return f"File not found: {filepath}"
            resp.raise_for_status()
        except Exception as e:
            return f"GitHub fetch failed: {e}"
        lines = resp.text.splitlines()
        total = len(lines)
        # Return up to 120 lines — enough to see imports, class defs, top-level functions.
        # Cap keeps transcript size manageable across 8 rounds.
        preview = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[:120]))
        suffix  = f"\n… ({total - 120} more lines)" if total > 120 else ""
        return f"# {repo} — {filepath}  ({total} lines)\n\n{preview}{suffix}"

    def _agentic_search_symbol(self, repo: str, symbol_name: str) -> str:
        """Find a class or function definition by name — wraps store.find_symbol."""
        matches = self._store.find_symbol(symbol_name, repo=repo)
        if not matches:
            return f"No definition found for '{symbol_name}'. Try search_symbol with the exact name."
        parts = []
        for i, c in enumerate(matches[:4], 1):
            loc = f"{c.get('filepath','?')} L{c.get('start_line','?')}–{c.get('end_line','?')}"
            parts.append(f"[{i}] {loc}\n{c.get('text','')[:400]}")
        return f"Definitions of '{symbol_name}':\n\n" + "\n\n".join(parts)

    def _agentic_find_callers(self, repo: str, function_name: str) -> str:
        """Find all call sites — wraps store.find_callers."""
        callers = self._store.find_callers(function_name, repo=repo)
        if not callers:
            return f"No call sites found for '{function_name}'."
        parts = []
        for i, c in enumerate(callers[:6], 1):
            loc = f"{c.get('filepath','?')} — {c.get('name','?')} L{c.get('start_line','?')}"
            parts.append(f"[{i}] {loc}\n{c.get('text','')[:300]}")
        return f"Call sites of '{function_name}' ({len(callers)} found):\n\n" + "\n\n".join(parts)

    def _agentic_trace_calls(self, repo: str, symbol_name: str) -> str:
        """Trace call graph from an entry point — same logic as mcp_server.trace_calls."""
        visited: set[str] = set()
        lines: list[str]  = [f"# Call trace from `{symbol_name}`\n"]

        def _walk(name: str, depth: int, prefix: str) -> None:
            if depth > 3 or name in visited:
                return
            visited.add(name)
            chunks = self._store.find_symbol(name, repo=repo)
            if not chunks:
                return
            c   = chunks[0]
            loc = f"{c.get('filepath','?')} L{c.get('start_line','?')}"
            lines.append(f"{prefix}→ {name}()  `{loc}`")
            for callee in (c.get("calls") or [])[:5]:
                _walk(callee, depth + 1, prefix + "  ")

        _walk(symbol_name, 0, "")
        return "\n".join(lines) if len(lines) > 1 else f"Symbol '{symbol_name}' not found in index."

    def _phase_map_agentic(self, repo: str, readme_text: str):
        """Generator: ReAct exploration loop for Phase 1.

        Yields dict trace events as it runs (forwarded to the UI live-log panel),
        then yields a final {"type": "result", "data": pipeline_map_dict} when done.

        Falls back to static _phase_map() on parse failure or exhausted rounds.
        """
        manifest_chunks = self._manifest_chunks(repo)
        manifest_text = _token_budget(
            "\n\n".join(
                f"── {c['filepath']}\n{c['text'].strip()[:500]}"
                for c in manifest_chunks
            ),
            max_tokens=500,
        )

        # Seed the transcript with the two highest-signal sources: README + manifests.
        # The agent decides where to go from here.
        transcript = f"Repository: {repo}\n\n"
        if readme_text:
            transcript += f"README:\n{readme_text}\n\n"
        if manifest_text.strip():
            transcript += f"Manifest files (dependencies / entry points):\n{manifest_text}\n\n"
        transcript += "Begin exploration. Start with list_files(\"\") to see the top-level repo structure.\n"

        max_rounds = 6  # 6 rounds × ~700 tokens ≈ 4200 tokens for Phase 1
        for round_n in range(max_rounds):
            raw = self._gen.generate(
                self._AGENTIC_MAP_SYSTEM, transcript,
                temperature=0.0, max_tokens=700,  # Gemma 4 needs ~700 for verbose THINK+TOOL
            )

            # Parse THINK + TOOL or DONE from the LLM's response
            think_m = _re.search(r'THINK:\s*(.+?)(?:\n|$)', raw, _re.IGNORECASE | _re.DOTALL)
            tool_m  = _re.search(r'TOOL:\s*(\w+)\(\s*"?([^")\n]*)"?\s*\)', raw, _re.IGNORECASE)
            done_m  = _re.search(r'DONE:\s*(\{.+)', raw, _re.DOTALL)

            think_text = (think_m.group(1).strip() if think_m else raw[:120].strip())

            # ── DONE ──────────────────────────────────────────────────────────
            if done_m:
                try:
                    result = _parse_json(done_m.group(1))
                    if result.get("pipeline_stages"):
                        yield {"type": "thinking",
                               "text": f"✓ Done in {round_n + 1} round(s): {think_text}"}
                        yield {"type": "result", "data": result}
                        return
                except Exception:
                    pass  # malformed JSON — keep going

            # ── TOOL CALL ─────────────────────────────────────────────────────
            if tool_m:
                tool_name = tool_m.group(1).lower().replace("-", "_")
                tool_arg  = tool_m.group(2).strip().strip('"').strip("'")

                if tool_name == "list_files":
                    tool_result = self._agentic_list_files(repo, tool_arg)
                    display     = f"list_files(\"{tool_arg}\")"
                elif tool_name == "read_file":
                    tool_result = self._agentic_read_file(repo, tool_arg)
                    display     = f"read_file(\"{tool_arg}\")"
                elif tool_name == "search_symbol":
                    tool_result = self._agentic_search_symbol(repo, tool_arg)
                    display     = f"search_symbol(\"{tool_arg}\")"
                elif tool_name == "find_callers":
                    tool_result = self._agentic_find_callers(repo, tool_arg)
                    display     = f"find_callers(\"{tool_arg}\")"
                elif tool_name == "trace_calls":
                    tool_result = self._agentic_trace_calls(repo, tool_arg)
                    display     = f"trace_calls(\"{tool_arg}\")"
                else:
                    tool_result = f"(unknown tool '{tool_name}' — use: list_files, read_file, search_symbol, find_callers, trace_calls)"
                    display     = tool_name

                # Emit a trace event for the UI live-log panel
                yield {"type": "react",
                       "think": think_text,
                       "tool":  display,
                       "text":  f"THINK: {think_text} → {display}"}

                # Truncate tool output so the transcript doesn't balloon
                tool_result = _token_budget(tool_result, max_tokens=300)
                transcript += (
                    f"\nTHINK: {think_text}\n"
                    f"TOOL: {display}\n"
                    f"RESULT:\n{tool_result}\n"
                )
            else:
                # LLM output couldn't be parsed — nudge it
                transcript += f"\n[No valid action found in round {round_n + 1}. Output TOOL: or DONE:]\n"
                yield {"type": "thinking", "text": f"Round {round_n + 1}: retrying parse…"}

        # ── Exhausted rounds — force final output ──────────────────────────────
        yield {"type": "thinking", "text": "Reached round limit — requesting final output…"}
        transcript += "\nROUND LIMIT REACHED. Output DONE: now with what you have found.\n"
        raw = self._gen.generate(
            self._AGENTIC_MAP_SYSTEM, transcript,
            temperature=0.0, max_tokens=700,  # pipeline_stages JSON — 700 is enough, generates faster
        )
        done_m = _re.search(r'DONE:\s*(\{.+)', raw, _re.DOTALL)
        try:
            result = _parse_json(done_m.group(1) if done_m else raw)
            if result.get("pipeline_stages"):
                yield {"type": "result", "data": result}
                return
        except Exception:
            pass

        # Complete fallback: static snapshot Phase 1
        yield {"type": "thinking", "text": "Agentic loop failed — falling back to static Phase 1"}
        try:
            result = self._phase_map(repo, readme_text)
            yield {"type": "result", "data": result}
        except Exception as e:
            yield {"type": "result", "data": {}}

    # ── Phase 1: Map (static fallback) ────────────────────────────────────────

    def _phase_map(self, repo: str, readme_text: str) -> dict:
        """
        Identify the main pipeline and its stages.

        Primary signal: the README (what the repo claims to do) + module-level
        imports and function signatures of every non-bootstrap file.

        A file's imports are a universal, language-agnostic signal of its role:
          - imports domain/math/IO libraries → does real work (pipeline stage)
          - imports web-framework routing primitives → dispatch layer
          - imports everything from sibling files → orchestration or bootstrap

        No directory-name heuristics — those break on any codebase that uses
        non-conventional naming (micrograd, game engines, compilers, etc.).
        """
        # Manifest files first — they declare dependencies and entry points,
        # which reveal the project type without any domain heuristics.
        # (Same principle as claude-code /init Phase 2: read manifest files
        # before touching code.)
        manifest_chunks = self._manifest_chunks(repo)
        module_chunks   = self._entry_module_chunks(repo)[:14]
        all_files       = self._list_file_paths(repo)

        dir_text = self._fmt_files_by_directory(all_files)

        manifest_text = _token_budget(
            "\n\n".join(
                f"── {c['filepath']}\n{c['text'].strip()[:600]}"
                for c in manifest_chunks
            ),
            max_tokens=800,
        )
        chunk_text = _token_budget(
            "\n\n".join(self._fmt_module_chunk(c) for c in module_chunks),
            max_tokens=1800,
        )

        readme_section = (
            f"README:\n{readme_text}\n\n"
            if readme_text else ""
        )
        manifest_section = (
            f"Project manifest files (dependencies and entry points):\n{manifest_text}\n\n"
            if manifest_text.strip() else ""
        )

        prompt = f"""Repository: {repo}

{manifest_section}{readme_section}File structure:
{dir_text}

Module-level imports and signatures:
{chunk_text}

Task: identify the KEY DESIGN DECISIONS in this codebase.

A design decision is a non-obvious choice — there was a simpler alternative
that was deliberately rejected. Design decisions are the concepts a new engineer
must understand to work on this system effectively.

How to find them:
1. The README often names them explicitly (look for technique names, library
   choices framed as decisions, or "why X instead of Y" language).
2. The manifest dependencies reveal which libraries were chosen — each choice
   of a non-trivial library is a decision.
3. The module imports reveal which file implements each technique.

For each decision, find the single file where it is most directly implemented.
Infrastructure files (database wrappers, config loaders, HTTP plumbing) are
not design decisions — skip them.

Return ONLY this JSON:
{{
  "entry_file": "the file that orchestrates the core system",
  "readme_summary": "1-2 sentences: what the README says this repo does",
  "pipeline_stages": [
    {{
      "name": "The technique or decision name (3-5 words)",
      "file": "path/to/file (must appear in the file structure above)",
      "key_aspect": "One sentence: what simpler approach this replaces and why that matters"
    }}
  ]
}}

Rules:
- 4-6 decisions — enough to cover the system's core value, not every file
- Every file must appear in the file structure above
- Each decision must have an identifiable simpler alternative that was rejected
- Skip pure infrastructure: storage wrappers, config loaders, HTTP routers,
  dependency injection — these have no interesting decision in the code itself
- Decision names describe the technique or choice, never a filename or class
- Prioritise decisions the README explicitly names — those are the ones the
  authors considered important enough to document
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

    # ── Phase 2: Agentic Investigate ─────────────────────────────────────────
    #
    # Why agentic: one-shot Phase 2 reads chunks from a single file and asks
    # WHY/HOW/WHERE/WHAT. If the key algorithm spans multiple files, or the most
    # important function is three call layers deep, the one-shot call misses it.
    #
    # The ReAct loop lets the agent start at the primary file, trace callers to
    # see how it's used, trace calls to follow the implementation depth, and
    # search for the exact function that implements the design decision. It stops
    # when it has enough evidence — not when a token budget runs out.
    #
    # Called from ThreadPoolExecutor (not a generator) — accumulates trace text
    # internally and logs it. Returns the same dict format as _phase_investigate.

    _AGENTIC_INVESTIGATE_SYSTEM = (
        "You are a senior engineer doing a deep-dive into ONE design decision in a codebase.\n\n"
        "Your goal: produce a 9/10 quality concept card by finding SPECIFIC, GROUNDED evidence.\n\n"
        "TOOLS — call exactly one per turn:\n"
        "  read_file(filepath)         read a source file (see imports, class/function bodies)\n"
        "  search_symbol(name)         find a class or function definition by exact name\n"
        "  find_callers(name)          find all call sites — reveals how this is used in practice\n"
        "  trace_calls(name)           trace call graph from an entry point (depth 3)\n\n"
        "FORMAT — exactly two lines per turn:\n"
        "  THINK: [what you learned and what specific evidence you still need]\n"
        "  TOOL: tool_name(\"argument\")\n\n"
        "  OR when you have enough evidence (verbatim function names, concrete failure mode):\n"
        "  THINK: [what makes this non-obvious]\n"
        "  DONE: {\"name\":\"...\",\"subtitle\":\"...\",\"insight\":\"...\","
        "\"key_functions\":[\"...\"],\"naive_rejected\":\"...\",\"gaps\":\"...\"}\n\n"
        "INVESTIGATION STRATEGY:\n"
        "  1. read_file(primary_file) — get the full picture: imports, class defs, logic\n"
        "  2. search_symbol(key_class_or_function) — find the exact implementation\n"
        "  3. find_callers(key_function) — see how it's invoked (reveals the design contract)\n"
        "  4. trace_calls(entry_point) — follow the call chain to see what it depends on\n"
        "  5. DONE when you can answer: WHY (failure without it), HOW (mechanism), "
        "WHERE (entry point), WHAT (non-obvious aspect)\n\n"
        "QUALITY RULES (enforced — the output is reviewed by an expert):\n"
        "  name: 3-5 words naming the TECHNIQUE, never a filename or class name\n"
        "  subtitle: the SPECIFIC failure/degradation if the simpler approach is used instead\n"
        "    BAD: 'Improves performance'  GOOD: 'Sequential embedding halves throughput on 1000-file repos'\n"
        "  insight: 2-3 sentences — HOW it works (naming real functions) + WHAT surprises a reader\n"
        "  key_functions: VERBATIM names from the code — copy-paste from RESULT blocks\n"
        "  naive_rejected: name the simpler approach + its concrete failure mode\n"
        "  gaps: what design rationale is NOT visible in the code (or 'None')\n"
        "  If the file is pure infrastructure with no interesting technique: "
        "name='Infrastructure: [what it does]', subtitle='', insight='', key_functions=[], "
        "naive_rejected='', gaps=''\n"
        "Return ONLY valid JSON in the DONE: line."
    )

    def _phase_investigate_agentic(self, repo: str, stage: dict, pipeline_context: str) -> dict:
        """
        Agentic deep-dive into one pipeline stage using a ReAct loop.

        Unlike the one-shot version, the agent has tools to follow the design
        decision across multiple files: read the primary file, trace callers to
        see how it's used, trace the call graph to find the core implementation.

        Runs up to 6 rounds — focused investigation (not broad exploration).
        Falls back to static _phase_investigate() if the loop exhausts without
        a valid DONE response.

        Called from ThreadPoolExecutor — NOT a generator. Returns a dict.
        Trace output is logged to stdout (not yielded to the UI).
        """
        stage_file   = stage.get("file", "")
        stage_name   = stage.get("name", "")
        stage_aspect = stage.get("key_aspect", "")

        # Seed the transcript with: what we know about this concept + pipeline context
        transcript = (
            f"Repository: {repo}\n\n"
            f"Design decision to investigate: {stage_name}\n"
            f"Primary file: {stage_file}\n"
            f"What it replaces: {stage_aspect}\n\n"
            f"Pipeline context (where this fits):\n{pipeline_context}\n\n"
            f"Begin investigation. Start with read_file(\"{stage_file}\") to see the full picture.\n"
        )

        max_rounds = 4  # 4 rounds × ~700 tokens × N stages — keep daily budget sane
        for round_n in range(max_rounds):
            raw = self._gen.generate(
                self._AGENTIC_INVESTIGATE_SYSTEM, transcript,
                temperature=0.0, max_tokens=700,  # Gemma 4 needs ~700 for verbose THINK+TOOL
            )

            # Parse THINK + TOOL or DONE
            think_m = _re.search(r'THINK:\s*(.+?)(?:\n|$)', raw, _re.IGNORECASE | _re.DOTALL)
            tool_m  = _re.search(r'TOOL:\s*(\w+)\(\s*"?([^")\n]*)"?\s*\)', raw, _re.IGNORECASE)
            done_m  = _re.search(r'DONE:\s*(\{.+)', raw, _re.DOTALL)

            think_text = (think_m.group(1).strip() if think_m else raw[:120].strip())

            # ── DONE ──────────────────────────────────────────────────────────
            if done_m:
                try:
                    result = _parse_json(done_m.group(1))
                    result.setdefault("name",          stage_name)
                    result.setdefault("subtitle",      stage_aspect)
                    result.setdefault("insight",       "")
                    result.setdefault("key_functions", [])
                    result.setdefault("naive_rejected","")
                    result.setdefault("gaps",          "")
                    print(f"TourAgent.investigate_agentic [{stage_name}] done in {round_n + 1} round(s): {think_text[:80]}")
                    return result
                except Exception:
                    pass  # malformed JSON — keep going

            # ── TOOL CALL ─────────────────────────────────────────────────────
            if tool_m:
                tool_name = tool_m.group(1).lower().replace("-", "_")
                tool_arg  = tool_m.group(2).strip().strip('"').strip("'")

                # No list_files for Phase 2 — investigation is focused, not exploratory
                if tool_name == "read_file":
                    tool_result = self._agentic_read_file(repo, tool_arg)
                    display     = f"read_file(\"{tool_arg}\")"
                elif tool_name == "search_symbol":
                    tool_result = self._agentic_search_symbol(repo, tool_arg)
                    display     = f"search_symbol(\"{tool_arg}\")"
                elif tool_name == "find_callers":
                    tool_result = self._agentic_find_callers(repo, tool_arg)
                    display     = f"find_callers(\"{tool_arg}\")"
                elif tool_name == "trace_calls":
                    tool_result = self._agentic_trace_calls(repo, tool_arg)
                    display     = f"trace_calls(\"{tool_arg}\")"
                else:
                    tool_result = (
                        f"(unknown tool '{tool_name}' — use: read_file, search_symbol, "
                        "find_callers, trace_calls)"
                    )
                    display = tool_name

                print(f"TourAgent.investigate_agentic [{stage_name}] r{round_n+1}: {display}")
                tool_result = _token_budget(tool_result, max_tokens=350)
                transcript += (
                    f"\nTHINK: {think_text}\n"
                    f"TOOL: {display}\n"
                    f"RESULT:\n{tool_result}\n"
                )
            else:
                transcript += f"\n[No valid action in round {round_n + 1}. Output TOOL: or DONE:]\n"

        # ── Exhausted rounds — force final output ──────────────────────────────
        print(f"TourAgent.investigate_agentic [{stage_name}] round limit — forcing DONE")
        transcript += "\nROUND LIMIT REACHED. Output DONE: now with what you have found.\n"
        raw = self._gen.generate(
            self._AGENTIC_INVESTIGATE_SYSTEM, transcript,
            temperature=0.0, max_tokens=1500,  # logs showed Gemma 4 outputs 4866-8344 chars of JSON at 600
        )
        done_m = _re.search(r'DONE:\s*(\{.+)', raw, _re.DOTALL)
        try:
            result = _parse_json(done_m.group(1) if done_m else raw)
            result.setdefault("name",          stage_name)
            result.setdefault("subtitle",      stage_aspect)
            result.setdefault("insight",       "")
            result.setdefault("key_functions", [])
            result.setdefault("naive_rejected","")
            result.setdefault("gaps",          "")
            return result
        except Exception:
            pass

        # Complete fallback: static one-shot investigation
        print(f"TourAgent.investigate_agentic [{stage_name}] failed — falling back to static")
        return self._phase_investigate(repo, stage, pipeline_context)

    # ── Phase 2: Investigate (static fallback) ────────────────────────────────

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
Concept to investigate: {stage_name}
What this replaces: {stage_aspect}

System context:
{pipeline_context}

Source code:
{code_text}

You are investigating the concept "{stage_name}".
The code above should contain its implementation.

Answer these questions using ONLY evidence visible in the code above.
Quote actual function names, class names, or docstring text to ground your answers.
If the code does not contain enough evidence to answer a question, say so — do not invent.

1. WHY: What specific failure or degradation occurs if this technique is removed
   and the simpler alternative is used instead?
2. HOW: What does the code actually do to implement this? Name the key functions
   or classes and describe their role.
3. WHERE: Which function or class is the entry point a new engineer should read first?
4. WHAT: What makes this non-obvious? What would a developer assume before reading
   this code that turns out to be wrong?

Return ONLY this JSON:
{{
  "name": "3-5 words — the technique name, grounded in what the code actually does",
  "subtitle": "One sentence from WHY: the specific failure the simpler approach causes",
  "insight": "2-3 sentences from HOW and WHAT: how it works and what surprises a reader",
  "key_functions": ["exact_function_or_class_name_from_code", "another_exact_name"],
  "naive_rejected": "One sentence: the simpler approach and its concrete failure mode",
  "gaps": "One sentence: what rationale is NOT visible in this code (or 'None')"
}}

Rules:
- key_functions must be names that appear verbatim in the code above
- name and subtitle must be derivable from evidence in the code — no invention
- If the code is pure infrastructure with no interesting technique, set
  name to "Infrastructure: [what it does]" so Phase 3 can skip it
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
      "ask": "MUST NAME a function from key_items and its specific failure: 'What breaks in [key_function] if [specific simpler behaviour] is used instead?'"
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
{_synthesize_negative_block(repo, self._store)}- ask RULE: each ask MUST reference a specific function from key_items and name a concrete failure.
  BAD: "Why was the naive approach rejected?" (zero-information — applies to any concept)
  BAD: "What would happen if this were removed?" (too vague — could be any code anywhere)
  GOOD: "What does [key_function] return when [specific edge case]?" (names the function, names the edge)
  GOOD: "Why does [key_function] need [specific_dependency] before it can run?" (names both pieces)
  GOOD: "What breaks in the output if [key_function] skips the [specific step] it performs?"
  Each ask must be answerable ONLY by reading THIS concept's specific implementation — never vague
- description for id=0: trace the full data flow — what enters (raw input), which stage file handles
  each transformation, what the user receives at the end. Name 3-4 key files in the flow.
"""
        raw = self._gen.generate(_SYNTHESIZE_SYSTEM, prompt, temperature=0.0,
                                  json_mode=True, max_tokens=3000)
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

        # ── Phase 1: Agentic Map (ReAct loop) ────────────────────────────────
        # The agent explores with list_directory + read_file tools, emitting
        # trace events for the UI live-log panel as it reasons about the code.
        yield {
            "stage": "mapping", "progress": 0.15,
            "message": "Exploring codebase with ReAct agent…",
            "trace": {"type": "thinking",
                      "text": "Starting agentic exploration: README → directories → key files…"},
        }

        pipeline_map: dict = {}
        react_prog = 0.15
        for event in self._phase_map_agentic(repo, readme_text):
            if event.get("type") == "result":
                pipeline_map = event.get("data", {})
                break
            # Forward trace events to the UI; advance progress slightly each round.
            # For react events, show only the tool call as the loading message —
            # the full THINK text belongs in the trace panel, not the progress label.
            react_prog = min(react_prog + 0.015, 0.24)
            if event.get("type") == "react":
                msg = event.get("tool", "") or event.get("text", "")
            else:
                msg = event.get("text", "")
            yield {
                "stage": "mapping",
                "progress": react_prog,
                "message": msg,
                "trace": event,
            }

        if not pipeline_map.get("pipeline_stages"):
            yield {"stage": "error", "progress": 1.0,
                   "error": "Pipeline mapping failed — no stages found"}
            return

        stages = pipeline_map.get("pipeline_stages", [])
        entry  = pipeline_map.get("entry_file", "")

        # ── Phase 1 artifact filter ───────────────────────────────────────────
        # Remove any stage whose name is a code identifier (method name, filename,
        # generic single word). These produce hollow Phase 2 results — better to
        # skip them now than waste a LLM call and get an empty concept card.
        clean_stages = [s for s in stages if not _is_artifact_stage_name(s.get("name", ""))]
        if len(clean_stages) < len(stages):
            removed = [s["name"] for s in stages if _is_artifact_stage_name(s.get("name", ""))]
            print(f"TourAgent: filtered {len(removed)} artifact stage(s) from Phase 1: {removed}")

        # ── Phase 1 diversity filter ──────────────────────────────────────────
        # If Phase 1 returned multiple stages from the same file, keep only the
        # first. A good tour covers different parts of the codebase — same-file
        # duplicates mean the agent didn't explore broadly enough. Keeping one
        # preserves the best finding and avoids redundant Phase 2 calls.
        seen_files: set[str] = set()
        diverse_stages: list[dict] = []
        duplicate_stages: list[str] = []
        for s in clean_stages:
            f = s.get("file", "")
            if f and f in seen_files:
                duplicate_stages.append(s["name"])
                continue
            seen_files.add(f)
            diverse_stages.append(s)
        if duplicate_stages:
            print(f"TourAgent: deduped {len(duplicate_stages)} same-file stage(s): {duplicate_stages}")
        clean_stages = diverse_stages if diverse_stages else clean_stages
        stages = clean_stages or stages  # keep originals if everything got filtered

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
        # Uses _phase_investigate_agentic — each worker runs a ReAct loop
        # (up to 6 rounds) with read_file/search_symbol/find_callers/trace_calls.
        insights: list[dict | None] = [None] * n_stages
        with ThreadPoolExecutor(max_workers=_INV_WORKERS) as pool:
            future_to_idx = {
                pool.submit(self._phase_investigate_agentic, repo, stage, pipeline_context): i
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

        # Filter out infrastructure concepts Phase 2 flagged.
        # Phase 2 sets name="Infrastructure: ..." when a file has no interesting
        # technique — these would produce empty concept cards in the tour.
        filtered_stages   = []
        filtered_insights = []
        for stage, insight in zip(stages, insights):
            if insight and insight.get("name", "").lower().startswith("infrastructure:"):
                print(f"TourAgent: skipping infrastructure concept: {insight['name']}")
                continue
            filtered_stages.append(stage)
            filtered_insights.append(insight)

        if filtered_stages != stages:
            stages   = filtered_stages
            insights = filtered_insights
            pipeline_map = {**pipeline_map, "pipeline_stages": stages}

        # ── Hollow insight filter ─────────────────────────────────────────────
        # Phase 2 may return an insight with empty subtitle AND empty insight text.
        # This happens when Phase 2 investigated a file and found no meaningful
        # technique (but also didn't return "Infrastructure: ..." cleanly). An
        # insight with no content produces an empty card in the tour — skip it.
        non_hollow = [
            (s, ins) for s, ins in zip(stages, insights)
            if ins and (ins.get("subtitle", "").strip() or ins.get("insight", "").strip())
        ]
        if len(non_hollow) < len(stages):
            hollow_names = [ins.get("name", "?") for s, ins in zip(stages, insights)
                            if not (ins and (ins.get("subtitle","").strip() or ins.get("insight","").strip()))]
            print(f"TourAgent: skipping {len(hollow_names)} hollow insight(s): {hollow_names}")
        if non_hollow:
            stages   = [s for s, _ in non_hollow]
            insights = [ins for _, ins in non_hollow]
            pipeline_map = {**pipeline_map, "pipeline_stages": stages}

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

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
Agent:    ~45-90s, 7-10 LLM calls, traced grounded understanding

Free-tier friendly: sequential (no parallel) to stay within Gemini's free quota.
"""

from __future__ import annotations

import json as _json
import re as _re
from typing import Generator


# Entry file patterns — most likely "top of the call stack".
_ENTRY_NAMES = {
    "main.py", "app.py", "server.py", "__init__.py",
    "agent.py", "run.py", "train.py", "pipeline.py", "index.py", "engine.py",
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
    "Return ONLY valid JSON, no markdown, no explanation."
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
        """
        all_c  = self._all_chunks(repo)
        result = []
        seen_files: set[str] = set()
        for c in all_c:
            fp    = c["filepath"]
            fname = fp.split("/")[-1]
            is_entry  = fname in _ENTRY_NAMES
            is_module = c["chunk_type"] == "module"
            if (is_entry or is_module) and fp not in seen_files:
                seen_files.add(fp)
                result.append(c)
        return result

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

        Combines README (stated purpose) with entry-file imports (actual call
        graph). This is the key improvement from /init Phase 2 — reading the
        README before mapping prevents misidentifying internal utilities as the
        main pipeline.
        """
        module_chunks = self._entry_module_chunks(repo)[:14]
        all_files     = self._list_file_paths(repo)

        chunk_text = "\n\n".join(self._fmt_module_chunk(c) for c in module_chunks)
        files_text = "\n".join(f"  {fp}" for fp in all_files[:80])

        readme_section = (
            f"What the repo claims to do (from README):\n{readme_text}\n"
            if readme_text else ""
        )

        prompt = f"""Repository: {repo}

{readme_section}All indexed files:
{files_text}

Module-level code from entry files (imports + calls visible):
{chunk_text}

Using BOTH the README (stated purpose) and the import graph (actual structure),
trace the main user-facing pipeline — the critical path from user action to output.

Return ONLY this JSON:
{{
  "entry_file": "file at the top of the call stack (e.g. main.py or app.py)",
  "readme_summary": "1-2 sentences: what the README says this repo does for the user",
  "pipeline_stages": [
    {{
      "name": "Short stage name (2-4 words)",
      "file": "filepath/to/stage.py (must appear in the indexed files above)",
      "key_aspect": "One sentence: what this stage does and why it matters"
    }}
  ]
}}

Rules:
- 3-6 stages on the critical path only, ordered by execution sequence
- Each file must appear in the indexed files list above
- Skip test files, config files, admin utilities, and UI files
- If README mentions a key technique (e.g. RAG, hybrid search, AST chunking),
  make sure the relevant stage appears in the pipeline
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
  "name": "3-5 words — the key technique or component (class names OK if they explain design)",
  "subtitle": "One sentence: WHY this exists — the specific problem it solves",
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
      "ask": "How does the full pipeline work end to end?"
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
      "ask": "Why was the naive approach rejected here?"
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

        # ── Phase 2: Investigate each stage ───────────────────────────────────
        pipeline_context = "\n".join(
            f"  {i+1}. {s['name']}: {s.get('key_aspect', '')}"
            for i, s in enumerate(stages)
        )
        insights   = []
        n_stages   = len(stages)
        base_prog  = 0.25
        stage_step = 0.55 / max(n_stages, 1)

        for i, stage in enumerate(stages):
            prog = base_prog + i * stage_step

            # Report whether we'll expand (broad) or go narrow
            primary_count = len(self._file_chunks(repo, stage.get("file", "")))
            search_mode   = "broad (expanding to related files)" if primary_count < 3 else "deep"

            yield {
                "stage": "investigating", "progress": prog,
                "message": f"Investigating: {stage['name']}… ({i+1}/{n_stages})",
                "trace": {"type": "file",
                          "text": f"{stage.get('file', '')} — {search_mode}",
                          "step": i + 1,
                          "total": n_stages},
            }

            insight = self._phase_investigate(repo, stage, pipeline_context)
            insights.append(insight)

            # Surface gaps in the trace panel — shows users what the agent
            # couldn't determine, building honest expectations about the tour.
            gaps_text = insight.get("gaps", "")
            trace_text = (insight.get("insight") or "")[:120]
            if gaps_text and gaps_text.lower() != "none":
                trace_text += f" | gap: {gaps_text[:60]}"

            yield {
                "stage": "investigating", "progress": prog + stage_step * 0.8,
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

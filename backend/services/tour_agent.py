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
each modelled after the structured investigation prompts in Claude Code
(architecture_overview → explain_tool × N → synthesize):

  Phase 1 — MAP   (= architecture_overview)
    "What is the main pipeline and which files own each stage?"
    Input:  module-level chunks from entry files (imports + calls visible)
    Output: { entry_file, pipeline_stages: [{name, file, key_aspect}] }

  Phase 2 — INVESTIGATE   (= explain_tool / how_does_it_work, once per stage)
    "What is the non-obvious design decision in this stage?"
    Input:  all chunks from one specific file (deep, narrow context)
    Output: { name, subtitle, insight, key_functions, naive_rejected }

  Phase 3 — SYNTHESIZE
    "Convert this traced understanding into tour JSON."
    Input:  pipeline map + per-stage insights (structured findings, no raw code)
    Output: full tour JSON with proper dependency tree

Each LLM call gets only what it needs. Phase 2 doesn't see other files.
Phase 3 doesn't see raw code at all — only structured findings. This mirrors
how a good engineer actually reads an unfamiliar codebase: overview first,
then targeted deep-dives, then mental-model consolidation.

TRADE-OFFS
───────────
One-shot: ~10-20s, 1 LLM call, shallow guessed understanding
Agent:    ~30-60s, 6-8 LLM calls, traced grounded understanding

Free-tier friendly: all calls stay within Gemini's free quota.
"""

from __future__ import annotations

import json as _json
import re as _re
from typing import Generator


# Entry file patterns — most likely "top of the call stack".
# Used in Phase 1 to narrow which module chunks to show the mapping LLM.
_ENTRY_NAMES = {
    "main.py", "app.py", "server.py", "__init__.py",
    "agent.py", "run.py", "train.py", "pipeline.py", "index.py", "engine.py",
}

# ── System prompts for each phase ─────────────────────────────────────────────
# Each is tightly scoped — the LLM knows exactly what it's trying to do.

_MAP_SYSTEM = (
    "You are a senior engineer mapping an unfamiliar codebase for the first time. "
    "Trace the main user-facing pipeline — the sequence of files and functions that "
    "execute when the system does its primary job. "
    "NEVER list every file. Focus on the critical path from user action to output. "
    "Return ONLY valid JSON, no markdown, no explanation."
)

_INVESTIGATE_SYSTEM = (
    "You are a senior engineer doing a deep-dive into one component of a pipeline. "
    "You know where this component sits in the larger system. "
    "Your job: identify the KEY non-obvious design decision in this code. "
    "State the failure mode that would occur with the naive alternative. "
    "Every claim must be grounded in the actual code shown. "
    "Return ONLY valid JSON, no markdown, no explanation."
)

_SYNTHESIZE_SYSTEM = (
    "You are a senior engineer writing the guided tour you wished existed before "
    "reading this codebase. You have already traced the full pipeline and investigated "
    "each stage. Convert your traced findings into the structured tour format. "
    "The dependency tree must reflect conceptual prerequisites: a developer cannot "
    "understand concept B without first understanding concept A. "
    "Return ONLY valid JSON, no markdown, no explanation."
)


class TourAgent:
    """
    Three-phase agent for building codebase concept tours.

    Each phase is a focused LLM call with only the relevant context.
    Progress is streamed via a generator — callers iterate over the yielded
    events and forward them as SSE to the frontend.

    Usage:
        agent = TourAgent(store, gen)
        for event in agent.build(repo):
            # event matches the existing tour SSE schema
            stream_to_client(event)

    The "trace" key in each event carries agent-trace info for the UI live-log:
        {"type": "file"|"finding"|"thinking"|"found"|"info", ...}
    """

    def __init__(self, store, gen):
        self._store = store
        self._gen   = gen
        # Cache all_chunks per repo within one build() call to avoid
        # repeated Qdrant scrolls (each scroll fetches all chunks).
        self._chunk_cache: dict[str, list[dict]] = {}

    # ── Internal tools ─────────────────────────────────────────────────────────

    def _all_chunks(self, repo: str) -> list[dict]:
        """
        Load all indexed chunks for a repo.

        Uses Qdrant scroll (no vectors) for complete coverage — semantic search
        would miss files outside the top-k for any particular query.
        Cached after the first call so all three phases share one Qdrant round-trip.
        """
        if repo not in self._chunk_cache:
            raw = self._store.scroll_repo(repo)
            # scroll_repo returns raw Qdrant payload dicts; normalise field names
            # to match what DiagramService's _list_chunks returns.
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
        """All chunks for a specific file."""
        all_c = self._all_chunks(repo)
        fp_lower = filepath.lower()
        return [
            c for c in all_c
            if (c["filepath"].lower() == fp_lower
                or c["filepath"].lower().endswith("/" + fp_lower)
                or fp_lower in c["filepath"].lower())
        ]

    def _entry_module_chunks(self, repo: str) -> list[dict]:
        """
        Module-level chunks from likely entry-point files.

        Module chunks contain imports and file-level calls — they reveal
        what a file connects to without exposing full implementation details.
        Phase 1 (mapping) uses these to trace the pipeline.
        """
        all_c  = self._all_chunks(repo)
        result = []
        seen_files: set[str] = set()
        for c in all_c:
            fp   = c["filepath"]
            fname = fp.split("/")[-1]
            is_entry  = fname in _ENTRY_NAMES
            is_module = c["chunk_type"] == "module"
            if (is_entry or is_module) and fp not in seen_files:
                seen_files.add(fp)
                result.append(c)
        return result

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

    # ── Prompt formatters ──────────────────────────────────────────────────────

    def _fmt_module_chunk(self, c: dict, max_len: int = 500) -> str:
        """Format a module-level chunk for the mapping phase."""
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

    def _fmt_file_for_investigation(self, chunks: list[dict], max_len: int = 700) -> str:
        """Format all chunks from one file for the investigation phase."""
        parts = []
        for c in chunks[:12]:
            name  = c["name"] or "?"
            ctype = c["chunk_type"]
            sl    = c["start_line"]
            el    = c["end_line"]
            text  = c["text"].strip()[:max_len]
            header = f"### {name} ({ctype}) — lines {sl}–{el}"
            parts.append(f"{header}\n{text}" if text else header)
        return "\n\n".join(parts)

    # ── Phase 1: Map ───────────────────────────────────────────────────────────

    def _phase_map(self, repo: str) -> dict:
        """
        Identify the main pipeline and its stages.

        Shows the LLM module-level chunks (imports + calls) from entry files.
        This reveals the call graph at a high level without overwhelming the
        LLM with implementation details.

        Returns: { "entry_file": str, "pipeline_stages": [{name, file, key_aspect}] }
        """
        module_chunks = self._entry_module_chunks(repo)[:14]
        all_files     = self._list_file_paths(repo)

        chunk_text = "\n\n".join(self._fmt_module_chunk(c) for c in module_chunks)
        files_text = "\n".join(f"  {fp}" for fp in all_files[:80])

        prompt = f"""Repository: {repo}

All indexed files:
{files_text}

Module-level code from entry files (imports + calls visible):
{chunk_text}

Trace the main user-facing pipeline. Follow the imports and calls above to
find the critical path from user action to final output.

Return ONLY this JSON:
{{
  "entry_file": "the file at the top of the call stack (e.g. main.py or app.py)",
  "pipeline_stages": [
    {{
      "name": "Short stage name (2-4 words)",
      "file": "filepath/to/stage.py (must exist in the index above)",
      "key_aspect": "One sentence: what this stage does in the pipeline"
    }}
  ]
}}

Rules:
- 3-6 stages on the critical path only, ordered by execution sequence
- Each stage file must appear in the indexed files list above
- Skip test files, config files, admin utilities, and UI files
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
            # Graceful fallback: use the first few module chunks as stages
            fallback_stages = [
                {"name": c["name"] or c["filepath"].split("/")[-1],
                 "file": c["filepath"],
                 "key_aspect": ""}
                for c in module_chunks[:5]
            ]
            return {
                "entry_file": module_chunks[0]["filepath"] if module_chunks else "",
                "pipeline_stages": fallback_stages,
            }

    # ── Phase 2: Investigate ───────────────────────────────────────────────────

    def _phase_investigate(self, repo: str, stage: dict, pipeline_context: str) -> dict:
        """
        Deep-dive into one pipeline stage.

        Gets all chunks for the stage file and asks for the key design decision.
        Pipeline context is included so the LLM understands WHERE this stage
        fits in the flow — same pattern as Claude Code's explain_tool prompt
        which always includes the surrounding context.

        Returns: { name, subtitle, insight, key_functions, naive_rejected }
        """
        stage_file   = stage.get("file", "")
        stage_name   = stage.get("name", "")
        stage_aspect = stage.get("key_aspect", "")

        chunks = self._file_chunks(repo, stage_file)
        if not chunks:
            # Fallback: find chunks whose name matches the stage name
            lname  = stage_name.lower()
            all_c  = self._all_chunks(repo)
            chunks = [c for c in all_c if lname in c["name"].lower()][:6]

        if not chunks:
            return {
                "name": stage_name, "subtitle": stage_aspect,
                "insight": stage_aspect, "key_functions": [], "naive_rejected": "",
            }

        code_text = self._fmt_file_for_investigation(chunks)

        prompt = f"""Repository: {repo}
Stage: {stage_name}
Role in pipeline: {stage_aspect}

Full pipeline (for context):
{pipeline_context}

Code for this stage — {stage_file}:
{code_text}

What is the KEY non-obvious design decision in this stage?

Return ONLY this JSON:
{{
  "name": "Technique or decision (3-5 words — never a class/file/service name)",
  "subtitle": "One sentence: the specific problem this solves in the pipeline",
  "insight": "2-3 sentences: the naive approach and its failure mode, what this code does instead, the non-obvious insight that makes it work",
  "key_functions": ["actual_function_1", "actual_function_2"],
  "naive_rejected": "One sentence: what simpler approach was NOT used and why"
}}

Rules:
- Name the TECHNIQUE, not the artifact (bad: 'QdrantStore', good: 'Dual-Vector Hybrid Search')
- key_functions must be actual method names visible in the code above
- insight must state a concrete failure mode with the naive approach
"""
        raw = self._gen.generate(_INVESTIGATE_SYSTEM, prompt, temperature=0.0,
                                  json_mode=True, max_tokens=800)
        try:
            result = _parse_json(raw)
            result.setdefault("name",          stage_name)
            result.setdefault("subtitle",      stage_aspect)
            result.setdefault("insight",       "")
            result.setdefault("key_functions", [])
            result.setdefault("naive_rejected","")
            return result
        except Exception as e:
            print(f"TourAgent._phase_investigate failed for {stage_name}: {e}")
            return {
                "name": stage_name, "subtitle": stage_aspect,
                "insight": stage_aspect, "key_functions": [], "naive_rejected": "",
            }

    # ── Phase 3: Synthesize ────────────────────────────────────────────────────

    def _phase_synthesize(self, repo: str, pipeline_map: dict, insights: list[dict]) -> dict:
        """
        Convert traced understanding to tour JSON.

        The LLM receives structured findings — not raw code. It only has to
        format and assign dependency relationships, not simultaneously read
        and understand. Separation of concerns produces far better output.
        """
        stages  = pipeline_map.get("pipeline_stages", [])
        entry   = pipeline_map.get("entry_file", "")

        stages_text = "\n".join(
            f"  {i+1}. {s['name']} — {s['file']}: {s.get('key_aspect','')}"
            for i, s in enumerate(stages)
        )
        insights_text = "\n\n".join(
            f"Stage {i+1} — {ins.get('name', '?')}\n"
            f"  subtitle:       {ins.get('subtitle', '')}\n"
            f"  insight:        {ins.get('insight', '')}\n"
            f"  key_functions:  {', '.join(ins.get('key_functions', []))}\n"
            f"  naive_rejected: {ins.get('naive_rejected', '')}"
            for i, ins in enumerate(insights)
        )

        prompt = f"""Repository: {repo}
Entry file: {entry}

Pipeline traced in order:
{stages_text}

Per-stage findings (already investigated — use these verbatim):
{insights_text}

Convert this traced understanding into a concept tour JSON.

Concept id=0 (reading_order=1, depends_on=[]) MUST be the end-to-end pipeline
overview — what enters, what stages transform it, what the user gets out.
All other concepts must have depends_on pointing to at least one earlier concept.

Return ONLY this JSON:
{{
  "summary": "2 sentences: (1) what the user can DO with this repo and what mechanism makes it possible — name the technique. (2) the single architectural decision that shapes everything else.",
  "entry_point": "{entry}",
  "concepts": [
    {{
      "id": 0,
      "name": "End-to-end pipeline name (3-5 words)",
      "subtitle": "What this pipeline does for the user",
      "file": "{entry}",
      "type": "module",
      "description": "2-3 sentences tracing the full flow: what enters, how each stage transforms it, what the user gets. Name the key files and the architectural split that makes it work.",
      "key_items": ["function_1", "function_2"],
      "depends_on": [],
      "reading_order": 1,
      "ask": "How does the full pipeline work end to end?"
    }},
    {{
      "id": 1,
      "name": "Use the exact 'name' field from stage 1 findings above",
      "subtitle": "Use the exact 'subtitle' field from stage 1 findings above",
      "file": "file from stage 1",
      "type": "class|function|module|algorithm",
      "description": "Use the exact 'insight' field from stage 1 findings above",
      "key_items": ["use exact key_functions from findings"],
      "depends_on": [0],
      "reading_order": 2,
      "ask": "Why was the naive approach rejected here?"
    }}
  ]
}}

Rules:
- 6-8 concepts total (concept 0 = pipeline overview, concepts 1+ = one per stage insight)
- Use the EXACT name, subtitle, insight, key_functions from the per-stage findings above
- All concepts except id=0 must have depends_on non-empty
- reading_order: sequential integers starting at 1
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

    # ── Main entry point ───────────────────────────────────────────────────────

    def build(self, repo: str) -> Generator[dict, None, None]:
        """
        Build a concept tour for a repo, yielding SSE-compatible progress events.

        Each yielded dict matches the existing tour SSE schema so DiagramService
        can drop this in without changing the router or frontend SSE consumer.

        Extra "trace" key carries live-log data for the UI agent trace panel:
            {"type": "info"|"thinking"|"found"|"file"|"finding"}
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
            "stage": "mapping", "progress": 0.10,
            "message": f"Mapping {n_files} files, {n_chunks} chunks…",
            "trace": {"type": "info",
                      "text": f"{n_chunks} chunks across {n_files} files"},
        }

        # ── Phase 1: Map ──────────────────────────────────────────────────────
        yield {
            "stage": "mapping", "progress": 0.15,
            "message": "Identifying pipeline stages…",
            "trace": {"type": "thinking", "text": "Tracing entry points and imports…"},
        }

        try:
            pipeline_map = self._phase_map(repo)
        except Exception as e:
            yield {"stage": "error", "progress": 1.0,
                   "error": f"Pipeline mapping failed: {e}"}
            return

        stages = pipeline_map.get("pipeline_stages", [])
        entry  = pipeline_map.get("entry_file", "")
        stage_list = " → ".join(s["name"] for s in stages)

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
        stage_step = 0.55 / max(n_stages, 1)   # investigation covers 25%→80%

        for i, stage in enumerate(stages):
            prog = base_prog + i * stage_step
            yield {
                "stage": "investigating", "progress": prog,
                "message": f"Investigating: {stage['name']}… ({i+1}/{n_stages})",
                "trace": {"type": "file",
                          "text": stage.get("file", ""),
                          "step": i + 1,
                          "total": n_stages},
            }

            insight = self._phase_investigate(repo, stage, pipeline_context)
            insights.append(insight)

            yield {
                "stage": "investigating", "progress": prog + stage_step * 0.8,
                "message": f"Found: {insight.get('name', stage['name'])}",
                "trace": {"type": "finding",
                          "name": insight.get("name", ""),
                          "text": (insight.get("insight") or "")[:140]},
            }

        # ── Phase 3: Synthesize ───────────────────────────────────────────────
        yield {
            "stage": "synthesizing", "progress": 0.82,
            "message": "Synthesizing concept tour…",
            "trace": {"type": "thinking",
                      "text": "Building dependency tree from traced insights…"},
        }

        try:
            tour = self._phase_synthesize(repo, pipeline_map, insights)
        except Exception as e:
            yield {"stage": "error", "progress": 1.0,
                   "error": f"Synthesis failed: {e}"}
            return

        yield {"stage": "done", "progress": 1.0, **tour}


# ── Shared JSON parser (same logic as DiagramService._parse_json) ──────────────

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

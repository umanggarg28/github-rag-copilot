"""
readme_service.py — On-demand README generation from indexed repo metadata.

═══════════════════════════════════════════════════════════════
WHAT THIS GENERATES
═══════════════════════════════════════════════════════════════

A full README.md for any indexed repository, grounded in real AST data:
  - Project description inferred from file structure and class names
  - Architecture section built from the repo map (modules, key classes)
  - Key components: top functions/classes with their roles
  - Tech stack: inferred from imports and language distribution
  - Setup stub: entry-point detection + language-appropriate install command

The result is cached to disk — regeneration requires an explicit force=True.

═══════════════════════════════════════════════════════════════
DESIGN: REPO MAP GROUNDING
═══════════════════════════════════════════════════════════════

The LLM never reads raw source files. Instead, we feed it the repo map:
a compact structured summary built from AST metadata at ingest time.
This keeps the prompt under ~500 tokens while giving the model enough
signal to write accurate, repo-specific documentation.

The repo map includes:
  - Total chunk count and file count
  - Entry-point files (detected heuristically)
  - Key class names across the repo
  - Per-file breakdown: class names and top function names

This is the same grounding technique used by diagram_service.py — the
LLM annotates structure it couldn't have invented; it doesn't invent structure.
"""

import json
from backend.services.generation import GenerationService
from backend.services.repo_map_service import RepoMapService
from ingestion.qdrant_store import QdrantStore


# README content is stored under (repo, kind="readme") in the shared
# Qdrant artifacts collection — see QdrantStore.save_artifact. No more
# disk caching: HF Spaces filesystem is ephemeral, so disk reads after
# every container restart returned a miss and triggered re-generation.

class ReadmeService:
    def __init__(
        self,
        repo_map_svc: RepoMapService,
        gen:          GenerationService,
        store:        QdrantStore,
    ):
        self._repo_map = repo_map_svc
        self._gen      = gen
        self._store    = store

    def build_readme_stream(self, repo: str, force: bool = False):
        """
        Generate a README for `repo`, yielding SSE-style progress events.

        Event shapes:
          {"stage": "loading",     "progress": 0.10, "message": "..."}
          {"stage": "generating",  "progress": 0.40, "message": "..."}
          {"stage": "done",        "progress": 1.0,  "content": "<markdown>"}
          {"stage": "error",       "progress": 1.0,  "error":   "<message>"}

        The final "done" event carries the full markdown as `content`.
        All preceding events are progress updates for the UI progress bar.
        """
        # ── Cache hit ─────────────────────────────────────────────────────────
        if not force:
            meta = self._store.load_artifact_meta(repo, "readme")
            if meta and meta.get("data") and meta["data"].get("content"):
                print(f"[cache hit] readme for {repo} ({meta.get('generated_by_model', 'unknown')})")
                yield {"stage": "loading", "progress": 0.1, "message": "Loading cached README…"}
                yield {"stage": "done", "progress": 1.0,
                       "content": meta["data"]["content"], "from_cache": True}
                return

        # ── Build repo map ────────────────────────────────────────────────────
        yield {"stage": "loading", "progress": 0.15, "message": "Analysing repository structure…"}
        try:
            repo_map = self._repo_map.get_or_build(repo)
        except Exception as e:
            yield {"stage": "error", "progress": 1.0, "error": f"Could not build repo map: {e}"}
            return

        # ── Prepare prompt context ────────────────────────────────────────────
        yield {"stage": "generating", "progress": 0.40, "message": "Generating README…"}

        owner, name = repo.split("/", 1) if "/" in repo else ("", repo)
        map_text    = self._repo_map.format_for_prompt(repo_map)
        n_chunks    = repo_map.get("total_chunks", 0)
        n_files     = repo_map.get("total_files", 0)
        languages   = repo_map.get("languages", {})
        primary_lang = max(languages, key=languages.get) if languages else "unknown"

        # Build a language stats line (e.g. "Python 80%, TypeScript 20%")
        lang_line = ""
        if languages:
            total = sum(languages.values()) or 1
            parts = [f"{lang} {round(v/total*100)}%"
                     for lang, v in sorted(languages.items(), key=lambda x: -x[1])[:4]]
            lang_line = ", ".join(parts)

        system = (
            "You are an expert technical writer specialising in developer documentation. "
            "Your READMEs give developers a mental model of the codebase, not a table of contents. "
            "NEVER write exhaustive class/function lists — those are what grep is for. "
            "NEVER pad with filler: 'This project aims to', 'This is a powerful tool', 'leverages', 'robust'. "
            "NEVER invent features, classes, or behaviours not present in the repo map provided. "
            "NEVER write a generic description that could apply to any project in the same category. "
            "ALWAYS answer: why does this architecture work this way (not just what it contains). "
            "Every sentence must be specific to THIS repository. "
            "Use GitHub-flavored markdown."
        )

        prompt = f"""Generate a high-quality README.md for the GitHub repository `{repo}`.

REPO MAP (built from AST metadata — authoritative ground truth):
{map_text}

Additional stats:
  - Primary language: {primary_lang}
  - Language breakdown: {lang_line or "N/A"}
  - Total indexed: {n_chunks} code chunks across {n_files} files

═══ DOCUMENTATION PHILOSOPHY ═══

Write for a developer who has found this repo and wants to know:
  1. Can this do what I need? (purpose — first sentence)
  2. How does the architecture work? (mental model, not a class listing)
  3. Where do I start reading? (entry point, key files)
  4. What non-obvious decisions were made? (design rationale)

NEVER write exhaustive class/function lists — those are what `grep` is for.
NEVER write sections that duplicate what the code already makes obvious.
EVERY sentence must earn its place: if it would be true for any project in this category, cut it.

═══ SECTIONS (use exactly these headings) ═══

# {name}

[One sentence: what this does and for whom. Start with the action verb.
Lead with mechanism, not category. Not "A web framework" but "Routes HTTP requests to handlers using a declarative path registry with middleware composited per-route."]

## How it works

[3-4 sentences tracing the data flow end to end. Name the key pipeline stages and the split
that shapes the architecture (e.g. "index upfront, query at runtime" or "embed → store → retrieve → generate").
Reference actual class names from the map when they clarify the mechanism.
Explain the ONE architectural decision that everything else depends on.]

## Architecture

[3-5 bullet points: key modules and their role in the pipeline.
Each bullet: `filename.py` — one sentence on what it does and WHY it's separate.
NEVER list every file — only the ones a new engineer must understand to navigate the codebase.
Skip test files, config boilerplate, and admin utilities.]

## Key Design Decisions

[2-3 bullet points on the non-obvious choices: why this approach over the simpler alternative.
e.g. "AST chunking over sliding windows — splits at function boundaries so every retrieved chunk is self-contained."
Only include decisions that are genuinely non-obvious or that answer a question a reader would ask.]

## Getting Started

[Minimal setup + usage. Show the entry point command or import.
Use a fenced code block with the correct language tag.]

## Tech Stack

[Compact bullet list: language(s), key libraries inferred from imports, any external services.]

---
Output ONLY the markdown. No preamble, no "Here is the README", no trailing commentary."""

        # ── Call LLM ─────────────────────────────────────────────────────────
        try:
            content = self._gen.generate(
                system=system,
                prompt=prompt,
                temperature=0.3,
                max_tokens=1800,
            )
        except Exception as e:
            yield {"stage": "error", "progress": 1.0, "error": f"Generation failed: {e}"}
            return

        import re as _re

        # Strip chain-of-thought reasoning blocks — some models (DeepSeek, Qwen, etc.)
        # emit <thought>...</thought> or <think>...</think> before the real output.
        # These must be removed before any other processing.
        content = _re.sub(r'<thought>.*?</thought>', '', content, flags=_re.DOTALL | _re.IGNORECASE)
        content = _re.sub(r'<think>.*?</think>',    '', content, flags=_re.DOTALL | _re.IGNORECASE)
        content = content.strip()

        # Strip outer markdown/text code fence if the LLM wraps its entire output.
        # e.g. ```markdown\n# micrograd\n...\n``` → # micrograd\n...
        content = _re.sub(r'^```(?:markdown|md|text)?\s*\n', '', content.strip(), flags=_re.IGNORECASE)
        content = _re.sub(r'\n```\s*$', '', content.strip())

        # Strip any accidental preamble the model sometimes adds before the heading
        if "# " in content and not content.lstrip().startswith("#"):
            content = content[content.index("# "):]

        # Enforce the correct repo name as the H1 title.
        # The LLM sometimes writes a descriptive title ("Simple autograd") instead of
        # the actual repo name ("micrograd"). Replace whatever H1 was generated with
        # the authoritative name so the document always identifies itself correctly.
        content = _re.sub(r'^# .+', f'# {name}', content, count=1, flags=_re.MULTILINE)

        # Clean up stray backticks the LLM sometimes appends to the H1 title.
        # e.g. "# micrograd`" → "# micrograd"
        content = _re.sub(r'^(#+ .+?)`+\s*$', r'\1', content, flags=_re.MULTILINE)

        # ── Cache + emit ──────────────────────────────────────────────────────
        self._store.save_artifact(
            repo, "readme", {"content": content},
            generated_by_model=self._gen.current_model(),
        )
        yield {"stage": "done", "progress": 1.0, "content": content, "from_cache": False}

    def invalidate(self, repo: str) -> None:
        """Remove cached README so the next request regenerates it."""
        self._store.delete_artifact(repo, "readme")

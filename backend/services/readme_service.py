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
from pathlib import Path

from backend.services.generation import GenerationService
from backend.services.repo_map_service import RepoMapService


_CACHE_DIR = Path(__file__).parent.parent / "readmes"
_CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(repo: str) -> Path:
    slug = repo.replace("/", "_")
    return _CACHE_DIR / f"{slug}_readme.md"


class ReadmeService:
    def __init__(self, repo_map_svc: RepoMapService, gen: GenerationService):
        self._repo_map = repo_map_svc
        self._gen      = gen

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
        cache = _cache_path(repo)

        # ── Cache hit ─────────────────────────────────────────────────────────
        if not force and cache.exists():
            yield {"stage": "loading", "progress": 0.1, "message": "Loading cached README…"}
            content = cache.read_text(encoding="utf-8")
            yield {"stage": "done", "progress": 1.0, "content": content, "from_cache": True}
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
            "You are an expert technical writer who creates README files for open-source projects. "
            "Your READMEs are known for being direct, accurate, and immediately useful — "
            "a developer reads them once and knows exactly what the project does and how to use it. "
            "NEVER pad with filler: 'This project aims to', 'This is a powerful tool', 'leverages', 'robust'. "
            "NEVER invent features, classes, or behaviours not present in the repo map provided. "
            "NEVER write a generic description that could apply to any project in the same category. "
            "Every sentence must be specific to THIS repository. "
            "Use GitHub-flavored markdown. Start each section with the most important information."
        )

        prompt = f"""Generate a complete, high-quality README.md for the GitHub repository `{repo}`.

REPO MAP (built from AST metadata — ground truth):
{map_text}

Additional stats:
  - Primary language: {primary_lang}
  - Language breakdown: {lang_line or "N/A"}
  - Total indexed: {n_chunks} code chunks across {n_files} files

Generate the README with these sections IN ORDER. Use the exact headings shown:

# {name}

[One punchy sentence describing what the project does and who it's for.
No "This is a..." — start with what it does. Lead with the action: what it builds, parses, runs, searches, or transforms.]

## What it does

[2-3 sentences. What problem does it solve? What is the core mechanism?
Be specific to THIS repo — reference actual class/function names from the map.]

## Architecture

[How is the code structured? What are the main modules and their roles?
Reference actual file names and key classes. 3-5 bullet points.]

## Key Components

[4-6 bullet points. Each: `ClassName` or `function_name` — what it does in one sentence.
Only include things that actually appear in the repo map.]

## Usage

[Minimal example showing the most important API or entry point.
Use a fenced code block. Base this on actual class names you see in the map.
If this is a library, show how to import and use the main class.
If this is a tool/app, show the command to run it.]

## Tech Stack

[Bullet list: language, key dependencies inferred from imports, notable design choices.]

---
Output ONLY the markdown. No preamble, no "Here is the README", no trailing notes."""

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
        cache.write_text(content, encoding="utf-8")
        yield {"stage": "done", "progress": 1.0, "content": content, "from_cache": False}

    def invalidate(self, repo: str) -> None:
        """Remove cached README so the next request regenerates it."""
        p = _cache_path(repo)
        if p.exists():
            p.unlink()

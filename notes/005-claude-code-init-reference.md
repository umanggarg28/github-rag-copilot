# Claude Code /init — Source Analysis Reference

> Studied from: https://github.com/codeaashu/claude-code/
> Key files: `src/commands/init.ts`, `src/tools/AgentTool/built-in/exploreAgent.ts`,
>            `src/tools/AgentTool/built-in/{generalPurposeAgent,planAgent,verificationAgent}.ts`
> Purpose: Reference for improving Cartographer's tour agent prompts and flow

---

## 1. Architecture: Two-Prompt System

Claude Code has `OLD_INIT_PROMPT` (single-pass legacy) and `NEW_INIT_PROMPT` (feature-gated 8-phase workflow).
`NEW_INIT_PROMPT` is the current design — it uses a **subagent** for exploration rather than doing it inline.

```
Phase 1: Ask user — project CLAUDE.md or personal?
Phase 2: Launch Explore subagent → survey codebase
Phase 3: Gap-fill interview (AskUserQuestion tool) — things code alone couldn't answer
Phase 4: Generate first-draft CLAUDE.md
Phase 5: Self-critique and refine
Phase 6: Show diff to user, get approval
Phase 7: Write final CLAUDE.md to disk
Phase 8: Confirm done
```

**Key insight**: exploration and generation are completely separated. The subagent only reads;
the main agent only writes. This is why Phase 2 can be aggressive (read everything) without
worrying about accidentally modifying anything.

---

## 2. File Reading Order (Phase 2 — verbatim from NEW_INIT_PROMPT)

```
1. Manifest files:   package.json, Cargo.toml, pyproject.toml, go.mod, pom.xml, etc.
2. README
3. Build/Makefile
4. CI config
5. Existing CLAUDE.md (if present)
6. AI tool configs:  .claude/rules/, AGENTS.md, .cursor/rules, .cursorrules,
                     .github/copilot-instructions.md, .windsurfrules, .clinerules, .mcp.json
```

**Why manifests first**: dependencies reveal project type without directory-name heuristics.
`fastapi + qdrant-client` → web API. `torch + transformers` → ML pipeline. `llvm-sys` → compiler.
The manifest is the repo's most honest self-description.

---

## 3. Detection Goals (explicit targets, not "explore until done")

Phase 2 doesn't say "explore and stop when satisfied." It names specific signals to find:

```
- Build, test, and lint commands (especially non-standard ones)
- Languages, frameworks, package manager
- Project structure: monorepo / multi-module / single project
- Code style rules that differ from language defaults
- Non-obvious gotchas, required env vars, workflow quirks
- Existing skills and rules directories
- Formatter config (prettier, ruff, black, gofmt, rustfmt, or unified scripts)
- Git worktree usage (git worktree list)
```

**Key insight for Cartographer**: our Phase 1 prompt says "5-8 decisions" as the stopping
criterion. A better stopping criterion: "have I found the primary data transformation, the
key algorithm, the non-obvious library choices, and the entry point?" — specific signals,
not a count.

---

## 4. The Gap Pattern (most important finding)

**Verbatim from NEW_INIT_PROMPT Phase 2 completion instruction**:
> "Note what you could NOT figure out from code alone — these become interview questions."

This is the termination signal: the agent stops not when it has read enough, but when it can
enumerate what it **couldn't** determine. Gaps are first-class outputs, not failures.

In Phase 3, these gaps become `AskUserQuestion` tool calls — structured interviews with 1-4
questions per gap, with preview content so the user sees what the agent found before answering.

**For Cartographer**: Phase 2 INVESTIGATE already has a `gaps` field. But Phase 1 MAP should
also surface gaps (e.g., "the entry point is ambiguous — README mentions X but Y is also called
at startup"). Currently Phase 1 just silently picks one.

---

## 5. The Explore Agent (Subagent used by /init Phase 2)

**From `src/tools/AgentTool/built-in/exploreAgent.ts`**:

```
STRICT READ-ONLY MODE — NO FILE MODIFICATIONS
Allowed tools: GlobTool, GrepTool, FileReadTool, BashTool (read-only)
Forbidden:     FileEditTool, FileWriteTool, NotebookEditTool

Model:         haiku (speed-optimized for exploration)
Min queries:   3 (won't DONE after a single lookup)
Optimization:  parallel tool invocation where possible
```

**Tool signatures**:
- `glob(pattern, path?)` → file paths matching pattern
- `grep(pattern, path?, glob?, type?, output_mode?)` → matches or file list
- `bash(command)` — read-only: ls, git log, git diff, find, cat, head, tail

**Key design**: model is haiku (fast + cheap), not the primary model. Exploration calls are
high-volume/low-value — each one is a navigation step. The expensive primary model is reserved
for the synthesis step (Phase 4-5). Same principle we use in Cartographer: Gemini for synthesis,
SambaNova as fallback for the heavy calls.

---

## 6. Stopping Criterion — Goal-Driven, Not Round-Limited

There is **no explicit round limit** in the Explore agent or in Phase 2's instructions.
Stopping is triggered by completion of the signal checklist + gap enumeration.

Our current Phase 1 has `max_rounds = 16` as a hard ceiling. The ceiling is necessary (infinite
loop protection) but the *primary* stopping signal should be: "I have read at least one file
from every non-trivial directory AND I can name what I could not determine from code alone."

The round limit should be a safety net, not a target.

---

## 7. What /init Does NOT Do (also instructive)

From Phase 2 explicit exclusions:
> "Do NOT include: file-by-file structure or component lists — Claude can discover these by
> reading the codebase."

This tells us: the goal of exploration is **compressed understanding**, not enumeration.
A tour that lists every file is useless. A tour that names 6 design decisions is gold.

Also: /init avoids "explaining what every file does" — it explains what the *system* does,
and names the specific gotchas a new contributor must know.

---

---

## 9. Phases 3–8 (Synthesis, Skills, Hooks — full picture)

**Phase 2 → synthesis is DIRECT — no per-component deep dive.**
After the single Explore sweep, Claude Code goes straight to Phase 3 (user interview) then writes.
There is NO equivalent of Cartographer's Phase 2 INVESTIGATE × N.

**Why Cartographer's approach is more thorough**: /init writes a *practical* CLAUDE.md (build commands,
code style, gotchas). Cartographer writes a *conceptual tour* (architectural decisions, design tradeoffs).
The tour requires understanding *why* decisions were made, which demands per-component depth — hence
our Phase 2 INVESTIGATE is justified and correct.

**Phase 3 — Preference Queue pattern (structurally valuable)**
Phase 3 builds a typed queue: `{type: 'hook'|'skill'|'note', target: 'project'|'personal', details}`.
Phases 4-7 consume this queue in order.
Cartographer analogy: Phase 1 produces `pipeline_stages`, Phase 2 consumes them. Same pattern.

**Phase 4 — Synthesis quality filter (verbatim, important)**
> "Every line must pass this test: 'Would removing this cause Claude to make mistakes?' If no, cut it."

Cartographer equivalent for tour cards:
> "Would removing this card leave a gap in understanding the system's core value?"
This is the right quality test for our evaluator pass.

**Phase 5 — NO automated self-critique.**
There is no self-critique phase in Claude Code. Quality review is user-driven via `AskUserQuestion`
previews in Phase 3. Cartographer's `_validate_concepts` evaluator-optimizer is MORE sophisticated.

**Phase 6 — Skills**: Creates `.claude/skills/<name>/SKILL.md` files. Not relevant to Cartographer.

**Phase 7 — Environment-aware suggestions** (GitHub CLI, linting, hooks). Not relevant.

**Phase 8 — Summary + to-do list.** Not relevant.

---

## 10. Six Built-In Agent Types

| Agent | Model | Tools | Purpose |
|---|---|---|---|
| **Explore** | haiku | Glob, Grep, Read, Bash (read-only) | Broad codebase survey — no writes |
| **General-Purpose** | inherit | All (`*`) | Multi-step research + code search |
| **Plan** | inherit | Read-only (no Edit/Write) | Architecture + implementation planning |
| **Verification** | inherit | Build/test tools, no Edit/Write | "Try to break it" — adversarial testing |
| **Claude Code Guide** | haiku | Fetch, Search | Answers questions about Claude Code itself |
| **Status Line Setup** | inherit | Read, Edit | Configures status line UI |

**Key design**: haiku for read-heavy navigation (Explore, Guide), inherit for deep reasoning (Plan, Verify).
The expensive model is reserved for synthesis and evaluation — cheap model for enumeration.

Cartographer currently uses Gemini for ALL phases. Worth noting that a tiered model approach
(fast model for Phase 1 tool calls, strong model for Phase 3 synthesis) could improve speed.

---

## 11. Verification Agent — Most Relevant to Our Evaluator

From `verificationAgent.ts` system prompt (verbatim):
> "Your job is not to confirm the implementation works — it's to try to BREAK it."
> Runs: builds, tests, linters, plus adversarial probes (concurrency, boundary values, idempotency).
> Output format: `VERDICT: PASS | FAIL | PARTIAL` with specific failure evidence.

**For Cartographer's `_validate_concepts`**: we use `PASS/FIXED` not `PASS/FAIL/PARTIAL`.
Adding `PARTIAL` (some concepts pass, some need renames, some need removal) would give finer signal
and avoid over-removing good concepts when only one is bad.

---

## 12. Implications for Cartographer — Applied Changes

| What /init does | What we applied |
|---|---|
| Manifest-first | Already done — `_manifest_chunks()` reads package.json/Cargo.toml/etc. first |
| Named detection targets, not round count | **APPLIED**: replaced "5-8 decisions" with 5-signal checklist (entry point, primary technique, key deps, directory breadth, gaps) |
| "Note what code alone couldn't tell you" | **APPLIED**: added `gaps` field to DONE JSON; surfaced in Phase 3 prompt as `gaps_section` for `ask` fields |
| Parallel exploration where possible | Phase 2 already uses ThreadPoolExecutor; Phase 1 is sequential by design (ReAct chain) |
| haiku for exploration, strong model for synthesis | Phase 1 currently uses primary model (Gemini) — could optimise but not a blocking issue |
| Min 3 queries before stopping | Enforced by DONE SIGNAL ④: "at least one file READ from every non-trivial directory" |

**NOT adopted (intentional)**:
- No per-component deep dive in /init: Cartographer's Phase 2 INVESTIGATE × N is MORE thorough by design. Tours require architectural depth; CLAUDE.md requires breadth only.
- No user interview phase: Cartographer is fully automated. /init's Phase 3 interview fills gaps via user input; we surface gaps in the tour's `ask` fields instead.

**Verification prompt quality test (from Phase 4's minimalist principle)**:
For each tour concept card, ask: "Would removing this card leave a gap in understanding the system's core value?"
If no → the evaluator should remove it. This is the right quality criterion for `_validate_concepts`.

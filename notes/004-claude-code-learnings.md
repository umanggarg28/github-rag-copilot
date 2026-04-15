# Claude Code Source Study — Learnings for Cartographer

Reference session: April 2026
Source: https://github.com/codeaashu/claude-code (reconstruction/tutorial repo)
Reference: anthropics/anthropic-cookbook patterns/agents/

---

## 1. `/init` Phase 2 — Manifest Files Before Code

**What claude-code does:**
> "Launch a subagent to survey the codebase, and ask it to read key files:
> manifest files (package.json, Cargo.toml, pyproject.toml, go.mod, pom.xml, etc.),
> README, Makefile/build configs..."

**Why this matters for cartographer:**
Manifest files are the universal, language-agnostic entry point to any repo:
- They declare **dependencies** → immediately reveals tech stack (fastapi = web API,
  torch = ML, tree-sitter = code parsing, no framework = pure library like micrograd)
- They declare **entry points/scripts** → reveals how the system is run
- They work for ANY repo: web apps, ML libraries, compilers, game engines, CLIs

**What we were doing wrong:**
Phase 1 was reading `main.py`/`app.py` module chunks — these are bootstrap files
that import ALL features equally. `main.py` in a FastAPI app imports every router,
every service, every feature. Its import graph says "everything is equally important",
which is the opposite of the pipeline signal we need.

**Fix applied:**
Added `_manifest_chunks()` to read project manifests first in Phase 1. The LLM
anchors on declared dependencies → understands project type → identifies pipeline
stages correctly.

---

## 2. No Hardcoded Heuristics — Principles Over Examples

**What claude-code does:**
The `/init` prompt states principles and asks the subagent to DISCOVER, not guess:
- "Detect: Build, test, and lint commands (especially non-standard ones)"
- "Note what you could NOT figure out from code alone — these become interview questions"

It never says "if you see a `routers/` directory, skip it" or "if you see `ingestion/`,
it's a pipeline stage". The agent reads actual file content and reasons from there.

**What we were doing wrong:**
Our Phase 1 prompt was full of domain-specific terms:
- "ingestion, parsing, embedding, retrieval, inference" — only valid for LLM/RAG apps
- "routers, routes, middleware, handlers" — only valid for web apps
- Good/bad examples like "Gradient Backpropagation" or "Token Embedding"

These break silently on any non-web, non-LLM repo.

**Fix applied:**
Phase 1 prompt now:
1. Reads manifest → understands tech stack from dependencies
2. Reads README → understands what the system does
3. Reads module-level imports → sees what each file ACTUALLY uses
4. States rules as **universal principles**: "a stage takes data in one form and
   produces it in another — evident from its imports and function signatures"

No domain terms, no directory name assumptions, no illustrative examples.

---

## 3. Evaluator-Optimizer Pattern (Anthropic Cookbook)

**From `patterns/agents/evaluator_optimizer.ipynb`:**
```python
def evaluate(prompt, content, task):
    # Returns <evaluation>PASS|NEEDS_IMPROVEMENT|FAIL</evaluation>
    # AND <feedback>specific actionable feedback</feedback>

def loop(task, evaluator_prompt, generator_prompt):
    while True:
        evaluation, feedback = evaluate(evaluator_prompt, result, task)
        if evaluation == "PASS":
            return result
        context = "Previous attempts: [memory] Feedback: [feedback]"
        result = generate(generator_prompt, task, context)
```

**Key properties:**
1. **Feedback accumulates across rounds** — each round gets context of what was tried
2. **Clear pass/fail criteria** — universal principles, not content-specific examples
3. **Evaluation and generation are separate concerns** — different prompts, different roles

**What we were doing wrong:**
Our evaluator's "remove trivial infrastructure" instruction had no corresponding
response format — the LLM said "remove" but didn't know HOW to signal removal.
Result: "Noise File Exclusion" passed because it sounds like a technique name.

**Fix applied:**
Explicit `action: keep|rename|remove` field. Three-test rubric for quality:
1. Is it a technique/decision name (not a filename/class name)?
2. Would removing it leave an engineer unable to understand the system's core behaviour?
3. Does the subtitle describe a real design decision with tradeoffs?

---

## 4. Tool Design — `get_architecture` vs Import Graph Reading

**What claude-code does:**
Exposes `get_architecture` as a curated tool that provides high-level overview.
Also `list_directory`, `search_source`, `read_source_file` for targeted exploration.

The agent DECIDES what to look at. It doesn't get a static snapshot of everything.

**What we were doing wrong:**
Phase 1 gave the LLM a static 80-file flat list + 14 random module chunks and
asked it to infer the pipeline. No structure, no priority, no anchoring.

**Fix applied:**
- Files grouped by directory (structural signal)
- Manifest files first (tech stack signal)
- Module chunks from non-bootstrap files only
- README as anchor for what capabilities must appear

**Still room to improve:**
The right long-term fix is to make Phase 1 truly agentic — give it tools
(list_files, search_code, read_file) and let it explore rather than giving
it a pre-assembled snapshot. This is how claude-code's `/init` actually works:
a subagent with tools reads specific files, not a single-shot prompt.

---

## 5. Bootstrap Files Are Wiring, Not Pipeline

**Universal pattern (any framework):**
- `main.py`, `app.py`, `server.py` — Python web apps (FastAPI, Flask, Django)
- `index.js`, `server.js` — Node.js apps (Express, Fastify)
- `main.go` — Go apps
- `main.rs` — Rust apps (Actix, Axum)
- `Application.java` — Spring Boot

These files wire together all features/routers/services. They import everything.
Their import graph tells you "all features exist" — not which ones are the core pipeline.

**Fix applied:**
Removed bootstrap filenames from `_ENTRY_NAMES`. Phase 1 now gets module chunks
from service/library files that actually DO the work.

---

## 6. What /init's "Note what you could NOT figure out" means for tours

The `/init` prompt says: "Note what you could NOT figure out from code alone —
these become interview questions."

This is already implemented in Phase 2 (`_phase_investigate`) via the `gaps` field:
> "GAPS: What important design rationale CANNOT be determined from this code alone?"

The `ask` field on each concept card also embodies this: it surfaces the question
a new engineer MUST understand to work with this component, often something not
visible in the code.

**This is the right model.** Keep it.

---

## 7. Contextual Retrieval Prompt Quality

Current `_CONTEXT_SYSTEM`:
> "Write 1-2 sentences that situate this chunk within the document: name the
> function/class, state its role in the file's pipeline, and name the key
> identifier(s) a developer would search for."

This is solid. The system prompt correctly tells the model its output is prepended
to chunks for embedding — it must match developer search queries, not explain failure modes.

The `chunk_question` correctly uses Anthropic's prompt caching pattern:
- Document block → `cache_control: ephemeral` (cached per file, ~10% cost for subsequent chunks)
- Chunk + question → varies per chunk (not cached)

**No changes needed here** — contextual retrieval was not touched in this session.

---

## 8. Remaining Quality Issues to Fix

From the latest tour output:
- "In-Memory Archive Extraction" from `qdrant_store.py` with garbled description
  → Phase 2 hallucinating about a storage layer file. Fix: Phase 2 investigation
  prompt needs stronger grounding: "only use information visible in the code above"
- Some concept descriptions still feel thin
- The `ask` questions could be sharper — they should be answerable ONLY by
  someone who read the specific concept, not generic questions

**Next session priorities:**
1. Fix Phase 2 hallucination on storage/infrastructure files
2. Improve `ask` question specificity
3. Consider agentic Phase 1 (tool-based exploration instead of static snapshot)

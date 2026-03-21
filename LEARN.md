# GitHub RAG Copilot — Learning Guide

This document grows with the project. Each section is added when the
corresponding feature is built. Read it alongside the code and `notes/` entries.

---

# Table of Contents

1. [What is RAG on Code?](#1-what-is-rag-on-code)
2. [AST-Based Chunking](#2-ast-based-chunking) ← _coming in Phase 1_
3. [Code Embeddings](#3-code-embeddings) ← _coming in Phase 1_
4. [Qdrant — A Hosted Vector Database](#4-qdrant) ← _coming in Phase 1_
5. [Native Hybrid Search in Qdrant](#5-native-hybrid-search) ← _coming in Phase 2_
6. [Generation for Code Queries](#6-generation-for-code-queries) ← _coming in Phase 2_
7. [Live Deployment](#7-live-deployment) ← _coming in Phase 4_
8. [Claude Code Features](#8-claude-code-features) ← _built throughout_
   - 8a. CLAUDE.md
   - 8b. Slash Commands
   - 8c. Hooks
   - 8d. Subagents

---

# 1. What is RAG on Code?

## The core idea

RAG stands for Retrieval-Augmented Generation. The pipeline has two phases:

**Ingestion** (done once, or when a repo is added):
```
Source → extract text → chunk → embed → store in vector DB
```

**Query** (done on every question):
```
Question → embed → search vector DB → retrieve relevant chunks → LLM → answer
```

For this project:
```
Ingestion:  GitHub repo → files → code chunks → embed → Qdrant Cloud
Query:      Question → embed → Qdrant hybrid search → retrieved chunks → Groq/Claude → answer with citations
```

## Why does RAG exist?

LLMs are trained on a static snapshot of the internet up to a cutoff date.
They cannot know about:
- Private repositories
- Repos that changed after their training cutoff
- Your specific codebase

Without RAG, asking "how does authentication work in this repo?" forces the
LLM to either guess (hallucinate) or say "I don't know." With RAG, the
relevant code is retrieved and placed directly in the prompt — the LLM reads
your actual code and answers from it.

## What makes code RAG different from document RAG?

Three things change when the source is code instead of documents:

**1. What you index**

A research paper contains only relevant content. A GitHub repo contains:
- Source code you care about (`.py`, `.ts`, `.go`, etc.)
- Auto-generated files you don't care about (`package-lock.json`, `*.lock`)
- Dependency directories with thousands of files (`node_modules/`, `.venv/`)
- Build output (`dist/`, `__pycache__/`)

Indexing everything would drown relevant code in noise. You need a
`file_filter.py` with explicit rules about what to include.

**2. How you chunk**

A research paper can be split anywhere — even mid-sentence, the surrounding
context still makes sense to an LLM. Code is structured differently:

```python
def embed_text(self, text: str) -> list[float]:
    """Embed a single text string into a vector."""
    tokens = self.tokenizer.encode(text)
    return self.model(tokens).pooled_output.tolist()
```

Split mid-function and you lose either:
- The signature (what it accepts, what it returns) — can't answer "what does this take?"
- The body (what it actually does) — can't answer "how does this work?"

Solution: **AST-based chunking** — parse the code into its syntax tree and
split at natural boundaries (functions, classes). Each chunk is a complete,
self-contained unit of code.

**3. What metadata matters**

For documents: `source` (paper name), `page` (page number)

For code: `repo`, `filepath`, `language`, `function_name`, `class_name`,
`start_line`, `end_line`

This makes citations meaningful:
```
Code: torch/nn/functional.py — scaled_dot_product_attention() — lines 4823–4891
```

And enables powerful filters you can't do with documents:
- "Only search in test files"
- "Only search in the `auth/` directory"
- "Only search in Python files"

---

# 8. Claude Code Features

This section explains how to use Claude Code effectively while building this
project — it's a meta-layer that makes you faster and more consistent.

## 8a. CLAUDE.md

`CLAUDE.md` is a file at the project root that Claude Code reads at the start
of every session. It's a permanent briefing document.

**Without CLAUDE.md:** Claude rediscovers the project structure on every
session by reading files and asking questions — costing time and tokens.

**With CLAUDE.md:** Claude immediately knows the architecture, conventions,
environment variables, and key design decisions. It can start working without
a warm-up phase.

**What to put in CLAUDE.md:**
```markdown
- Directory structure (quick mental map)
- How to run the project
- Environment variables required
- Coding conventions (e.g. "no LangChain", "write docstrings")
- Key decisions and WHY they were made
- Available slash commands
```

**What NOT to put in CLAUDE.md:**
- Detailed implementation docs (put those in code comments)
- Anything that changes frequently (stale info wastes tokens)
- Things Claude can infer from reading the code

**Token cost:** Claude reads CLAUDE.md on every message. Keep it concise —
every unnecessary line is paid for in tokens across hundreds of messages.

Open `CLAUDE.md` at the project root to see this project's example.

## 8b. Slash Commands

Slash commands are custom prompts stored in `.claude/commands/*.md`.
Invoke them with `/command-name [args]`.

```
.claude/commands/
    ingest-repo.md    →  /ingest-repo https://github.com/owner/repo
    search-code.md    →  /search-code "how does auth work"
    add-to-notes.md   →  /add-to-notes
```

Each file is a markdown prompt. `$ARGUMENTS` is replaced with whatever
you type after the command name.

**Why slash commands?**

Instead of typing:
> "Run the ingestion pipeline for this GitHub URL, filter files by language,
> chunk with AST, embed, upsert to Qdrant, then print a summary of files
> indexed, languages detected, total chunks stored, and any files skipped."

You type: `/ingest-repo https://github.com/owner/repo`

The command encodes the full instruction once. Every subsequent invocation
is consistent — you don't accidentally forget to ask for the summary, or
phrase the instruction slightly differently each time.

**When to create a slash command:**
- Any operation you do more than twice
- Multi-step workflows that need to be consistent
- Instructions that are long or easy to get wrong

## 8c. Hooks

Hooks are shell commands that run automatically in response to Claude Code
events. They're defined in `.claude/settings.json`.

**Hook events:**
- `PreToolUse` — before Claude calls a tool (e.g. before writing a file)
- `PostToolUse` — after Claude calls a tool (e.g. after editing a file)
- `Stop` — when Claude finishes responding

**Example: auto-lint after every Python file edit**
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "ruff check $CLAUDE_FILE_PATH --fix --quiet"
      }]
    }]
  }
}
```

Now every file Claude edits is automatically linted and auto-fixed. You
never commit code with style errors, and you don't have to remember to run
the linter.

**Example: remind Claude to update notes after a commit**
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "echo '$CLAUDE_TOOL_INPUT' | grep -q 'git commit' && echo 'REMINDER: run /add-to-notes' || true"
      }]
    }]
  }
}
```

**Why hooks matter:**
They enforce quality gates without relying on Claude (or you) to remember.
A lint hook means the gate is always applied — it's not optional, it doesn't
get skipped when you're in a hurry. Think of them as `pre-commit` hooks but
triggered by Claude's actions rather than git commands.

## 8d. Subagents

Subagents are Claude instances spawned to handle independent tasks in parallel
or in isolation. In Claude Code, you use the `Agent` tool.

**Why subagents?**

Two reasons:

1. **Parallelism** — independent tasks run simultaneously instead of sequentially.
   Ingesting three repos takes 2 minutes instead of 6.

2. **Context isolation** — a subagent does its work in a separate context window.
   If you ask a subagent to explore a large repo, it reads thousands of lines
   without filling the main context. Only the summary comes back.

**Subagent types:**
- `general-purpose` — full tool access, can read/write/run commands
- `Explore` — read-only, optimised for fast codebase exploration
- `Plan` — architecture and design, returns structured plans

**Patterns in this project:**

*Parallel ingestion:*
```
Main agent spawns:
  ├── Subagent A: ingest pytorch/pytorch
  ├── Subagent B: ingest huggingface/transformers
  └── Subagent C: ingest openai/openai-python
All three run simultaneously → done in ~2 min instead of ~6 min
```

*Pre-PR review:*
```
Before committing, spawn an expert subagent:
  "Review the ingestion pipeline for correctness, edge cases, and clarity.
   Rate it and list specific issues."
Main agent gets back a structured review without reading all the files again.
```

*Repo exploration before indexing:*
```
Before ingesting an unfamiliar repo, spawn an Explore subagent:
  "Read the repo structure and README. What are the main modules?
   What languages are used? What should be filtered out?"
Main agent uses the summary to configure ingestion — no need to read everything.
```

---

_Sections 2–7 will be added as each phase is built._

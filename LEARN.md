# GitHub RAG Copilot — Learning Guide

This document grows with the project. Each section is added when the corresponding
feature is built. Read it alongside the code and the `notes/` entries.

---

# Table of Contents

1. [Why Code RAG is Different from Document RAG](#1-why-code-rag-is-different)
2. [AST-Based Chunking](#2-ast-based-chunking) ← _coming in Phase 1_
3. [Code Embeddings](#3-code-embeddings) ← _coming in Phase 1_
4. [Qdrant — A Hosted Vector Database](#4-qdrant) ← _coming in Phase 1_
5. [Native Hybrid Search in Qdrant](#5-native-hybrid-search) ← _coming in Phase 2_
6. [Generation for Code Queries](#6-generation-for-code-queries) ← _coming in Phase 2_
7. [Claude Code Features](#7-claude-code-features) ← _built throughout_
   - 7a. CLAUDE.md
   - 7b. Slash Commands
   - 7c. Hooks
   - 7d. Subagents

---

# 1. Why Code RAG is Different

## The same idea, different inputs

In the PDF RAG Copilot, the pipeline was:
```
PDF → pages → text chunks → embed → ChromaDB → query → answer
```

In this project it's:
```
GitHub repo → files → code chunks → embed → Qdrant → query → answer
```

The retrieval, LLM generation, and API layers are nearly identical.
The differences are all in **ingestion** — how you get the text, and how you
chunk it.

## Problem 1: What files do you index?

A GitHub repo contains many things that shouldn't be indexed:
- **Binary files** — images, compiled artifacts, `.pyc` files
- **Auto-generated files** — `package-lock.json`, `*.lock`, migration files
- **Dependency directories** — `node_modules/`, `.venv/`, `vendor/`
- **Build output** — `dist/`, `build/`, `__pycache__/`

If you index these, queries like "how does authentication work?" return
lock file entries and compiled output instead of actual code.

Solution: a `file_filter.py` with explicit include/exclude rules per language.

## Problem 2: How do you chunk code?

In the PDF project, we used **fixed character windows** with overlap:
```
[---- chunk 1 (800 chars) ----]
              [---- chunk 2 (800 chars) ----]
```

This works for prose because a sentence is a self-contained unit — splitting
a 200-page paper anywhere still yields readable text.

**Code is different.** A function is the natural unit:
```python
def embed_text(self, text: str) -> list[float]:
    """Embed a single text string into a 384-dim vector."""
    return self.model.encode([text])[0].tolist()
```

Splitting this mid-way loses the function signature (what it takes/returns)
or the body (what it actually does). A chunk without the signature can't
answer "what does embed_text take as input?". A chunk without the body can't
answer "how does embed_text work?".

**Solution: AST-based chunking** — parse the code into its syntax tree, then
use function and class boundaries as natural split points.

## Problem 3: What metadata matters?

PDF RAG metadata: `source` (paper name), `page` (page number)
Code RAG metadata: `repo`, `filepath`, `language`, `function_name`,
                   `class_name`, `start_line`, `end_line`

This makes citations meaningful:
```
PDF:  (Source: attention_2017, Page 4)
Code: (repo: pytorch/pytorch, file: torch/nn/functional.py, lines 1823–1851)
```

And it enables powerful filters:
- "Only search in Python files"
- "Only search in the `auth/` directory"
- "Only search in test files"

## What stays the same

Everything downstream of ingestion:
- Embedding queries and comparing to stored vectors
- Relevance threshold to reject out-of-domain queries
- Hybrid search combining semantic + keyword
- LLM generation from retrieved context
- Citations, confidence scores, streaming

---

# 7. Claude Code Features

This section is different from the rest — instead of explaining RAG concepts,
it explains how to use Claude Code more effectively while building this project.

## 7a. CLAUDE.md

`CLAUDE.md` is a file Claude Code reads at the start of every session.
Think of it as a briefing document for Claude — it tells Claude:
- What the project does
- How the codebase is structured
- Coding conventions to follow
- What commands to run
- Key design decisions that shouldn't be changed without thought

Without `CLAUDE.md`, Claude starts every session with no project context and
has to rediscover everything from reading files. With it, Claude immediately
knows the architecture, conventions, and constraints.

**Best practices for CLAUDE.md:**
- Keep it concise — Claude reads it on every message, so every line costs tokens
- Include the directory structure (a quick mental map)
- List environment variables needed
- Document non-obvious decisions and _why_ they were made
- Keep commands up-to-date (wrong commands waste time)

**What NOT to put in CLAUDE.md:**
- Detailed implementation explanations (that's what code comments are for)
- Anything that changes frequently (stale info is worse than no info)
- Things derivable from the code itself

Our `CLAUDE.md` lives at the root of the project. Open it to see an example.

## 7b. Slash Commands

Slash commands are custom prompts stored in `.claude/commands/`.
They're invoked with `/command-name [args]`.

```
.claude/
  commands/
    ingest-repo.md    → /ingest-repo <github-url>
    search-code.md    → /search-code <query>
    add-to-notes.md   → /add-to-notes
```

Each command file is a markdown prompt with `$ARGUMENTS` as the placeholder
for whatever you pass after the command name.

**Why slash commands?**
Instead of typing a long, precise instruction every time ("run the ingestion
pipeline for this repo, print a summary of files indexed, languages detected,
and chunks stored"), you define it once and invoke it with `/ingest-repo <url>`.

They're especially useful for:
- Repetitive operations (ingest a repo, update notes, run tests)
- Multi-step workflows you want to be consistent
- Sharing workflows with others on the project

## 7c. Hooks

Hooks are shell commands that run automatically in response to Claude Code events.
Configure them in `.claude/settings.json`.

Available hook events:
- `PreToolUse` — runs before Claude calls a tool (e.g., before editing a file)
- `PostToolUse` — runs after Claude calls a tool (e.g., after writing a file)
- `Stop` — runs when Claude finishes a response

**Example: auto-lint after every file edit**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "cd /path/to/project && ruff check $CLAUDE_FILE_PATH --fix"
        }]
      }
    ]
  }
}
```

**Example: auto-update notes after a commit**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "if echo '$CLAUDE_TOOL_INPUT' | grep -q 'git commit'; then echo 'Remember to run /add-to-notes'; fi"
        }]
      }
    ]
  }
}
```

**Why hooks?**
They automate quality gates without relying on Claude to remember to run them.
A lint hook means every file Claude edits is checked — you never accidentally
commit code with style errors.

## 7d. Subagents

Subagents are Claude instances spawned from the main Claude session to handle
independent tasks in parallel. In Claude Code, you use the `Agent` tool.

**When to use subagents:**
- Tasks that are independent and can run simultaneously
- Tasks that would pollute the main context with large outputs
- Specialised tasks (research, exploration, review)

**Example: parallel repo ingestion**
Instead of ingesting repos sequentially:
```
Ingest repo A (2 min) → Ingest repo B (2 min) → Ingest repo C (2 min) = 6 min
```

Spawn three subagents in parallel:
```
Ingest repo A ─┐
Ingest repo B ─┼→ done in ~2 min
Ingest repo C ─┘
```

**Example: exploration agent**
Before implementing a feature, spawn an Explore agent to read all relevant
files and return a summary — without filling the main context with file contents.

**Subagent types in Claude Code:**
- `general-purpose` — full tool access, good for implementation tasks
- `Explore` — read-only, fast codebase exploration
- `Plan` — architecture and design planning

We'll use subagents when:
1. Ingesting multiple repos simultaneously
2. Running an expert review before a PR
3. Exploring unfamiliar repos before answering questions about them

---

_Sections 2–6 will be added as each phase is built._

"""
mcp_server.py — FastMCP server exposing code-search capabilities via MCP protocol.

═══════════════════════════════════════════════════════════════
WHAT IS AN MCP SERVER?
═══════════════════════════════════════════════════════════════

An MCP server is a process that exposes capabilities (tools, data, prompts)
to any MCP-compatible client. Instead of hardcoding tool logic inside the agent,
we publish it here — and ANY client that speaks MCP can use it automatically.

This is the same pattern Claude Code uses: it connects to MCP servers at startup,
discovers their tools, and calls them during conversations. Our agent does the
same thing, but WE built both sides.

THREE MCP PRIMITIVES:

  Tools (model-controlled)
  ─────────────────────────
  Functions the LLM decides to call. The LLM reads the tool description and
  decides WHEN and HOW to use it. This is the core of agentic behavior.
  Examples: search_code, find_callers, get_file_chunk

  Resources (application-controlled)
  ────────────────────────────────────
  Read-only data the application exposes for the LLM to read as context.
  Identified by URI (like a URL). The client decides when to include them.
  Examples: qdrant://repos (list repos), qdrant://repos/karpathy/micrograd (browse)

  Prompts (user-controlled)
  ─────────────────────────
  Reusable prompt templates the USER explicitly invokes (like slash commands).
  They produce structured messages to inject into the conversation.
  Examples: /analyze_repo, /explain_function

═══════════════════════════════════════════════════════════════
TRANSPORT: STREAMABLE HTTP
═══════════════════════════════════════════════════════════════

We use the Streamable HTTP transport (the production-grade MCP transport):
  - Client sends JSON-RPC 2.0 messages as HTTP POST to /mcp
  - Server responds with JSON (simple) or SSE stream (long operations)
  - Session management via Mcp-Session-Id header

vs Stdio transport (for local CLI tools):
  - Client spawns server as subprocess
  - Communication over stdin/stdout
  - Simpler but can't serve multiple clients

We use HTTP because our server is already a FastAPI process and we want
multiple clients to be able to connect simultaneously.

═══════════════════════════════════════════════════════════════
MOUNTING IN FASTAPI
═══════════════════════════════════════════════════════════════

FastMCP.streamable_http_app() returns a Starlette ASGI app.
We mount it inside our existing FastAPI app:

  app.mount("/mcp", mcp.streamable_http_app())

Now the MCP endpoint lives at http://localhost:8000/mcp alongside all
the REST endpoints (/ingest, /query, /agent/stream, etc.).
One process, two protocols.
"""

import sys
from pathlib import Path
from typing import Optional

import requests as http_requests
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings


# ── Service holders ────────────────────────────────────────────────────────────
# These are module-level singletons set during FastAPI lifespan (see main.py).
#
# Why not inject via constructor?
# FastMCP tools are registered at import time with @mcp.tool().
# The decorated functions close over these module-level variables.
# Python closures capture by reference — when a tool is called, it reads the
# CURRENT value of _retrieval (set during lifespan), not the None at import time.
#
# This is the same pattern used in main.py for _ingestion_service etc.

_retrieval = None  # RetrievalService — set by init_services()
_store = None      # QdrantStore       — set by init_services()


def init_services(retrieval_service, qdrant_store):
    """
    Inject shared service instances into the MCP server.

    Called from main.py lifespan after the embedding model has loaded
    and Qdrant connection is established. This avoids loading the
    600MB model twice (once in FastAPI, once in the MCP server).
    """
    global _retrieval, _store
    _retrieval = retrieval_service
    _store = qdrant_store
    print("MCP server: services injected.")


# ── FastMCP instance ───────────────────────────────────────────────────────────
# FastMCP is the high-level Python SDK for building MCP servers.
# It handles: JSON-RPC 2.0 protocol, tool/resource/prompt registration,
# transport setup, session management, and capability negotiation.
#
# streamable_http_path="/" means the MCP endpoint within the sub-app is at "/"
# So when mounted at "/mcp" in FastAPI, the full URL is: /mcp  (not /mcp/mcp)

mcp = FastMCP(
    name="github-rag-copilot",
    instructions=(
        "Code search server for indexed GitHub repositories. "
        "Use search_code to find relevant code by concept or identifier. "
        "Use find_callers to understand how a function is used. "
        "Use get_file_chunk to read raw source lines for context. "
        "Always search before answering questions about a codebase."
    ),
    streamable_http_path="/",
    # stateless_http=True: each request is independent — no persistent sessions.
    # Required when mounting inside FastAPI (app.mount), because the sub-app's
    # lifespan doesn't automatically run the session manager task group.
    # Our MCPClient uses one connection per call anyway, so stateless is correct.
    stateless_http=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — functions the LLM calls to take action
#
# The description is what the LLM reads to decide WHEN to call the tool.
# Be precise: what it does, when to use it, and when NOT to use it.
# Type annotations in the function signature become the tool's JSON Schema.
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_code(
    query: str,
    repo: Optional[str] = None,
    mode: str = "hybrid",
) -> str:
    """
    Search the indexed GitHub repositories for code relevant to a query.

    Uses hybrid BM25 + semantic search with RRF fusion by default.
    Returns ranked code chunks with file paths, function names, and line numbers.

    Call this first when answering any question about the codebase.
    Call multiple times with different queries for broader coverage.

    Args:
        query: Natural language description or code identifier to search for
        repo:  Optional 'owner/repo' to restrict search (e.g. 'karpathy/micrograd')
        mode:  'hybrid' (default) | 'semantic' (concepts) | 'keyword' (exact names)
    """
    if _retrieval is None:
        return "Search service not ready — backend still initializing."

    results = _retrieval.search(
        query=query,
        top_k=6,
        repo_filter=repo,
        mode=mode,
    )
    if not results:
        return f"No results found for: '{query}'"
    return _retrieval.format_context(results)


@mcp.tool()
def find_callers(function_name: str, repo: Optional[str] = None) -> str:
    """
    Find all places in the codebase that call a specific function or class.

    Essential for understanding HOW something is used, not just what it does.
    Use this after search_code when you need usage patterns and call sites.

    Uses the 'calls' payload field populated during AST chunking — this is
    a structural lookup, not text search, so it finds exact call sites only.

    Args:
        function_name: The exact function or class name to find callers of
        repo:          Optional 'owner/repo' to restrict search
    """
    if _store is None:
        return "Search service not ready."

    callers = _store.find_callers(function_name, repo=repo)
    if not callers:
        return f"No call sites found for '{function_name}' in the 'calls' index."

    # Format the same way as retrieval.format_context for consistency
    parts = []
    for i, c in enumerate(callers[:8], 1):
        citation = c.get("filepath", "")
        if c.get("name"):
            citation += f" — {c['name']}()"
        citation += f" | lines {c.get('start_line', '?')}–{c.get('end_line', '?')}"
        parts.append(f"[Source {i} | {c.get('repo', '')} | {citation}]\n{c.get('text', '')}")

    return f"Found {len(callers)} caller(s) of '{function_name}':\n\n" + \
           "\n\n" + "─" * 40 + "\n\n".join(parts)


@mcp.tool()
def get_file_chunk(
    repo: str,
    filepath: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    Fetch raw source lines from a file in a GitHub repository.

    Use when search returns a function but you need more context:
    the surrounding class, the docstring, or what comes right after.

    Args:
        repo:       'owner/repo' (e.g. 'karpathy/micrograd')
        filepath:   Path within the repo (e.g. 'micrograd/engine.py')
        start_line: First line to fetch, 1-indexed
        end_line:   Last line to fetch, inclusive
    """
    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/contents/{filepath}"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    if settings.github_token:
        headers["Authorization"] = f"token {settings.github_token}"

    try:
        resp = http_requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return f"File not found: {filepath} in {repo}"
        resp.raise_for_status()
    except Exception as e:
        return f"GitHub fetch failed: {e}"

    lines = resp.text.splitlines()
    start = max(1, start_line)
    end   = min(len(lines), end_line)
    chunk = "\n".join(
        f"{i + start}: {line}"
        for i, line in enumerate(lines[start - 1:end])
    )
    return f"# {repo} — {filepath} (lines {start}–{end})\n\n{chunk}"


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES — read-only data the LLM can access as context
#
# Resources are identified by URI (Uniform Resource Identifier).
# The client reads them and injects their content into the LLM's context.
# Think of them as "live documents" — always reflecting current state.
#
# URI scheme we define: qdrant://repos  and  qdrant://repos/{slug}
# ══════════════════════════════════════════════════════════════════════════════

@mcp.resource("qdrant://repos")
def list_indexed_repos() -> str:
    """
    List all GitHub repositories currently indexed in the vector database.

    Returns repo slugs (owner/name) and chunk counts.
    Read this resource to discover what's available before searching.
    """
    if _store is None:
        return "Store not ready."

    repos = _store.list_repos()
    if not repos:
        return "No repositories indexed yet. Use POST /ingest to add one."

    lines = ["# Indexed Repositories\n"]
    for slug in repos:
        count = _store.count(repo=slug)
        lines.append(f"- **{slug}** — {count:,} chunks indexed")
    return "\n".join(lines)


@mcp.resource("qdrant://repos/{slug}")
def get_repo_index(slug: str) -> str:
    """
    Browse the indexed chunks for a specific repository.

    Returns all function and class names grouped by file.
    Use to understand a repo's structure before asking questions about it.

    Args:
        slug: 'owner/name' (forward slash URL-encoded as %2F by some clients)
    """
    if _store is None:
        return "Store not ready."

    slug = slug.replace("%2F", "/")
    chunks = _store.scroll_repo(
        repo=slug,
        with_payload=["filepath", "name", "chunk_type", "start_line"],
    )
    if not chunks:
        return f"No chunks found for '{slug}'. Re-ingest with POST /ingest."

    # Group by file
    by_file: dict[str, list[str]] = {}
    for chunk in chunks:
        fp   = chunk.get("filepath", "unknown")
        name = chunk.get("name", "")
        if name:
            by_file.setdefault(fp, []).append(
                f"  {chunk.get('chunk_type','?')}: {name}  (L{chunk.get('start_line','?')})"
            )

    lines = [
        f"# {slug} — Index Summary\n",
        f"**{len(chunks)} chunks** across {len(by_file)} files\n",
    ]
    for fp, items in sorted(by_file.items()):
        lines.append(f"\n### {fp}")
        lines.extend(items[:20])
        if len(items) > 20:
            lines.append(f"  … and {len(items) - 20} more")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS — reusable templates the USER explicitly invokes
#
# Prompts appear as slash commands in MCP clients (Claude Desktop, etc.).
# Unlike tools (model-driven), prompts are user-triggered.
# They return structured messages that get injected into the conversation.
#
# The returned string becomes a user message asking the LLM to do a task.
# ══════════════════════════════════════════════════════════════════════════════

@mcp.prompt()
def analyze_repo(repo: str) -> str:
    """
    Generate a structured analysis prompt for any indexed repository.

    Produces a multi-section prompt that guides the LLM to:
    1. Give a high-level overview of the project
    2. Identify entry points and key modules
    3. Trace the main data flow
    4. Highlight the 5 most important functions/classes

    Args:
        repo: 'owner/name' slug of an indexed repository
    """
    return f"""Perform a comprehensive analysis of the repository '{repo}'.

Use search_code to explore the codebase thoroughly. Structure your response:

## 1. Overview
What does this repo do? What problem does it solve? Who is the target user?

## 2. Architecture
What are the main modules/packages? How are they organized?
What are the key dependencies?

## 3. Entry Points
Where does execution begin? What are the public APIs or CLI commands?

## 4. Core Data Flow
How does data move through the system end-to-end?
What are the key transformations?

## 5. The 5 Most Important Functions/Classes
For each: name, file, what it does, and why it matters.

Start with search_code("overview {repo}"), then search_code("main entry point").
Cite all file paths and line numbers in your response."""


@mcp.prompt()
def explain_function(function_name: str, repo: Optional[str] = None) -> str:
    """
    Generate a deep-dive explanation prompt for a specific function or class.

    Guides the LLM to find the implementation, trace its callers,
    and explain both WHAT it does and WHY it's designed that way.

    Args:
        function_name: Name of the function or class to explain
        repo:          Optional 'owner/name' to restrict search scope
    """
    scope = f" in '{repo}'" if repo else ""
    return f"""Give a complete technical explanation of `{function_name}`{scope}.

Use the available tools:
1. search_code("{function_name}") — find the implementation
2. get_file_chunk(...) — get surrounding context if needed
3. find_callers("{function_name}") — find where it's used

Your explanation must cover:
- **What it does**: Plain English, one paragraph
- **Signature**: Parameters (types + purpose) and return value
- **Algorithm**: Step-by-step walkthrough of the logic
- **Usage patterns**: How and where it's called in the codebase
- **Design rationale**: Why is it implemented this way?

Be precise. Cite every file path and line number you reference."""

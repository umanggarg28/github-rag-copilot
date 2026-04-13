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

# ── Session working memory ─────────────────────────────────────────────────────
# Two-layer memory: in-process dict for fast same-session access, Qdrant for
# cross-session persistence.
#
# On each new query, clear_notes(repo) is called:
#   1. Clears the in-memory dict
#   2. Loads any previously persisted notes for this repo from Qdrant
#      so the agent can recall facts from prior sessions immediately
#
# note() writes to both layers.  recall_notes() reads only the in-memory dict
# (already warm from Qdrant load) — no per-call Qdrant round-trip.
#
# _current_repo tracks which repo's notes are loaded so note() knows where
# to persist without changing the tool's public interface.
_session_notes: dict[str, str] = {}
_current_repo:  str | None     = None


def clear_notes(repo: str | None = None) -> None:
    """
    Reset working memory and pre-load persisted notes for the given repo.

    Called by AgentService.stream() at the start of every new query.
    If repo is provided and a QdrantStore is available, existing notes
    for that repo are loaded so the agent has cross-session recall.
    """
    global _current_repo
    _session_notes.clear()
    _current_repo = repo

    # Pre-load persisted notes from Qdrant — agent can use recall_notes()
    # immediately without any extra round-trips.
    if repo and _store is not None:
        try:
            persisted = _store.load_notes(repo)
            _session_notes.update(persisted)
        except Exception as e:
            # Non-fatal: degraded to in-memory only if Qdrant is unavailable
            print(f"[notes] Could not load persisted notes for {repo}: {e}")


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
    name="cartographer",
    instructions=(
        "Code search server for indexed GitHub repositories. "
        "Use list_files to browse directory structure before diving in. "
        "Use search_code to find relevant code by concept or identifier. "
        "Use search_symbol to jump directly to a named function or class definition. "
        "Use find_callers to understand how a function is used across the codebase. "
        "Use trace_calls to follow an execution path end-to-end across multiple functions. "
        "Use read_file to read an entire source file (imports, structure, context). "
        "Use get_file_chunk to read specific line ranges when you know the location. "
        "Use note(key, value) to record key discoveries and recall_notes() to retrieve them. "
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
    # Validate repo format — LLM-supplied args must never be passed raw into URLs.
    if "/" not in repo or repo.count("/") != 1:
        return f"Invalid repo format '{repo}'. Expected 'owner/name'."

    # Reject path traversal — an LLM (or prompt injection) could pass "../.env"
    # which would resolve to an unintended path in the GitHub API URL.
    from pathlib import PurePosixPath
    try:
        parts = PurePosixPath(filepath).parts
    except Exception:
        return "Invalid filepath."
    if ".." in parts or filepath.startswith("/"):
        return "Invalid filepath: path traversal not allowed."

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


@mcp.tool()
def search_symbol(symbol_name: str, repo: Optional[str] = None) -> str:
    """
    Find the exact definition of a function or class by name.

    This is a STRUCTURAL lookup — it searches the 'name' metadata field in
    Qdrant, not the vector space. Use this when you know the exact name of
    a function or class and want its definition immediately, without the
    approximation of semantic search.

    When to use instead of search_code:
    - You already know the exact name (e.g. from a previous search result)
    - search_code returned a caller but not the definition itself
    - You want to be sure you're reading the right function, not a similar one

    Args:
        symbol_name: Exact function or class name (e.g. 'backward', 'Value', '_embed')
        repo:        Optional 'owner/repo' to restrict search
    """
    if _store is None:
        return "Search service not ready."

    matches = _store.find_symbol(symbol_name, repo=repo)
    if not matches:
        return (
            f"No indexed definition found for '{symbol_name}'. "
            "The symbol may not be in the index — try search_code instead."
        )

    parts = []
    for i, c in enumerate(matches[:5], 1):
        citation = c.get("filepath", "")
        citation += f" | lines {c.get('start_line', '?')}–{c.get('end_line', '?')}"
        parts.append(f"[Source {i} | {c.get('repo', '')} | {citation}]\n{c.get('text', '')}")

    return (
        f"Found {len(matches)} definition(s) of '{symbol_name}':\n\n"
        + "\n\n" + "─" * 40 + "\n\n".join(parts)
    )


@mcp.tool()
def read_file(repo: str, filepath: str) -> str:
    """
    Read the ENTIRE source file from a GitHub repository.

    Use when you need to understand a module's full structure — all its
    imports, class definitions, and how functions relate to each other.
    This reveals things that chunked search cannot: file-level imports,
    module docstrings, and the order in which things are defined.

    When to use instead of get_file_chunk:
    - You don't know the line numbers yet and want the full picture
    - You need to see imports (they're usually not in any function chunk)
    - You want to understand how multiple functions in a file interact

    Cost: one GitHub API call. Avoid calling on very large files (>500 lines);
    use get_file_chunk for targeted reads after search_code gives you line numbers.

    Args:
        repo:     'owner/repo' (e.g. 'karpathy/micrograd')
        filepath: Path within the repo (e.g. 'micrograd/engine.py')
    """
    if "/" not in repo or repo.count("/") != 1:
        return f"Invalid repo format '{repo}'. Expected 'owner/name'."

    from pathlib import PurePosixPath
    try:
        parts = PurePosixPath(filepath).parts
    except Exception:
        return "Invalid filepath."
    if ".." in parts or filepath.startswith("/"):
        return "Invalid filepath: path traversal not allowed."

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
    total = len(lines)

    # Warn on large files so the agent can decide whether to read a chunk instead.
    # 300 lines is ~15k characters — comfortably within context limits.
    header = f"# {repo} — {filepath}  ({total} lines total)\n\n"
    if total > 500:
        header += f"# ⚠️  Large file ({total} lines). Consider get_file_chunk for targeted reads.\n\n"

    numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    return header + numbered


@mcp.tool()
def list_files(repo: str, path: str = "") -> str:
    """
    List files and directories at a path within a GitHub repository.

    Use at the START of exploring an unfamiliar codebase to understand its structure.
    Then use read_file or get_file_chunk to read specific files you find.

    Returns file names, types (file/dir), and sizes. Directories are marked with /
    so you can drill down into them with another list_files call.

    Args:
        repo: 'owner/repo' (e.g. 'karpathy/micrograd')
        path: Directory path within the repo (e.g. 'src/models'). Empty = repo root.
    """
    if "/" not in repo or repo.count("/") != 1:
        return f"Invalid repo format '{repo}'. Expected 'owner/name'."

    from pathlib import PurePosixPath
    if path:
        try:
            parts = PurePosixPath(path).parts
        except Exception:
            return "Invalid path."
        if ".." in parts or path.startswith("/"):
            return "Invalid path: traversal not allowed."

    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/contents/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if settings.github_token:
        headers["Authorization"] = f"token {settings.github_token}"

    try:
        resp = http_requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return f"Path not found: '{path}' in {repo}"
        resp.raise_for_status()
    except Exception as e:
        return f"GitHub fetch failed: {e}"

    entries = resp.json()
    if not isinstance(entries, list):
        # Single file returned — path was a file, not a directory
        return f"'{path}' is a file, not a directory. Use read_file to read it."

    lines = [f"# {repo}/{path or ''}\n"]
    dirs  = sorted([e for e in entries if e["type"] == "dir"],  key=lambda e: e["name"])
    files = sorted([e for e in entries if e["type"] == "file"], key=lambda e: e["name"])

    for e in dirs:
        lines.append(f"  {e['name']}/")
    for e in files:
        size = e.get("size", 0)
        size_str = f"{size // 1024}KB" if size >= 1024 else f"{size}B"
        lines.append(f"  {e['name']}  ({size_str})")

    lines.append(f"\n{len(dirs)} director{'ies' if len(dirs) != 1 else 'y'}, {len(files)} file{'s' if len(files) != 1 else ''}")
    return "\n".join(lines)


@mcp.tool()
def note(key: str, value: str) -> str:
    """
    Save a key fact to working memory for this session.

    Use this to record important discoveries so you don't need to re-search them:
      note("entry_point", "train.py:main() at line 234")
      note("optimizer",   "AdamW, lr=6e-4, defined in model.py:320")
      note("data_flow",   "DataLoader → batch_encode → model.forward → loss.backward")

    Keeps your reasoning grounded: note facts as you find them, then recall_notes()
    before answering to make sure you haven't forgotten anything discovered earlier.

    Args:
        key:   Short label for the fact (e.g. 'entry_point', 'main_class')
        value: The fact itself — be specific: include file, line, and what it does
    """
    _session_notes[key] = value

    # Write-through to Qdrant for cross-session persistence.
    # Non-fatal: if Qdrant is unavailable, the note lives in-memory for this session.
    if _current_repo and _store is not None:
        try:
            _store.save_note(_current_repo, key, value)
        except Exception as e:
            print(f"[notes] Could not persist note '{key}': {e}")

    return f"✓ Noted: {key} = {value}"


@mcp.tool()
def recall_notes() -> str:
    """
    Retrieve all facts saved with note() during this session.

    Call this before writing your final answer to ensure you haven't
    forgotten any discoveries from earlier in the conversation.
    Also useful at the START of a follow-up question to check what you
    already know before deciding which searches to run.

    Returns all (key, value) pairs recorded so far, or a message if none.
    """
    if not _session_notes:
        return "No notes yet. Use note(key, value) to record discoveries as you search."
    lines = ["# Session notes\n"]
    for k, v in _session_notes.items():
        lines.append(f"**{k}**: {v}")
    return "\n".join(lines)


@mcp.tool()
def draw_diagram(description: str, diagram_type: str = "flowchart") -> str:
    """
    Signal that you want to draw a diagram inline in the chat response.

    After calling this tool, include a fenced code block with language 'diagram'
    in your answer containing valid Mermaid syntax. The frontend renders it as an
    SVG diagram with an expand-to-fullscreen button.

    When to use:
    - User asks to "draw", "visualize", "diagram", or "show" a relationship
    - The answer is clearer as a visual: architecture, data flow, class hierarchy,
      call graph, sequence of steps, component dependencies

    IMPORTANT: Always search the codebase (search_code, search_symbol, read_file)
    BEFORE calling this tool. The diagram must be grounded in real code you found.
    Call draw_diagram LAST in your tool sequence, right before writing the diagram block.

    Use any Mermaid diagram type that best fits the content:
      flowchart / graph   → components, steps, decisions, data flow
      classDiagram        → classes, inheritance, methods
      sequenceDiagram     → call sequences between objects
      stateDiagram-v2     → state machines, lifecycle
      erDiagram           → entity relationships
      gitGraph            → git branching
      mindmap             → hierarchical concepts
      timeline            → chronological events
      Choose whichever type communicates the idea most clearly.

    After calling this tool, always write 1-3 sentences describing what the diagram
    shows, then output the diagram block:

      Here is a diagram showing how X connects to Y and Z:

      ```diagram
      flowchart LR
          A[DataLoader] --> B[Model]
          B --> C[Loss]
          C --> D[Optimizer]
      ```

      Never output just the code block alone — always include a description.

    Args:
        description:  What you plan to draw (e.g. "class hierarchy for the model")
        diagram_type: Any Mermaid diagram type (flowchart, classDiagram, sequenceDiagram,
                      graph, stateDiagram-v2, erDiagram, gitGraph, mindmap, timeline, etc.)
    """
    # Only flowchart and graph support direction modifiers (LR/TD).
    # All other types use their keyword alone as the opening line.
    directional = {"flowchart", "graph"}
    if diagram_type in directional:
        starter = f"{diagram_type} LR"
    else:
        starter = diagram_type

    return (
        f"Ready. Draw '{description}' using Mermaid {diagram_type} syntax.\n\n"
        f"Output it in your response as a fenced code block with language 'diagram':\n\n"
        f"```diagram\n"
        f"{starter}\n"
        f"    %% nodes and edges here\n"
        f"```\n\n"
        f"IMPORTANT rules for valid Mermaid:\n"
        f"- Node labels must be SHORT (2-4 words max) — they are rendered as-is, no wrapping\n"
        f"- NEVER use <br>, <br/>, or any HTML tags in labels — they show as literal text\n"
        f"- NEVER use 'style' commands or 'classDef' — they break the renderer\n"
        f"- Keep node IDs short, no spaces (e.g. Value, MLP, Neuron)\n"
        f"- ALWAYS quote labels that contain parentheses, operators, or special chars:\n"
        f"  WRONG: D[Call backward()]   RIGHT: D[\"Call backward()\"]\n"
        f"  WRONG: K[grad += x * y]     RIGHT: K[\"grad += x * y\"]\n"
        f"  Special chars that REQUIRE quotes: ( ) [ ] {{ }} + = * / % < > & | # ;\n"
        f"- classDiagram: always include key attributes and methods inside curly braces, then list relationships\n"
        f"  class Foo {{\n    +type attr\n    +method()\n  }}\n"
        f"  WRONG: class Foo <|-- Bar  (cannot combine declaration + relationship)\n"
        f"  RIGHT: declare all classes first (with members), then relationships on their own lines\n"
        f"  Relationship arrows: <|-- inheritance, --> association, *-- composition\n"
        f"  Empty class boxes are useless — always populate them with real attributes/methods from the code\n"
        f"- flowchart/graph: use LR or TD direction, quote labels with spaces: A[\"My Label\"]\n"
        f"- sequenceDiagram: use participant declarations, ->> for messages\n"
        f"- Good label example: Value, MLP, Neuron, Layer\n"
        f"- Bad label example: Value Class<br>Core scalar unit — NEVER do this\n"
        f"The frontend renders this as a live SVG diagram with expand-to-fullscreen."
    )


@mcp.tool()
def trace_calls(
    repo: str,
    symbol_name: str,
    max_depth: int = 3,
) -> str:
    """
    Trace the execution path starting from a function — who it calls and who they call.

    Walks the call graph stored during AST indexing (the 'calls' payload field).
    This answers "what happens when X runs?" by following the chain of function
    calls up to max_depth levels deep.

    Unlike find_callers (who calls X?), trace_calls answers "what does X call?"
    and recursively follows those calls to map the full execution path.

    Use this for:
      - Understanding data flow: "how does the forward pass work end-to-end?"
      - Tracing feature pipelines: "what does train() actually do step by step?"
      - Debugging: "which functions does backward() touch?"

    Args:
        repo:        'owner/repo' (e.g. 'karpathy/micrograd')
        symbol_name: Starting function or method name (e.g. 'train', 'forward')
        max_depth:   How many levels deep to follow calls (default 3, max 5)
    """
    if _store is None:
        return "Search service not ready."

    max_depth = min(max_depth, 5)  # hard cap to avoid runaway traversal
    visited:  set[str] = set()
    lines:    list[str] = [f"# Call trace from `{symbol_name}` in {repo}\n"]
    missing:  list[str] = []  # names we saw in calls[] but couldn't find in index

    def _walk(name: str, depth: int, prefix: str) -> None:
        if depth > max_depth or name in visited:
            return
        visited.add(name)

        chunks = _store.find_symbol(name, repo=repo)
        if not chunks:
            missing.append(name)
            return

        c = chunks[0]  # take the first definition (most repos have exactly one)
        loc = f"{c.get('filepath', '?')} L{c.get('start_line', '?')}"
        lines.append(f"{prefix}→ **{name}**()  `{loc}`")

        calls = [fn for fn in (c.get("calls") or []) if fn not in visited]
        # Limit fan-out to 6 per level so the trace stays readable
        for callee in calls[:6]:
            _walk(callee, depth + 1, prefix + "  ")

    _walk(symbol_name, 0, "")

    if len(lines) == 1:
        return (
            f"Symbol '{symbol_name}' not found in the index for {repo}. "
            "Try search_symbol() to check the exact name, or search_code() to find it."
        )

    if missing:
        lines.append(f"\n_Could not find definitions for: {', '.join(missing[:8])}_")

    lines.append(f"\n_{len(visited)} unique symbol(s) traced · depth limit {max_depth}_")
    return "\n".join(lines)


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

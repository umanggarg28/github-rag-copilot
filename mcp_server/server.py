"""
mcp_server/server.py — Our GitHub RAG Copilot as an MCP server.

═══════════════════════════════════════════════════════════════
WHAT IS MCP?
═══════════════════════════════════════════════════════════════

MCP (Model Context Protocol) is an open standard created by Anthropic that
defines HOW AI models connect to external tools and data sources.

Think of it like USB-C for AI:
  Before USB-C, every device had a different charging port.
  Before MCP, every AI application built its own custom tool integration.

With MCP:
  - You build a server ONCE exposing your capabilities
  - ANY MCP client (Claude Desktop, Cursor, your custom app) can use it
  - The AI model gets a consistent interface regardless of the tool

Without MCP (what we had before):
  our_app → hardcoded API calls → specific tools

With MCP:
  our_app ←→ MCP protocol ←→ ANY tools
  Claude Desktop ←→ MCP protocol ←→ our RAG server
  Cursor ←→ MCP protocol ←→ our RAG server

═══════════════════════════════════════════════════════════════
MCP'S THREE PRIMITIVES
═══════════════════════════════════════════════════════════════

MCP defines exactly three things a server can expose:

  1. TOOLS      — functions the LLM can call
                  "search for code", "read a file", "run a query"
                  → LLM decides when to call them (autonomous)

  2. RESOURCES  — data the LLM can read (like files or DB records)
                  "here is the list of indexed repos"
                  → Client controls when to read them (not LLM)

  3. PROMPTS    — reusable prompt templates with arguments
                  "explain this function: {code}"
                  → User triggers these (shown as slash commands in Claude Desktop)

Each primitive has a different actor:
  Tools     → LLM-driven  (model decides to call them mid-reasoning)
  Resources → Client-driven (app fetches them at context-building time)
  Prompts   → User-driven  (user picks them from a menu)

═══════════════════════════════════════════════════════════════
TWO TRANSPORT MODES
═══════════════════════════════════════════════════════════════

MCP servers communicate over one of two transports:

  STDIO (standard input/output):
    - Claude Desktop spawns your server as a subprocess
    - Communication happens over stdin/stdout pipes
    - Simpler, no network configuration needed
    - Best for: local tools, Claude Desktop integration

  HTTP + SSE (Server-Sent Events):
    - Your server runs as a web service
    - LLM connects over the network
    - Supports multiple concurrent clients
    - Best for: deployed services, shared team tools

This server supports BOTH — stdio for local dev, HTTP for production.

═══════════════════════════════════════════════════════════════
TOOLS WE EXPOSE
═══════════════════════════════════════════════════════════════

  search_code(query, repo?, language?, mode?, top_k?)
    → Hybrid BM25 + semantic search over indexed repos
    → Returns ranked code chunks with filepath + line numbers

  list_repos()
    → Returns all repos currently in the index

  get_file_chunk(repo, filepath, start_line, end_line)
    → Fetches a specific range of lines from GitHub
    → Used for follow-up: "show me more of that function"

  find_callers(function_name, repo)
    → Searches for all call sites of a function
    → Enables "who calls this?" multi-hop reasoning
"""

from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from retrieval.retrieval import RetrievalService
from ingestion.qdrant_store import QdrantStore
from ingestion.repo_fetcher import fetch_repo_files, parse_github_url
from backend.config import settings

# ── Server init ───────────────────────────────────────────────────────────────
# The Server object is the MCP server. It handles the protocol:
# - responding to tool/resource/prompt list requests
# - dispatching tool calls to our handlers
# - serialising results back in MCP format

app = Server("github-rag-copilot")

# Services loaded once — same pattern as FastAPI lifespan
_retrieval: RetrievalService | None = None
_store: QdrantStore | None = None


def get_retrieval() -> RetrievalService:
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalService()
    return _retrieval


def get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# Every tool needs:
#   1. A name (what the LLM calls)
#   2. A description (what the LLM reads to decide whether to call it)
#   3. An inputSchema (JSON Schema — the LLM fills this in)
#   4. A handler (@app.call_tool)
#
# The description is CRITICAL — it's the only thing the LLM reads when
# deciding which tool to use. Write it like documentation for a smart
# person who can't see your code.
# ══════════════════════════════════════════════════════════════════════════════

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Called by MCP clients to discover what this server can do."""
    return [
        types.Tool(
            name="search_code",
            description=(
                "Search for code chunks relevant to a query using hybrid BM25 + semantic search. "
                "Returns ranked code snippets with file paths and line numbers. "
                "Use this to find function definitions, class implementations, usage examples, "
                "or any code related to a concept. "
                "Specify repo to restrict search to a single repository."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or code identifier to search for",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Optional: restrict to a repo slug like 'karpathy/micrograd'",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional: filter by language like 'python', 'typescript'",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "keyword"],
                        "description": "Search strategy. hybrid (default) combines semantic + BM25.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_repos",
            description=(
                "List all GitHub repositories currently indexed and available for search. "
                "Returns repo slugs (owner/name) and chunk counts. "
                "Call this first to know which repos are available before searching."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get_file_chunk",
            description=(
                "Fetch the raw content of a specific file section from GitHub. "
                "Use this to see more context around a search result — for example, "
                "if search returns lines 45–52 but you need the full function including "
                "its docstring at lines 38–44. "
                "Requires the repo to be publicly accessible on GitHub."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository slug like 'karpathy/micrograd'",
                    },
                    "filepath": {
                        "type": "string",
                        "description": "File path within the repo like 'micrograd/engine.py'",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to fetch (1-indexed)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to fetch (inclusive)",
                    },
                },
                "required": ["repo", "filepath", "start_line", "end_line"],
            },
        ),
        types.Tool(
            name="find_callers",
            description=(
                "Find all places in the indexed code that call a specific function or class. "
                "Use this for multi-hop reasoning: after finding a function definition, "
                "call this to understand how it's used and in what context. "
                "Returns code chunks containing calls to the specified name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Function or class name to search for call sites",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Optional: restrict to a specific repository",
                    },
                },
                "required": ["function_name"],
            },
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL HANDLERS
# @app.call_tool() receives the tool name + arguments from the LLM.
# Returns a list of content blocks (text, image, or resource).
# ══════════════════════════════════════════════════════════════════════════════

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Route tool calls to the appropriate handler."""
    if name == "search_code":
        return await _handle_search_code(arguments)
    elif name == "list_repos":
        return await _handle_list_repos(arguments)
    elif name == "get_file_chunk":
        return await _handle_get_file_chunk(arguments)
    elif name == "find_callers":
        return await _handle_find_callers(arguments)
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_search_code(args: dict) -> list[types.TextContent]:
    retrieval = get_retrieval()
    results = retrieval.search(
        query=args["query"],
        top_k=args.get("top_k", 5),
        repo_filter=args.get("repo"),
        language_filter=args.get("language"),
        mode=args.get("mode", "hybrid"),
    )
    if not results:
        return [types.TextContent(type="text", text="No results found.")]
    return [types.TextContent(
        type="text",
        text=retrieval.format_context(results),
    )]


async def _handle_list_repos(args: dict) -> list[types.TextContent]:
    store = get_store()
    repos = store.list_repos()
    if not repos:
        return [types.TextContent(type="text", text="No repositories indexed yet.")]
    lines = [f"- {slug} ({store.count(repo=slug)} chunks)" for slug in repos]
    return [types.TextContent(type="text", text="Indexed repositories:\n" + "\n".join(lines))]


async def _handle_get_file_chunk(args: dict) -> list[types.TextContent]:
    """Fetch a file from GitHub and return the requested line range."""
    import requests
    repo     = args["repo"]
    filepath = args["filepath"]
    start    = args["start_line"]
    end      = args["end_line"]

    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/contents/{filepath}"
    headers = {"Accept": "application/vnd.github.v3.raw"}
    if settings.github_token:
        headers["Authorization"] = f"token {settings.github_token}"

    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code == 404:
        return [types.TextContent(type="text", text=f"File not found: {filepath}")]
    response.raise_for_status()

    lines = response.text.splitlines()
    # Clamp to actual file length
    start = max(1, start)
    end   = min(len(lines), end)
    chunk = "\n".join(f"{i+start}: {line}" for i, line in enumerate(lines[start-1:end]))
    return [types.TextContent(
        type="text",
        text=f"# {repo} — {filepath} (lines {start}–{end})\n\n{chunk}",
    )]


async def _handle_find_callers(args: dict) -> list[types.TextContent]:
    """Find call sites by keyword-searching for the function name."""
    retrieval = get_retrieval()
    # Keyword mode is best for exact identifier matching
    results = retrieval.search(
        query=args["function_name"],
        top_k=8,
        repo_filter=args.get("repo"),
        mode="keyword",
    )
    # Filter to chunks that actually contain the name (keyword search may return
    # chunks that share tokens with the name)
    name = args["function_name"]
    callers = [r for r in results if name in r["text"]]
    if not callers:
        return [types.TextContent(type="text", text=f"No call sites found for '{name}'.")]
    return [types.TextContent(
        type="text",
        text=retrieval.format_context(callers),
    )]


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# Resources are read-only data the LLM (or client) can browse.
# Unlike tools (LLM calls them to act), resources are like open tabs —
# the client can read them to build up context.
#
# We expose each indexed repo as a resource with a custom URI scheme:
#   rag://repos/karpathy/micrograd
# ══════════════════════════════════════════════════════════════════════════════

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    """Expose indexed repos as browsable resources."""
    store = get_store()
    repos = store.list_repos()
    return [
        types.Resource(
            uri=f"rag://repos/{slug}",
            name=slug,
            description=f"Indexed code from {slug} ({store.count(repo=slug)} chunks)",
            mimeType="text/plain",
        )
        for slug in repos
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Return a summary for a repo resource."""
    # Parse rag://repos/owner/name
    slug = uri.removeprefix("rag://repos/")
    store = get_store()
    count = store.count(repo=slug)
    return f"Repository: {slug}\nIndexed chunks: {count}\n\nUse the search_code tool to query this repo."


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# Prompts are reusable templates shown to the user as slash commands in
# Claude Desktop. The user picks a prompt, fills in arguments, and Claude
# executes it with the template expanded.
#
# This is different from tools (which the LLM calls) and resources (which
# the client reads). Prompts are USER-INITIATED templates.
# ══════════════════════════════════════════════════════════════════════════════

@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="explain-function",
            description="Retrieve and explain a specific function from the indexed repos",
            arguments=[
                types.PromptArgument(
                    name="function_name",
                    description="Name of the function to explain",
                    required=True,
                ),
                types.PromptArgument(
                    name="repo",
                    description="Repository slug (optional, e.g. karpathy/micrograd)",
                    required=False,
                ),
            ],
        ),
        types.Prompt(
            name="repo-overview",
            description="Generate an architectural overview of an indexed repository",
            arguments=[
                types.PromptArgument(
                    name="repo",
                    description="Repository slug like 'karpathy/micrograd'",
                    required=True,
                ),
            ],
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    if name == "explain-function":
        fn   = arguments.get("function_name", "")
        repo = arguments.get("repo", "")
        repo_clause = f" in {repo}" if repo else ""
        return types.GetPromptResult(
            description=f"Explain {fn}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=(
                            f"Use the search_code tool to find the implementation of `{fn}`{repo_clause}. "
                            f"Then explain what it does, its parameters, return value, and any important "
                            f"implementation details. Cite the source file and line numbers."
                        ),
                    ),
                )
            ],
        )
    elif name == "repo-overview":
        repo = arguments.get("repo", "")
        return types.GetPromptResult(
            description=f"Overview of {repo}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=(
                            f"Use search_code with repo='{repo}' to explore the codebase. "
                            f"Search for: main entry points, core data structures, key abstractions. "
                            f"Then write a structured architectural overview covering: "
                            f"1) What the project does, 2) Main modules and their responsibilities, "
                            f"3) Key data flow, 4) Important design patterns used."
                        ),
                    ),
                )
            ],
        )
    raise ValueError(f"Unknown prompt: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# Run as stdio server (for Claude Desktop) or imported for HTTP mode.
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run the MCP server over stdio (for Claude Desktop integration)."""
    print("Starting GitHub RAG MCP server...", flush=True)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

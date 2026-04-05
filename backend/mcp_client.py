"""
mcp_client.py — Async MCP client that connects to our FastMCP server.

═══════════════════════════════════════════════════════════════
CLIENT vs SERVER
═══════════════════════════════════════════════════════════════

mcp_server.py  — defines WHAT tools exist and implements their logic
mcp_client.py  — connects to the server, DISCOVERS those tools, and calls them

This separation is the whole point of MCP. The client doesn't know in advance
what tools the server has — it discovers them at runtime via tools/list.
This means:

  1. Add a new tool to mcp_server.py
  2. Restart the server
  3. Client automatically sees and can use the new tool — zero code changes

This is also how you'd connect to a SECOND MCP server (e.g. a GitHub server
or a documentation server) — just create another MCPClient pointing to a
different URL and merge the tool lists.

═══════════════════════════════════════════════════════════════
WHY ASYNC?
═══════════════════════════════════════════════════════════════

The MCP Python SDK is built on asyncio. All protocol operations are coroutines:
  - Connecting to the server
  - Sending tools/list
  - Sending tools/call
  - Reading the response

FastAPI is also async, so this fits naturally. We use `await` to call MCP
operations without blocking the server from handling other requests.

═══════════════════════════════════════════════════════════════
CONNECTION STRATEGY: ONE SESSION PER CALL
═══════════════════════════════════════════════════════════════

Each tool call opens a fresh HTTP connection, sends the request, reads the
response, and closes the connection. This is "stateless HTTP" mode.

Trade-off:
  + Simple: no connection pooling, no reconnection logic, no stale sessions
  + Correct: each call is fully independent, no shared state can corrupt
  - Slightly slower: TCP handshake + MCP init per call (~10ms overhead)

For our use case (2–5 tool calls per agent loop, not high-throughput),
this is the right trade-off. A production system would maintain a persistent
session pool, but that's complexity we don't need for learning purposes.

═══════════════════════════════════════════════════════════════
MCP PROTOCOL FLOW (what happens on each call)
═══════════════════════════════════════════════════════════════

1. Client opens HTTP connection to server URL
2. Client sends: initialize request with protocol version + capabilities
3. Server responds: its capabilities (tools, resources, prompts it supports)
4. Client sends: tools/call with {name, arguments}
5. Server executes the tool and streams back content blocks
6. Client reads content, extracts text, returns to caller
7. Connection closes

The SDK handles steps 1–3 (the handshake) inside ClientSession.initialize().
"""

import sys
from pathlib import Path
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

sys.path.insert(0, str(Path(__file__).parent.parent))


class MCPClient:
    """
    Async client for the Cartographer MCP server.

    Used by AgentService to call tools via the MCP protocol instead of
    calling retrieval/GitHub APIs directly. The agent doesn't know HOW
    tools are implemented — it just discovers them and calls them by name.

    Usage in agent.py:
        client = MCPClient("http://localhost:8000/mcp")
        tools = await client.list_tools()
        # ... LLM decides to call search_code ...
        result = await client.call_tool("search_code", {"query": "backward()"})
    """

    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        """
        Args:
            server_url: Full URL to the MCP server's Streamable HTTP endpoint.
                        Default assumes the server is mounted at /mcp in FastAPI.
        """
        self.server_url = server_url
        # Cache tool list after first discovery — tool definitions don't change
        # between calls in a single agent session.
        self._cached_tools: list | None = None

    # ── Tool discovery ──────────────────────────────────────────────────────────

    async def list_tools(self) -> list:
        """
        Discover all tools available on the MCP server.

        Sends the MCP tools/list request. The server returns a list of
        ToolDefinition objects, each with:
          - name:        the tool's identifier
          - description: what it does (this is what the LLM reads)
          - inputSchema: JSON Schema for the arguments

        Results are cached so discovery only pays the round-trip cost once
        per MCPClient instance.
        """
        if self._cached_tools is not None:
            return self._cached_tools

        async with streamablehttp_client(self.server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                self._cached_tools = result.tools
                print(f"MCP client: discovered {len(result.tools)} tools from {self.server_url}")
                return result.tools

    # ── Tool execution ──────────────────────────────────────────────────────────

    async def call_tool(self, name: str, arguments: dict) -> str:
        """
        Call a tool on the MCP server and return its result as a string.

        Sends the MCP tools/call request with the tool name and arguments.
        The server executes the corresponding Python function and returns
        content blocks. We extract all text blocks and join them.

        Args:
            name:      Tool name (must match one returned by list_tools)
            arguments: Dict matching the tool's inputSchema

        Returns:
            Tool output as a string, or an error message.
        """
        async with streamablehttp_client(self.server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)

        # MCP results are lists of content blocks (text, image, resource_link...)
        # We extract text blocks and concatenate them.
        if result.isError:
            error_text = (
                result.content[0].text
                if result.content and hasattr(result.content[0], "text")
                else "unknown error"
            )
            return f"Tool error: {error_text}"

        text_parts = [
            block.text
            for block in result.content
            if hasattr(block, "text") and block.text
        ]
        return "\n".join(text_parts) if text_parts else "(tool returned no output)"

    # ── Resource access ─────────────────────────────────────────────────────────

    async def list_resources(self) -> list:
        """
        List all resources available on the MCP server.

        Resources are read-only data sources identified by URI.
        Each resource has a URI, name, and optional description.
        """
        async with streamablehttp_client(self.server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_resources()
                return result.resources

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource by URI and return its content as a string.

        Args:
            uri: Resource URI (e.g. 'qdrant://repos' or 'qdrant://repos/owner/repo')
        """
        async with streamablehttp_client(self.server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.read_resource(uri)

        if result.contents:
            content = result.contents[0]
            return content.text if hasattr(content, "text") else str(content)
        return "(empty resource)"

    # ── Prompt access ───────────────────────────────────────────────────────────

    async def list_prompts(self) -> list:
        """
        List all prompts available on the MCP server.

        Prompts are reusable templates the user can invoke (like slash commands).
        Each has a name, description, and list of required arguments.
        """
        async with streamablehttp_client(self.server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_prompts()
                return result.prompts

    # ── Format conversion ───────────────────────────────────────────────────────
    # MCP tool definitions use a generic format. LLMs expect provider-specific
    # formats. These methods convert between them — enabling the same MCP tool
    # definitions to drive any LLM provider.

    def tools_as_openai_format(self, tools: list) -> list:
        """
        Convert MCP ToolDefinition objects to OpenAI/Groq function-calling format.

        MCP format:          tool.name, tool.description, tool.inputSchema
        OpenAI/Groq format:  {"type": "function", "function": {"name", "description", "parameters"}}

        This is how we get provider independence — the same MCP tool definitions
        drive both Groq (Llama) and any other OpenAI-compatible LLM.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name":        tool.name,
                    "description": tool.description or "",
                    "parameters":  tool.inputSchema,
                },
            }
            for tool in tools
        ]

    def tools_as_anthropic_format(self, tools: list) -> list:
        """
        Convert MCP ToolDefinition objects to Anthropic tool-use format.

        Anthropic format: {"name", "description", "input_schema"}
        Note: "input_schema" not "parameters" — key difference from OpenAI.
        """
        return [
            {
                "name":         tool.name,
                "description":  tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

    # ── Status / health ─────────────────────────────────────────────────────────

    async def get_server_info(self) -> dict:
        """
        Fetch a full status summary of the connected MCP server.

        Called by GET /mcp/status to show live server info in the UI.
        Returns tool names/descriptions, resource URIs, and prompt names.
        On connection failure, returns connected=False with the error.
        """
        try:
            async with streamablehttp_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_r     = await session.list_tools()
                    resources_r = await session.list_resources()
                    prompts_r   = await session.list_prompts()

            return {
                "connected":  True,
                "server_url": self.server_url,
                "tools": [
                    {"name": t.name, "description": (t.description or "")[:120]}
                    for t in tools_r.tools
                ],
                "resources": [
                    {"uri": str(r.uri), "name": r.name}
                    for r in resources_r.resources
                ],
                "prompts": [
                    {"name": p.name, "description": (p.description or "")[:120]}
                    for p in prompts_r.prompts
                ],
            }
        except Exception as e:
            return {
                "connected":  False,
                "server_url": self.server_url,
                "error":      str(e),
                "tools":      [],
                "resources":  [],
                "prompts":    [],
            }

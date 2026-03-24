"""
agent.py — Async Agentic RAG via MCP tool use (Groq or Anthropic).

═══════════════════════════════════════════════════════════════
THE BIG PICTURE: HOW THIS FITS TOGETHER
═══════════════════════════════════════════════════════════════

Previous version: tools were hardcoded in this file.
  search_code(), find_callers(), get_file_chunk() all lived here.
  The agent knew exactly what tools exist and how they work.

This version: tools are discovered via MCP at runtime.
  The agent connects to our MCP server (mcp_server.py) on startup.
  It asks "what tools do you have?" and gets back a list.
  It passes those tool definitions to the LLM — the LLM reads the
  descriptions and decides when to call them.
  When a call is needed, the agent routes it back through the MCP client.

Why does this matter?
  1. Add a new tool to mcp_server.py → agent picks it up automatically
  2. Replace the MCP server with a different one → same agent, new capabilities
  3. Connect to MULTIPLE MCP servers → tools from all are merged seamlessly
  This is exactly how Claude Code works with its tools.

═══════════════════════════════════════════════════════════════
REACT LOOP (unchanged from before)
═══════════════════════════════════════════════════════════════

  question → think → [tool call via MCP] → observe → think → ...→ answer

  1. REASON: "I need to find backward()"
  2. ACT:    call search_code("backward implementation") via MCP
  3. OBSERVE: "Found Value.backward in engine.py"
  4. REASON: "Need to see its callers"
  5. ACT:    call find_callers("backward") via MCP
  6. OBSERVE: "Called by train() and evaluate()"
  7. REASON: "I have enough context"
  8. RESPOND: full answer with citations

═══════════════════════════════════════════════════════════════
WHY ASYNC?
═══════════════════════════════════════════════════════════════

MCP tool calls are async (HTTP requests). The LLM calls (Groq/Anthropic) are
sync but we run them in a thread pool with asyncio.to_thread() so they don't
block FastAPI's event loop during the HTTP call.

async/await is needed for:
  - await mcp.list_tools()         — MCP discovery (HTTP)
  - await mcp.call_tool(...)       — MCP tool execution (HTTP)
  - await asyncio.to_thread(...)   — sync LLM call in thread pool

The stream() method is an async generator (AsyncIterator) — it yields events
as they happen. FastAPI's StreamingResponse handles async generators natively.

═══════════════════════════════════════════════════════════════
TWO PROVIDERS, ONE INTERFACE (unchanged from before)
═══════════════════════════════════════════════════════════════

  GROQ (primary — free tier, Llama 3.3 70B)
    Tool format: OpenAI-compatible → mcp_client.tools_as_openai_format()
    Tool results: role="tool" (one message per result)
    Args come as JSON strings → json.loads()

  ANTHROPIC (fallback — paid, Claude Haiku)
    Tool format: Anthropic-specific → mcp_client.tools_as_anthropic_format()
    Tool results: role="user" with content=[{type: tool_result}]
    Args come as dicts → no parsing needed
"""

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.mcp_client import MCPClient


SYSTEM_PROMPT = """You are an expert code assistant with access to a searchable index of GitHub repositories.

When answering questions about code:
1. Start by calling search_code to find relevant code
2. If the initial results don't fully answer the question, search again with a different query
3. Use get_file_chunk to see more context around a result (the full class or surrounding code)
4. Use find_callers to understand how functions are used, not just defined
5. Only answer when you have enough evidence from the actual code

Always cite your sources: mention the file path and line numbers.
Be precise — if the code doesn't show what you're looking for, say so rather than guessing."""


class AgentService:
    """
    Async ReAct agent that uses MCP for tool discovery and execution.

    Key difference from the previous version:
      Before: tools = hardcoded list of dicts in this file
      Now:    tools = await self.mcp.list_tools()  (discovered from server)

    Provider auto-detection (unchanged):
      - GROQ_API_KEY set → Groq, Llama 3.3 70B (free)
      - ANTHROPIC_API_KEY set → Anthropic, Claude Haiku (paid fallback)
    """

    MAX_ITERATIONS = 8

    def __init__(self, mcp_client: MCPClient):
        """
        Args:
            mcp_client: Connected MCPClient pointing to our FastMCP server.
                        Tools are discovered lazily on first run/stream call.
        """
        self.mcp = mcp_client

        # ── Provider detection ─────────────────────────────────────────────────
        if settings.groq_api_key:
            from groq import Groq
            self._client   = Groq(api_key=settings.groq_api_key)
            self._provider = "groq"
            self._model    = "llama-3.3-70b-versatile"
            print("AgentService: using Groq (llama-3.3-70b-versatile) via MCP tools")
        elif settings.anthropic_api_key:
            import anthropic
            self._client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._provider = "anthropic"
            self._model    = "claude-haiku-4-5-20251001"
            print("AgentService: using Anthropic (claude-haiku) via MCP tools")
        else:
            raise ValueError("AgentService requires GROQ_API_KEY or ANTHROPIC_API_KEY")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self, question: str, repo_filter: str | None = None) -> dict:
        """
        Run the full ReAct loop and return the final answer + trace.

        Returns:
            {"answer": str, "tool_calls": list[dict], "iterations": int}
        """
        # Discover tools from MCP server
        mcp_tools   = await self.mcp.list_tools()
        tools_llm   = self._format_tools(mcp_tools)
        messages    = self._build_initial_messages(question, repo_filter)
        tool_trace  = []

        for iteration in range(self.MAX_ITERATIONS):
            # LLM call is synchronous — run in thread pool to avoid blocking
            step = await asyncio.to_thread(self._call_llm, messages, tools_llm)

            if step["done"]:
                return {
                    "answer":     step["answer"],
                    "tool_calls": tool_trace,
                    "iterations": iteration + 1,
                }

            messages.append(step["assistant_message"])

            for tc in step["tool_calls"]:
                # Tool execution via MCP protocol (async HTTP)
                try:
                    result = await self.mcp.call_tool(tc["name"], tc["input"])
                except Exception as e:
                    result = f"Tool error: {e}"

                tool_trace.append({
                    "tool":   tc["name"],
                    "input":  tc["input"],
                    "output": result[:500] + "..." if len(result) > 500 else result,
                })
                messages.append(self._build_tool_result(tc["id"], tc["name"], result))

        return {
            "answer":     "I was unable to fully answer within the allowed reasoning steps.",
            "tool_calls": tool_trace,
            "iterations": self.MAX_ITERATIONS,
        }

    async def stream(
        self, question: str, repo_filter: str | None = None
    ) -> AsyncIterator[dict]:
        """
        Stream agent progress as an async generator.

        Yields dicts as events happen in real time:
          {"type": "tool_call",   "tool": "search_code", "input": {...}}
          {"type": "tool_result", "tool": "search_code", "output": "..."}
          {"type": "token",       "text": "According to..."}
          {"type": "done",        "iterations": 3}

        Why async generator?
          Tool calls are async (await mcp.call_tool). Using 'async def' with
          'yield' creates an AsyncIterator — FastAPI's StreamingResponse and
          async for loops both consume it natively.

        Real token streaming:
          For the tool-calling iterations, we use non-streaming LLM calls —
          we need the FULL response to decide what tool to call next.
          Once the agent decides to give a final answer (no tool calls),
          we re-run with stream=True so tokens arrive in real time.
          This is one extra LLM call but delivers genuine streaming UX.
        """
        # Discover tools from MCP server (cached after first call)
        mcp_tools = await self.mcp.list_tools()
        tools_llm = self._format_tools(mcp_tools)
        messages  = self._build_initial_messages(question, repo_filter)

        for iteration in range(self.MAX_ITERATIONS):
            # Run sync LLM call in thread pool — doesn't block the event loop
            step = await asyncio.to_thread(self._call_llm, messages, tools_llm)

            if step["done"]:
                # Stream the final answer with real token-by-token delivery.
                # We pass messages (with all tool results) to the streaming call
                # and tell the LLM not to use tools (tool_choice="none") so it
                # goes straight to answering.
                async for token in self._stream_final_answer(messages):
                    yield {"type": "token", "text": token}
                yield {"type": "done", "iterations": iteration + 1}
                return

            messages.append(step["assistant_message"])

            for tc in step["tool_calls"]:
                yield {"type": "tool_call", "tool": tc["name"], "input": tc["input"]}

                # MCP tool call — async, goes through the full MCP protocol
                try:
                    result = await self.mcp.call_tool(tc["name"], tc["input"])
                except Exception as e:
                    result = f"Tool error: {e}"

                yield {"type": "tool_result", "tool": tc["name"], "output": result[:300]}
                messages.append(self._build_tool_result(tc["id"], tc["name"], result))

        yield {"type": "done", "iterations": self.MAX_ITERATIONS}

    async def _stream_final_answer(self, messages: list) -> AsyncIterator[str]:
        """
        Stream the final answer token by token using the LLM's native streaming.

        The challenge: Groq/Anthropic SDKs are synchronous (blocking iteration).
        We bridge sync → async using asyncio.Queue:
          1. A background thread runs the sync streaming loop, pushing tokens to a queue
          2. This async generator reads from the queue as tokens arrive
          3. A None sentinel signals the end of the stream

        This is the standard pattern for wrapping sync iterators in async code
        without blocking the event loop. Any async generator that needs to consume
        a sync blocking iterator should use this approach.
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_sync():
            try:
                if self._provider == "groq":
                    stream = self._client.chat.completions.create(
                        model=self._model,
                        max_tokens=2048,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                        # No tools parameter → model goes straight to answering
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            loop.call_soon_threadsafe(queue.put_nowait, delta)
                else:
                    # Anthropic: omit tools entirely for the final answer
                    with self._client.messages.stream(
                        model=self._model,
                        max_tokens=2048,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                    ) as stream:
                        for text in stream.text_stream:
                            loop.call_soon_threadsafe(queue.put_nowait, text)
            finally:
                # Always send the sentinel so the consumer loop ends
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Schedule the sync call in the default thread pool without blocking.
        # run_in_executor returns an asyncio.Future — we await it at the end
        # to propagate any exception raised inside _run_sync.
        task = loop.run_in_executor(None, _run_sync)

        # Consume tokens as they arrive from the background thread
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        await task  # re-raises any exception from the streaming thread

    # ── LLM dispatch ───────────────────────────────────────────────────────────

    def _format_tools(self, mcp_tools: list) -> list:
        """Convert MCP tool definitions to provider-specific format."""
        if self._provider == "groq":
            return self.mcp.tools_as_openai_format(mcp_tools)
        else:
            return self.mcp.tools_as_anthropic_format(mcp_tools)

    def _call_llm(self, messages: list, tools: list) -> dict:
        """
        Make one LLM call and return a normalised step dict.

        This is a SYNCHRONOUS method — called via asyncio.to_thread() from the
        async loop so it doesn't block the event loop while the HTTP call to
        Groq/Anthropic is in flight.

        Returns:
            {
              "done":              bool,
              "answer":            str | None,      # if done=True
              "tool_calls":        list[dict],      # if done=False
              "assistant_message": dict,            # for conversation history
            }
        """
        if self._provider == "groq":
            return self._call_groq(messages, tools)
        else:
            return self._call_anthropic(messages, tools)

    def _call_groq(self, messages: list, tools: list) -> dict:
        """
        Call Groq (OpenAI-compatible API).

        Key Groq specifics:
          - Tool arguments arrive as JSON strings → json.loads() needed
          - finish_reason="tool_calls" → more tools to run
          - finish_reason="stop" → done, answer in msg.content
          - Tool results: role="tool" with tool_call_id, one per call
        """
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=tools,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg    = choice.message

        if not msg.tool_calls or choice.finish_reason == "stop":
            return {
                "done":              True,
                "answer":            msg.content or "",
                "tool_calls":        [],
                "assistant_message": None,
            }

        tool_calls = [
            {
                "id":    tc.id,
                "name":  tc.function.name,
                "input": json.loads(tc.function.arguments),  # JSON string → dict
            }
            for tc in msg.tool_calls
        ]

        # Groq history needs raw tool_calls objects (not our normalised dicts)
        assistant_message = {
            "role":       "assistant",
            "content":    msg.content,  # may be None
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,  # keep as JSON string
                    },
                }
                for tc in msg.tool_calls
            ],
        }

        return {
            "done":              False,
            "answer":            None,
            "tool_calls":        tool_calls,
            "assistant_message": assistant_message,
        }

    def _call_anthropic(self, messages: list, tools: list) -> dict:
        """
        Call Anthropic Messages API.

        Key Anthropic specifics:
          - Tool arguments arrive as dicts (no json.loads needed)
          - stop_reason="end_turn" → done
          - stop_reason="tool_use" → more tools to run
          - Tool results: role="user" with content=[{type: tool_result}]
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            answer = "".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            return {
                "done":              True,
                "answer":            answer,
                "tool_calls":        [],
                "assistant_message": None,
            }

        tool_calls = [
            {
                "id":    block.id,
                "name":  block.name,
                "input": block.input,  # already a dict
            }
            for block in response.content
            if block.type == "tool_use"
        ]

        return {
            "done":              False,
            "answer":            None,
            "tool_calls":        tool_calls,
            "assistant_message": {"role": "assistant", "content": response.content},
        }

    # ── Message formatting ─────────────────────────────────────────────────────

    def _build_initial_messages(self, question: str, repo_filter: str | None) -> list:
        content = question
        if repo_filter:
            content += f"\n\n(Focus search on repo: {repo_filter})"
        return [{"role": "user", "content": content}]

    def _build_tool_result(self, tool_id: str, tool_name: str, result: str) -> dict:
        """
        Format a tool result for the conversation history.

        The two providers expect completely different formats for tool results:

        Groq/OpenAI: one message per result, role="tool"
          {"role": "tool", "tool_call_id": id, "content": result}

        Anthropic: all results in one user turn, role="user"
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, ...}]}
        """
        if self._provider == "groq":
            return {
                "role":         "tool",
                "tool_call_id": tool_id,
                "content":      result,
            }
        else:
            return {
                "role":    "user",
                "content": [
                    {
                        "type":        "tool_result",
                        "tool_use_id": tool_id,
                        "content":     result,
                    }
                ],
            }

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
import re
from pathlib import Path
from typing import AsyncIterator
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.mcp_client import MCPClient
from backend.services.generation import _is_exhausted

def _extract_thought(assistant_message: dict | None, provider: str) -> str:
    """
    Extract the LLM's pre-tool-call reasoning text from an assistant message.

    Before calling a tool, LLMs often emit a short reasoning sentence like
    "I need to search for the backward pass implementation" or "Let me check
    the callers of this function." This is the 'content' field in the assistant
    message before the tool_calls list.

    We surface this in the UI as a "thought bubble" so users can see *why*
    the agent chose each tool, not just *what* it called.

    Returns empty string if no reasoning text was present (model went straight
    to tool use without narrating its thinking).
    """
    if not assistant_message:
        return ""
    content = assistant_message.get("content")
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Anthropic format: list of content blocks, some with type="text"
        parts = [
            b.text.strip()
            for b in content
            if hasattr(b, "type") and b.type == "text" and b.text
        ]
        return " ".join(parts).strip()
    return ""


def _parse_xml_tool_calls(content: str) -> list[dict] | None:
    """
    Parse XML-style tool calls that some models emit instead of proper JSON.

    Some free-tier models (Nemotron, StepFun, etc.) reliably use JSON tool calls
    for the first few turns but then fall back to an XML format mid-conversation:

        <tool_call> <function=search_code>
          <parameter=query> attention mechanism </parameter>
          <parameter=repo> karpathy/micrograd </parameter>
        </function> </tool_call>

    Rather than blacklisting every misbehaving model, we parse this format
    and convert it to our internal tool-call dict — the rest of the agent
    pipeline works unchanged.
    """
    matches = re.findall(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        content, re.DOTALL
    )
    if not matches:
        return None
    result = []
    for i, (name, params_str) in enumerate(matches):
        params = {}
        for m in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', params_str, re.DOTALL):
            params[m.group(1)] = m.group(2).strip()
        result.append({"id": f"call_xml_{i}_{name}", "name": name, "input": params})
    return result or None


def _source_from_chunk_call(tool_input: dict, result: str) -> dict | None:
    """
    Build a SourceCard-compatible source dict from a get_file_chunk tool call.

    The get_file_chunk input has repo, filepath, start_line, end_line.
    We extract the raw code lines from the result string (which is formatted
    as "# repo — filepath (lines N–M)\n\nN: line\nN+1: line...").
    """
    repo       = tool_input.get("repo", "")
    filepath   = tool_input.get("filepath", "")
    start_line = tool_input.get("start_line", 0)
    end_line   = tool_input.get("end_line", 0)

    if not repo or not filepath:
        return None

    # Infer language from file extension
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java", ".cpp": "cpp",
        ".c": "c", ".md": "markdown", ".yaml": "yaml", ".json": "json",
    }
    ext  = "." + filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
    lang = ext_map.get(ext, "text")

    # Strip the header line, then remove the "N: " line-number prefixes.
    lines = result.splitlines()
    code_lines = []
    for line in lines:
        # Skip the header "# repo — filepath (lines N–M)"
        if line.startswith("# "):
            continue
        # Strip leading "N: " prefix added by get_file_chunk
        if ": " in line:
            code_lines.append(line.split(": ", 1)[1])
        else:
            code_lines.append(line)

    text = "\n".join(code_lines).strip()

    return {
        "repo":       repo,
        "filepath":   filepath,
        "language":   lang,
        "chunk_type": "function",
        "name":       filepath.rsplit("/", 1)[-1],
        "start_line": int(start_line),
        "end_line":   int(end_line),
        "score":      1.0,
        "text":       text,
    }


def _sources_from_search_result(result_text: str, fallback_repo: str | None) -> list[dict]:
    """
    Parse source metadata from a search_code or find_callers result string.

    The format produced by retrieval.format_context and find_callers is:
      [Source N | owner/repo | filepath — name() | lines S–E]
      <code text>

    We extract repo, filepath, name, and line numbers from the header lines.
    Each "[Source N | ...]" header introduces a new chunk.
    """
    import re
    sources = []

    # Match header lines like: [Source 1 | karpathy/micrograd | micrograd/engine.py — backward() | lines 45–78]
    header_re = re.compile(
        r'\[Source \d+ \| ([^|]+) \| ([^\]|]+?)(?:\s+—\s+(\S+))?\s*\|\s*lines\s+(\d+)[–\-](\d+)\]'
    )

    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java", ".cpp": "cpp",
        ".c": "c", ".md": "markdown", ".yaml": "yaml", ".json": "json",
    }

    # Split on blank lines to isolate each chunk block
    blocks = re.split(r'\n{2,}', result_text)
    for block in blocks:
        m = header_re.search(block)
        if not m:
            continue
        repo_raw  = m.group(1).strip()
        filepath  = m.group(2).strip()
        # group(2) may contain " — name()" suffix from find_callers format
        if " — " in filepath:
            filepath = filepath.split(" — ")[0].strip()
        name      = m.group(3).strip().rstrip("()") if m.group(3) else ""
        start_ln  = int(m.group(4))
        end_ln    = int(m.group(5))

        # repo might be empty string if the search was across all repos
        repo = repo_raw if repo_raw else (fallback_repo or "")

        ext  = "." + filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
        lang = ext_map.get(ext, "text")

        # Everything after the header line is code text
        header_end = block.find("]")
        text = block[header_end + 1:].strip() if header_end != -1 else ""

        sources.append({
            "repo":       repo,
            "filepath":   filepath,
            "language":   lang,
            "chunk_type": "function",
            "name":       name,
            "start_line": start_ln,
            "end_line":   end_ln,
            "score":      1.0,
            "text":       text,
        })

    return sources


# OpenRouter: free model with confirmed tool-calling support.
# Required headers: HTTP-Referer (for attribution) and X-Title (app name).
# Without HTTP-Referer, free tier access may be denied.
_OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"


def _openrouter_client(api_key: str):
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "GitHub RAG Copilot",
        },
    )


SYSTEM_PROMPT = """You are an expert code assistant with access to a searchable index of GitHub repositories.

IMPORTANT: Before EVERY tool call, write one sentence explaining what you are about to search for and why. For example: "I need to find the backward pass implementation to understand how gradients flow." This reasoning must appear as plain text before the tool call, not inside it.

When answering questions about code:
1. Start by calling search_code to find relevant code
2. If the initial results don't fully answer the question, search again with a different query
3. Use get_file_chunk to see more context around a result (the full class or surrounding code)
4. Use find_callers to understand how functions are used, not just defined
5. If a search returns no new information compared to a previous search, stop searching and answer from what you have
6. If you have tried 3+ different queries for the same thing and still haven't found it, it likely doesn't exist in the indexed code — say so and answer from what you do have

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

    MAX_ITERATIONS = 12

    def __init__(self, mcp_client: MCPClient):
        """
        Args:
            mcp_client: Connected MCPClient pointing to our FastMCP server.
                        Tools are discovered lazily on first run/stream call.
        """
        self.mcp = mcp_client

        # ── Provider detection ─────────────────────────────────────────────────
        # Priority: Gemini → OpenRouter → Anthropic (Groq excluded — hermes bug).
        # Groq's Llama 3.3 intermittently generates <function=name{...}> hermes-format
        # tool calls which Groq's own API rejects with a 400.
        # Gemini and OpenRouter both emit correct OpenAI JSON tool calls.
        if settings.gemini_api_key:
            from openai import OpenAI
            self._client   = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._provider = "gemini"
            self._model    = "gemini-2.0-flash"
            print("AgentService: using Google Gemini (gemini-2.0-flash) via MCP tools")
        elif settings.openrouter_api_key:
            self._client   = _openrouter_client(settings.openrouter_api_key)
            self._provider = "openrouter"
            self._model    = _OPENROUTER_MODEL
            print(f"AgentService: using OpenRouter ({_OPENROUTER_MODEL}) via MCP tools")
        elif settings.anthropic_api_key:
            import anthropic
            self._client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._provider = "anthropic"
            self._model    = "claude-haiku-4-5-20251001"
            print("AgentService: using Anthropic (claude-haiku) via MCP tools")
        elif settings.groq_api_key:
            from groq import Groq
            self._client   = Groq(api_key=settings.groq_api_key)
            self._provider = "groq"
            self._model    = "llama-3.3-70b-versatile"
            print("AgentService: using Groq (llama-3.3-70b-versatile) via MCP tools [hermes bug possible]")
        else:
            raise ValueError("AgentService requires GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self, question: str, repo_filter: str | None = None, history: list[dict] | None = None) -> dict:
        """
        Run the full ReAct loop and return the final answer + trace.

        Returns:
            {"answer": str, "tool_calls": list[dict], "iterations": int}
        """
        # Discover tools from MCP server
        mcp_tools  = await self.mcp.list_tools()
        messages   = self._build_initial_messages(question, repo_filter, history)
        tool_trace = []

        # Loop detection: track (tool, args) pairs already executed this run.
        # Prevents wasting all MAX_ITERATIONS on duplicate searches when the
        # model gets confused and repeats the same call over and over.
        seen_calls: set[tuple] = set()

        for iteration in range(self.MAX_ITERATIONS):
            # LLM call is synchronous — run in thread pool to avoid blocking
            # Pass raw mcp_tools so _call_llm can reformat if provider switches mid-run
            step = await asyncio.to_thread(self._call_llm, messages, mcp_tools)

            if step["done"]:
                return {
                    "answer":     step["answer"],
                    "tool_calls": tool_trace,
                    "iterations": iteration + 1,
                }

            messages.append(step["assistant_message"])

            for tc in step["tool_calls"]:
                # Deduplicate: skip calls already made with identical arguments.
                call_key = (tc["name"], tuple(sorted(tc["input"].items())))
                if call_key in seen_calls:
                    result = f"[Skipped duplicate {tc['name']} call — already ran with these arguments]"
                    tool_trace.append({"tool": tc["name"], "input": tc["input"], "output": result})
                    messages.append(self._build_tool_result(tc["id"], tc["name"], result))
                    continue
                seen_calls.add(call_key)

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
        self, question: str, repo_filter: str | None = None, history: list[dict] | None = None
    ) -> AsyncIterator[dict]:
        """
        Stream agent progress as an async generator.

        Yields dicts as events happen in real time:
          {"type": "tool_call",   "tool": "search_code", "input": {...}}
          {"type": "tool_result", "tool": "search_code", "output": "..."}
          {"type": "token",       "text": "According to..."}
          {"type": "sources",     "sources": [...]}   ← new: collected file references
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
        messages  = self._build_initial_messages(question, repo_filter, history)

        # Loop detection: skip duplicate tool calls in the stream path too.
        seen_calls: set[tuple] = set()

        # Collect source references from tool calls for the sources panel.
        # Keyed by (repo, filepath, start_line) to deduplicate across iterations.
        collected_sources: dict[tuple, dict] = {}

        for iteration in range(self.MAX_ITERATIONS):
            # Run sync LLM call in thread pool — doesn't block the event loop
            # Pass raw mcp_tools so _call_llm can reformat if provider switches mid-run
            step = await asyncio.to_thread(self._call_llm, messages, mcp_tools)

            if step["done"]:
                # Stream the final answer with real token-by-token delivery.
                # We pass messages (with all tool results) to the streaming call
                # and tell the LLM not to use tools (tool_choice="none") so it
                # goes straight to answering.
                async for token in self._stream_final_answer(messages, mcp_tools):
                    yield {"type": "token", "text": token}
                # Emit sources collected across all tool calls before done event
                if collected_sources:
                    yield {"type": "sources", "sources": list(collected_sources.values())}
                yield {"type": "done", "iterations": iteration + 1}
                return

            messages.append(step["assistant_message"])

            # Emit any pre-tool reasoning text the LLM produced before calling tools.
            # This lets the UI show "thought bubbles" in the trace timeline —
            # the user sees WHY each tool was chosen, not just WHAT was called.
            thought = _extract_thought(step["assistant_message"], self._provider)
            if thought:
                yield {"type": "thought", "text": thought}

            for tc in step["tool_calls"]:
                # Deduplicate: skip calls already made with identical arguments.
                call_key = (tc["name"], tuple(sorted(tc["input"].items())))
                if call_key in seen_calls:
                    dup_msg = f"[Skipped duplicate {tc['name']} call — already ran with these arguments]"
                    yield {"type": "tool_result", "tool": tc["name"], "output": dup_msg}
                    messages.append(self._build_tool_result(tc["id"], tc["name"], dup_msg))
                    continue
                seen_calls.add(call_key)

                yield {"type": "tool_call", "tool": tc["name"], "input": tc["input"]}

                # MCP tool call — async, goes through the full MCP protocol
                try:
                    result = await self.mcp.call_tool(tc["name"], tc["input"])
                except Exception as e:
                    result = f"Tool error: {e}"

                # Collect source metadata from get_file_chunk calls.
                # The input dict has repo, filepath, start_line, end_line — enough
                # to construct a SourceCard-compatible object for the UI.
                if tc["name"] == "get_file_chunk":
                    src = _source_from_chunk_call(tc["input"], result)
                    if src:
                        key = (src["repo"], src["filepath"], src["start_line"])
                        collected_sources[key] = src

                # Parse search_code results to extract file references.
                # search_code returns formatted text blocks with "[Source N | repo | filepath ...]" headers.
                if tc["name"] in ("search_code", "find_callers") and not result.startswith("No results"):
                    for src in _sources_from_search_result(result, tc["input"].get("repo") or repo_filter):
                        key = (src["repo"], src["filepath"], src["start_line"])
                        collected_sources[key] = src

                # Show truncated output in the UI trace, but pass the full result
                # to messages so the LLM has complete context for its next reasoning step.
                display = result[:500] + "…" if len(result) > 500 else result
                yield {"type": "tool_result", "tool": tc["name"], "output": display}
                messages.append(self._build_tool_result(tc["id"], tc["name"], result))

        # MAX_ITERATIONS hit — LLM never voluntarily stopped, but it has gathered
        # context from all its tool calls. Force a final answer from that context
        # rather than returning silence.
        async for token in self._stream_final_answer(messages, mcp_tools):
            yield {"type": "token", "text": token}
        # Emit any collected sources even when we hit the iteration cap
        if collected_sources:
            yield {"type": "sources", "sources": list(collected_sources.values())}
        yield {"type": "done", "iterations": self.MAX_ITERATIONS}

    async def _stream_final_answer(self, messages: list, mcp_tools: list) -> AsyncIterator[str]:
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

        Why tool_choice="none" with tools present?
          Passing tools=[] (no tools) causes some models to still emit XML tool calls
          mid-answer because the conversation history contains tool patterns — the model
          learned to call tools and keeps doing it. Passing the actual tool list with
          tool_choice="none" is more explicit: the model knows the tools exist but is
          forced to answer in plain text. This kills XML generation reliably.
        """
        tools = self._format_tools(mcp_tools)

        # Explicit "answer now" instruction: appended as a user turn so the model
        # sees it as a direct instruction rather than ambient system context.
        # Without this, models sometimes generate another round of tool calls.
        final_messages = messages + [{
            "role":    "user",
            "content": "Based on the information gathered above, provide a comprehensive answer to the original question. Do not call any more tools.",
        }]

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_sync():
            try:
                if self._provider in ("groq", "gemini", "openrouter"):
                    stream = self._client.chat.completions.create(
                        model=self._model,
                        max_tokens=2048,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + final_messages,
                        tools=tools,
                        tool_choice="none",  # know about tools but forced to answer in text
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            loop.call_soon_threadsafe(queue.put_nowait, delta)
                else:
                    # Anthropic: omit tools for the final answer (no XML problem with Anthropic)
                    with self._client.messages.stream(
                        model=self._model,
                        max_tokens=2048,
                        system=SYSTEM_PROMPT,
                        messages=final_messages,
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

    def _try_fallback(self) -> bool:
        """Switch to the next provider if the current one is quota-exhausted.

        Fallback order: gemini → openrouter → anthropic (Groq intentionally skipped).
        Groq generates hermes-format tool calls that its own API rejects with 400.
        """
        if self._provider == "gemini" and settings.openrouter_api_key:
            self._client   = _openrouter_client(settings.openrouter_api_key)
            self._provider = "openrouter"
            self._model    = _OPENROUTER_MODEL
            print(f"AgentService: Gemini limit hit — switched to OpenRouter ({_OPENROUTER_MODEL})")
            return True
        if self._provider in ("gemini", "openrouter") and settings.anthropic_api_key:
            import anthropic
            self._client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._provider = "anthropic"
            self._model    = "claude-haiku-4-5-20251001"
            print("AgentService: switched to Anthropic as final fallback")
            return True
        return False

    def _format_tools(self, mcp_tools: list) -> list:
        """Convert MCP tool definitions to provider-specific format."""
        if self._provider in ("groq", "gemini", "openrouter"):
            return self.mcp.tools_as_openai_format(mcp_tools)
        else:
            return self.mcp.tools_as_anthropic_format(mcp_tools)

    def _call_llm(self, messages: list, mcp_tools: list) -> dict:
        """
        Make one LLM call and return a normalised step dict.

        Accepts raw mcp_tools (not pre-formatted) so that if _try_fallback() switches
        providers mid-run, the recursive retry automatically reformats for the new
        provider. Pre-formatting outside this method caused Anthropic to reject
        OpenAI-format tools after a Gemini → Anthropic fallback.

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
        tools = self._format_tools(mcp_tools)  # format for current provider
        try:
            if self._provider in ("groq", "gemini", "openrouter"):
                return self._call_groq(messages, tools)
            else:
                return self._call_anthropic(messages, tools)
        except Exception as e:
            if _is_exhausted(e) and self._try_fallback():
                return self._call_llm(messages, mcp_tools)  # reformat + retry with new provider
            raise

    def _call_groq(self, messages: list, tools: list) -> dict:
        """
        Call Groq (OpenAI-compatible API).

        Key Groq specifics:
          - Tool arguments arrive as JSON strings → json.loads() needed
          - finish_reason="tool_calls" → more tools to run
          - finish_reason="stop" → done, answer in msg.content
          - Tool results: role="tool" with tool_call_id, one per call
        """
        extra = {}
        if self._provider == "groq":
            # parallel_tool_calls=False prevents llama from emitting the broken
            # <function=name{...}> hermes format instead of proper OpenAI JSON.
            # Gemini doesn't have this issue.
            extra["parallel_tool_calls"] = False

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=tools,
            tool_choice="auto",
            **extra,
        )

        choice = response.choices[0]
        msg    = choice.message

        if not msg.tool_calls or choice.finish_reason == "stop":
            # Some models fall back to XML tool calls mid-conversation instead of
            # using the OpenAI JSON format. Parse and execute them if present.
            if msg.content:
                xml_calls = _parse_xml_tool_calls(msg.content)
                if xml_calls:
                    # Rewrite history with proper OpenAI tool_calls format so the
                    # role="tool" result messages that follow are coherent.
                    assistant_message = {
                        "role":    "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id":       tc["id"],
                                "type":     "function",
                                "function": {
                                    "name":      tc["name"],
                                    "arguments": json.dumps(tc["input"]),
                                },
                            }
                            for tc in xml_calls
                        ],
                    }
                    return {
                        "done":              False,
                        "answer":            None,
                        "tool_calls":        xml_calls,
                        "assistant_message": assistant_message,
                    }
            return {
                "done":              True,
                "answer":            msg.content or "",
                "tool_calls":        [],
                "assistant_message": None,
            }

        tool_calls = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                # Some models (especially on degraded free-tier instances) return
                # malformed JSON in the arguments field. Log and skip rather than
                # crashing the entire agent run.
                print(f"AgentService: malformed tool args for {tc.function.name}: {tc.function.arguments!r}")
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})

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

    def _build_initial_messages(self, question: str, repo_filter: str | None, history: list[dict] | None = None) -> list:
        # Prepend prior conversation turns so the agent has follow-up context.
        # History items are bare {role, content} dicts — the agent can re-search
        # any code it needs, so we don't need to re-attach retrieved context.
        messages = [{"role": h["role"], "content": h["content"]} for h in (history or [])]
        content = question
        if repo_filter:
            content += f"\n\n(Focus search on repo: {repo_filter})"
        else:
            # Cross-repo mode: tell the agent it can search across all indexed repos
            # and should explicitly mention which repo each finding comes from.
            content += "\n\n(Searching across all indexed repos. For each finding, mention which repo it comes from.)"
        messages.append({"role": "user", "content": content})
        return messages

    def _build_tool_result(self, tool_id: str, tool_name: str, result: str) -> dict:
        """
        Format a tool result for the conversation history.

        The two providers expect completely different formats for tool results:

        Groq/OpenAI: one message per result, role="tool"
          {"role": "tool", "tool_call_id": id, "content": result}

        Anthropic: all results in one user turn, role="user"
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, ...}]}
        """
        if self._provider in ("groq", "gemini", "openrouter"):
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

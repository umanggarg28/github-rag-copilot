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
        return _strip_xml_tags(content)
    if isinstance(content, list):
        # Anthropic format: list of content blocks, some with type="text"
        parts = [
            b.text.strip()
            for b in content
            if hasattr(b, "type") and b.type == "text" and b.text
        ]
        return _strip_xml_tags(" ".join(parts))
    return ""


import re as _re
_XML_TAG_RE = _re.compile(r"<(thought|plan|thinking|reflection)>(.*?)</(thought|plan|thinking|reflection)>", _re.DOTALL | _re.IGNORECASE)

def _strip_xml_tags(text: str) -> str:
    """
    Unwrap <thought>, <plan>, <thinking> XML blocks emitted by some models.

    Keep the inner content — the tags are just wrapper markup, not the thinking itself.
    E.g. "<thought>I should search for X</thought>" → "I should search for X"

    This is different from the final-answer stream filter which strips thought blocks
    entirely (those are internal monologue not meant for the user).
    """
    return _XML_TAG_RE.sub(lambda m: m.group(2), text).strip()


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


def _parse_qwen_tool_calls(content: str) -> list[dict] | None:
    """
    Parse Qwen3/Kimi-style tool calls emitted as special tokens in plain text.

    Some models (Qwen3, Kimi K2) use a token-delimited format instead of the
    OpenAI tool_calls JSON field:

        <|tool_calls_section_begin|>
        <|tool_call_begin|>functions.search_code:0<|tool_call_argument_begin|>
        {"query": "backward pass"}
        <|tool_call_end|>
        <|tool_calls_section_end|>

    We extract the tool name and JSON args and normalise to the same dict format
    as _parse_xml_tool_calls so the rest of the pipeline is unaffected.
    """
    matches = re.findall(
        r'<\|tool_call_begin\|>functions\.(\w+):\d+<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>',
        content, re.DOTALL
    )
    if not matches:
        return None
    result = []
    for i, (name, args_str) in enumerate(matches):
        try:
            params = json.loads(args_str.strip())
        except json.JSONDecodeError:
            params = {}
        result.append({"id": f"call_qwen_{i}_{name}", "name": name, "input": params})
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
# ── Model catalog ─────────────────────────────────────────────────────────────
# Each entry describes a model the user can select from the UI.
# "requires" is the settings key that must be non-empty for this model to appear.
# "provider" must match the strings used in _call_groq / _call_anthropic routing.
AGENT_MODELS: list[dict] = [
    {
        "id":          "cerebras/qwen3-235b",
        "name":        "Qwen3 235B",
        "provider":    "cerebras",
        "model":       "qwen-3-235b-a22b-instruct-2507",
        "requires":    "cerebras_api_key",
        "speed":       "fast",
        "speed_label": "~40s",
        "note":        "Best balance. Fast inference (1400 tok/s), strong tool use, generous free quota.",
    },
    {
        "id":          "google/gemma4-31b",
        "name":        "Gemma 4 31B",
        "provider":    "gemini",
        "model":       "gemma-4-31b-it",
        "requires":    "gemini_api_key",
        "speed":       "slow",
        "speed_label": "~90s",
        "note":        "Highest quality. Reads actual source files. Slower but thorough. Free via AI Studio.",
    },
    {
        "id":          "google/gemini-flash",
        "name":        "Gemini 2.0 Flash",
        "provider":    "gemini",
        "model":       "gemini-2.0-flash",
        "requires":    "gemini_api_key",
        "speed":       "fast",
        "speed_label": "~15s",
        "note":        "Fastest. Lower quality than Gemma 4. 1500 req/day free limit.",
    },
]

def _make_client(model_entry: dict):
    """Instantiate the right API client for a model catalog entry."""
    from openai import OpenAI
    if model_entry["provider"] == "cerebras":
        return OpenAI(api_key=settings.cerebras_api_key, base_url="https://api.cerebras.ai/v1")
    else:  # gemini
        return OpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

_OPENROUTER_MODEL = "qwen/qwen3-coder:free"

# Groq models tried in order when the primary is over capacity or decommissioned.
# Kimi K2 → Llama 4 Scout → Qwen3 32B
_GROQ_MODELS = ["moonshotai/kimi-k2-instruct", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b"]


def _openrouter_client(api_key: str):
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Cartographer",
        },
    )


SYSTEM_PROMPT = """You are an expert code navigator — a senior engineer who answers questions about GitHub repositories by reading the actual source code, never from memory or assumption. You are pair programming with the user: your job is to find the truth in the code, not to guess.

NEVER claim a function, class, or file exists without having read it using a tool.
NEVER invent method signatures, parameter names, or return types — read the source first.
NEVER say something "likely" or "probably" works a certain way. Either read the code and say what it does, or say you haven't found it yet.

═══════════════════════════════
REPO MAP — ALWAYS READ FIRST
═══════════════════════════════

When answering about a specific repo, a ╔══ REPO MAP ══╗ block appears in the
user's message. It tells you the entry files, key classes, and top files.
Use it to skip list_files — you already know the repo layout.

═══════════════════════════════
PLAN BEFORE ACTING
═══════════════════════════════

Before your FIRST tool call each session, write a 2-3 line plan:
  <plan>
  Goal: [what I need to find]
  Search 1: [first tool + query]
  Search 2: [second tool + query, if needed]
  </plan>

This appears as your first thought bubble — it shows users your reasoning
before you execute. Keep it short. Don't plan beyond the next 2-3 steps.

═══════════════════════════════
WORKING MEMORY — USE IT
═══════════════════════════════

  note("key", "value")   → Record a discovery immediately when you find it:
                            note("entry_point", "train.py:main() L234")
                            note("main_class",  "GPT in model.py — handles forward pass")
                            note("data_flow",   "DataLoader → forward → loss → backward")

  recall_notes()         → Read everything you've noted. Call this:
                            - Before your FIRST tool call (check what you already know)
                            - Before writing your final answer (check nothing was missed)

═══════════════════════════════
TOOL SELECTION GUIDE
═══════════════════════════════

  recall_notes()
    → Always call FIRST. You may already know the answer.

  list_files(repo, path="")
    → Only if the repo map doesn't give enough orientation.

  search_code(query, repo, mode)
    → Semantic + keyword search. For multi-part questions, call 2-3 searches
      in a SINGLE turn — they run in parallel and don't waste extra iterations.
      e.g. search_code("forward pass") + search_code("loss function") together.

  search_symbol(symbol_name, repo)
    → Exact lookup by name. Faster and more precise than search_code for known names.

  trace_calls(repo, symbol_name, max_depth=3)
    → Follow an execution path end-to-end. "What does train() actually do?"

  find_callers(function_name, repo)
    → Who calls X? Use to understand usage patterns and dependency direction.

  read_file(repo, filepath)
    → Read an entire file. Use for imports, module structure, full class context.

  get_file_chunk(repo, filepath, start_line, end_line)
    → Read specific lines. Use when search already gave you line numbers.

  draw_diagram(description, diagram_type)
    → Call this AFTER your research is complete (after search_code/search_symbol/read_file),
      just before writing the diagram block. Never skip the research step — the diagram
      must be grounded in what you actually found in the codebase.
      Output the diagram ONLY as a ```diagram``` fenced block with Mermaid syntax —
      NEVER as a plain ```mermaid``` or ```classDiagram``` block. Pick whatever type
      fits best: flowchart, classDiagram, sequenceDiagram, stateDiagram-v2, erDiagram,
      mindmap, timeline, etc.

═══════════════════════════════
STRATEGY
═══════════════════════════════

  1. READ MAP  — check the ╔══ REPO MAP ══╗ and ╔══ REPO PURPOSE ══╗ in the user message
  2. RECALL    — recall_notes() to see what you already know this session
  3. PLAN      — write a <plan> block with your first 2-3 search steps
  4. FIND      — fire ALL searches for the same question in ONE turn (parallel execution)
                 e.g. search_code("forward pass") + search_code("loss function") together
                 NEVER send one search, wait, then send the next — that wastes turns
  5. DRILL     — search_symbol for exact names found in step 4; find_callers/trace_calls for paths
  6. NOTE      — note() every key discovery immediately after finding it
  7. ANSWER    — recall_notes() to compile, then cite file + line for every claim

═══════════════════════════════
RULES
═══════════════════════════════

- Read REPO PURPOSE + REPO MAP before anything else — skip list_files if the map covers it
- Always call recall_notes() before your first search and before your final answer
- Write a <plan> before your first tool call every session
- PARALLEL: group all searches covering the same question into one turn — they execute concurrently
- Note discoveries immediately — don't rely on scrolling back through results
- If search_code gives you a name, use search_symbol to get the full definition
- Stop when you have enough — over-searching wastes turns
- If something isn't in the index after 3 targeted searches, say so clearly
- Cite every claim: file path + function name + line numbers
- DIAGRAMS: search the codebase first (search_code/search_symbol/read_file), THEN call draw_diagram() after research is complete, then write 1-3 sentences describing the diagram, then output as ```diagram``` — never as a plain mermaid/code block"""


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

    MAX_ITERATIONS = 20  # increased from 12 — complex repos need more search turns

    def __init__(self, mcp_client: MCPClient, repo_map_svc=None):
        """
        Args:
            mcp_client:   Connected MCPClient pointing to our FastMCP server.
                          Tools are discovered lazily on first run/stream call.
            repo_map_svc: Optional RepoMapService. When provided, a compact repo
                          map is injected into the user message at session start
                          so the agent skips the list_files orientation step.
        """
        self.mcp          = mcp_client
        self._repo_map    = repo_map_svc

        # ── Provider detection ─────────────────────────────────────────────────
        # Priority: Cerebras (Qwen3-235B) → Gemini → OpenRouter → Anthropic → Groq.
        #
        # Cerebras qwen-3-235b-a22b-instruct-2507: confirmed tool calling, 1400 tok/s.
        # It's a 235B MoE model (22B active params) — strong reasoning, reliable tools.
        # Status is "preview" on Cerebras so Gemini is the stable fallback.
        # llama3.1-8b was tested and skips tools on complex prompts — not used here.
        #
        # Groq last: hermes-format tool-call bug causes occasional 400s.
        if settings.cerebras_api_key:
            from openai import OpenAI
            self._client   = OpenAI(
                api_key=settings.cerebras_api_key,
                base_url="https://api.cerebras.ai/v1",
            )
            self._provider = "cerebras"
            self._model    = "qwen-3-235b-a22b-instruct-2507"
            print("AgentService: using Cerebras (qwen-3-235b, 1400 tok/s) via MCP tools")
        elif settings.gemini_api_key:
            from openai import OpenAI
            self._client   = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._provider = "gemini"
            self._model    = "gemma-4-31b-it"
            print("AgentService: using Gemma 4 31B (gemma-4-31b-it) via MCP tools")
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
            self._model    = "moonshotai/kimi-k2-instruct"
            print("AgentService: using Groq (moonshotai/kimi-k2-instruct) via MCP tools [kimi-k2 fallback]")
        else:
            raise ValueError("AgentService requires CEREBRAS_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY")

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
        self,
        question: str,
        repo_filter: str | None = None,
        history: list[dict] | None = None,
        model_id: str | None = None,
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
        # ── Per-request model override ────────────────────────────────────────
        # If the user selected a specific model in the UI, temporarily swap to it.
        # We save/restore self._client/provider/model in a finally block so the
        # default priority chain is preserved for the next request.
        _orig = (self._client, self._provider, self._model)
        entry = next((m for m in AGENT_MODELS if m["id"] == model_id), None)
        if entry:
            self._client   = _make_client(entry)
            self._provider = entry["provider"]
            self._model    = entry["model"]

        try:
            # Discover tools from MCP server (cached after first call)
            mcp_tools = await self.mcp.list_tools()
            messages  = self._build_initial_messages(question, repo_filter, history)

            # Clear in-memory notes and pre-load any persisted notes for this repo.
            # Passing repo_filter lets clear_notes() hydrate working memory from
            # Qdrant so the agent has cross-session recall from the first iteration.
            from backend.mcp_server import clear_notes
            clear_notes(repo=repo_filter)

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
                    yield {"type": "done", "iterations": iteration + 1, "model": self._model}
                    return

                messages.append(step["assistant_message"])

                # Emit any pre-tool reasoning text the LLM produced before calling tools.
                # This lets the UI show "thought bubbles" in the trace timeline —
                # the user sees WHY each tool was chosen, not just WHAT was called.
                thought = _extract_thought(step["assistant_message"], self._provider)
                if thought:
                    yield {"type": "thought", "text": thought}

                # ── Parallel tool execution ───────────────────────────────────────
                # The LLM may return multiple tool calls in one turn (e.g. search_code
                # called 2-3 times for different query angles simultaneously).
                # Instead of serial execution, we:
                #   1. Emit tool_call events for all new (non-duplicate) calls upfront
                #   2. Run them concurrently with asyncio.gather
                #   3. Emit tool_result events for all after they complete
                #
                # This reduces latency proportionally to the number of parallel calls
                # (3 serial 500ms searches → 1 parallel 500ms round trip).

                # Separate new calls from duplicates
                new_calls: list[dict] = []
                for tc in step["tool_calls"]:
                    call_key = (tc["name"], tuple(sorted(tc["input"].items())))
                    if call_key in seen_calls:
                        dup_msg = f"[Skipped duplicate {tc['name']} call — already ran with these arguments]"
                        yield {"type": "tool_result", "tool": tc["name"], "output": dup_msg}
                        messages.append(self._build_tool_result(tc["id"], tc["name"], dup_msg))
                    else:
                        seen_calls.add(call_key)
                        new_calls.append(tc)
                        # Emit tool_call events immediately so UI shows them in parallel
                        yield {"type": "tool_call", "tool": tc["name"], "input": tc["input"]}

                if not new_calls:
                    continue

                # Execute all new calls concurrently — MCP calls are async HTTP round trips
                async def _run_tool(tc: dict) -> str:
                    # Retry once on transient MCP connection failures (TaskGroup /
                    # HTTP errors from the SDK's internal connection management).
                    for attempt in range(2):
                        try:
                            return await self.mcp.call_tool(tc["name"], tc["input"])
                        except Exception as e:
                            if attempt == 0 and "TaskGroup" in str(e):
                                await asyncio.sleep(0.3)
                                continue
                            return f"Tool error: {e}"

                parallel_results = await asyncio.gather(*[_run_tool(tc) for tc in new_calls])

                # Process results in the same order as the calls
                for tc, result in zip(new_calls, parallel_results):
                    # Collect source metadata for the sources panel
                    if tc["name"] == "get_file_chunk":
                        src = _source_from_chunk_call(tc["input"], result)
                        if src:
                            key = (src["repo"], src["filepath"], src["start_line"])
                            collected_sources[key] = src

                    if tc["name"] in ("search_code", "find_callers", "search_symbol") and not result.startswith("No results"):
                        for src in _sources_from_search_result(result, tc["input"].get("repo") or repo_filter):
                            key = (src["repo"], src["filepath"], src["start_line"])
                            collected_sources[key] = src

                    # read_file returns a whole file — record it as a single source entry
                    if tc["name"] == "read_file" and tc["input"].get("filepath"):
                        repo     = tc["input"].get("repo", repo_filter or "")
                        filepath = tc["input"]["filepath"]
                        key = (repo, filepath, 0)
                        if key not in collected_sources:
                            ext = "." + filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
                            lang = {"py": "python", "js": "javascript", "ts": "typescript",
                                    "go": "go", "rs": "rust", "java": "java"}.get(ext.lstrip("."), "text")
                            collected_sources[key] = {
                                "repo": repo, "filepath": filepath, "language": lang,
                                "chunk_type": "file", "name": filepath.rsplit("/", 1)[-1],
                                "start_line": 1, "end_line": result.count("\n"),
                                "score": 1.0, "text": result,
                            }

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
            yield {"type": "done", "iterations": self.MAX_ITERATIONS, "model": self._model}

        finally:
            # Restore original client/provider/model so the next request uses the
            # default priority chain regardless of what model was selected this time.
            self._client, self._provider, self._model = _orig

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
                if self._provider in ("cerebras", "groq", "gemini", "openrouter"):
                    stream = self._client.chat.completions.create(
                        model=self._model,
                        max_tokens=4096,  # increased: complex answers need room
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
                        max_tokens=4096,
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

        # Consume tokens as they arrive from the background thread.
        # Some models (Gemma 4) emit <thought>...</thought> tags at the start
        # of their final answer. We strip them here with a stateful buffer so
        # the UI never renders raw XML thought tags.
        buf        = ""   # accumulates partial text while we check for tags
        in_thought = False
        OPEN_TAG   = "<thought>"
        CLOSE_TAG  = "</thought>"

        while True:
            token = await queue.get()
            if token is None:
                # Flush whatever is buffered (can't be inside a tag at EOF)
                if buf and not in_thought:
                    yield buf
                break

            buf += token

            # Process buf until no more complete decisions can be made
            while buf:
                if in_thought:
                    # Looking for </thought>
                    idx = buf.find(CLOSE_TAG)
                    if idx != -1:
                        # Found the close tag — discard everything up to and including it
                        buf = buf[idx + len(CLOSE_TAG):]
                        in_thought = False
                    else:
                        # Might be a partial </thought> at the end — keep the last
                        # len(CLOSE_TAG)-1 chars buffered in case the tag spans chunks
                        safe = len(buf) - (len(CLOSE_TAG) - 1)
                        if safe > 0:
                            buf = buf[safe:]  # discard confirmed-inside-thought text
                        break
                else:
                    # Looking for <thought>
                    idx = buf.find(OPEN_TAG)
                    if idx == 0:
                        # Tag starts right here — enter thought mode, discard the tag
                        buf = buf[len(OPEN_TAG):]
                        in_thought = True
                    elif idx > 0:
                        # Emit everything before the tag, then enter thought mode
                        yield buf[:idx]
                        buf = buf[idx + len(OPEN_TAG):]
                        in_thought = True
                    else:
                        # No open tag found — safe to emit, but keep a small tail
                        # in case <thought> is split across chunks
                        safe = len(buf) - (len(OPEN_TAG) - 1)
                        if safe > 0:
                            yield buf[:safe]
                            buf = buf[safe:]
                        break

        await task  # re-raises any exception from the streaming thread

    # ── LLM dispatch ───────────────────────────────────────────────────────────

    def _try_groq_model_fallback(self) -> bool:
        """Within Groq, cycle to the next model when the current one is over capacity."""
        if self._provider != "groq":
            return False
        try:
            idx = _GROQ_MODELS.index(self._model)
        except ValueError:
            idx = -1
        if idx < len(_GROQ_MODELS) - 1:
            next_model = _GROQ_MODELS[idx + 1]
            print(f"AgentService: Groq {self._model} over capacity — trying {next_model}")
            self._model = next_model
            return True
        return False

    def _try_fallback(self) -> bool:
        """Switch to the next provider if the current one is quota-exhausted.

        Fallback order: cerebras → gemini → openrouter → anthropic → groq (last resort).
        Groq has a hermes-format tool-call bug that causes occasional 400s,
        but it's better than returning nothing when every other provider is exhausted.
        """
        if self._provider == "cerebras" and settings.gemini_api_key:
            from openai import OpenAI
            self._client   = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._provider = "gemini"
            self._model    = "gemma-4-31b-it"
            print("AgentService: Cerebras limit hit — switched to Gemma 4 31B (gemma-4-31b-it)")
            return True
        if self._provider in ("cerebras", "gemini") and settings.openrouter_api_key:
            self._client   = _openrouter_client(settings.openrouter_api_key)
            self._provider = "openrouter"
            self._model    = _OPENROUTER_MODEL
            print(f"AgentService: Gemini limit hit — switched to OpenRouter ({_OPENROUTER_MODEL})")
            return True
        if self._provider in ("cerebras", "gemini", "openrouter") and settings.anthropic_api_key:
            import anthropic
            self._client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._provider = "anthropic"
            self._model    = "claude-haiku-4-5-20251001"
            print("AgentService: switched to Anthropic as final fallback")
            return True
        if self._provider in ("cerebras", "gemini", "openrouter", "anthropic") and settings.groq_api_key:
            from groq import Groq
            self._client   = Groq(api_key=settings.groq_api_key)
            self._provider = "groq"
            self._model    = "moonshotai/kimi-k2-instruct"
            print("AgentService: switched to Groq as last resort (kimi-k2 fallback)")
            return True
        return False

    def _format_tools(self, mcp_tools: list) -> list:
        """Convert MCP tool definitions to provider-specific format."""
        if self._provider in ("cerebras", "groq", "gemini", "openrouter"):
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
            if self._provider in ("cerebras", "groq", "gemini", "openrouter"):
                return self._call_groq(messages, tools)
            else:
                return self._call_anthropic(messages, tools)
        except Exception as e:
            msg = str(e).lower()
            # Groq's hermes bug: model generates <function=name{...}> format which
            # Groq's own API rejects with 400 "tool call validation failed".
            # Treat this the same as exhaustion — skip Groq and try the next provider.
            is_groq_hermes = self._provider == "groq" and "tool call validation" in msg
            # Groq model unavailable: over capacity (503) or decommissioned (400).
            # Try the next Groq model before giving up on Groq entirely.
            is_groq_model_unavailable = self._provider == "groq" and (
                "over capacity" in msg or "decommissioned" in msg or "model_decommissioned" in msg
            )
            if is_groq_model_unavailable and self._try_groq_model_fallback():
                return self._call_llm(messages, mcp_tools)
            if (is_groq_hermes or _is_exhausted(e)) and self._try_fallback():
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
            max_tokens=1024,  # tool-calling turns only need short reasoning + tool name
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
                xml_calls = _parse_xml_tool_calls(msg.content) or _parse_qwen_tool_calls(msg.content)
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
            max_tokens=1024,  # tool-calling turns only need short reasoning + tool name
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
            # Inject two layers of repo context so the agent starts informed:
            #
            # 1. README summary — the repo's STATED PURPOSE (what it's for).
            #    Without this, the agent treats a RAG system and a game engine
            #    identically — both are "just files". The README anchors every
            #    search in intent, not just structure.
            #
            # 2. Repo map — STRUCTURAL metadata (entry files, key classes).
            #    Lets the agent skip list_files and go straight to targeted searches.
            #
            # Combined: the agent knows what the repo does AND where to find it.
            if self._repo_map:
                try:
                    readme_summary = self._get_readme_summary(repo_filter)
                    if readme_summary:
                        content = (
                            f"╔══ REPO PURPOSE ══╗\n{readme_summary}\n╚══════════════════╝\n\n"
                            + content
                        )

                    repo_map = self._repo_map.get_or_build(repo_filter)
                    map_text = self._repo_map.format_for_prompt(repo_map)
                    if map_text:
                        content = map_text + "\n\n" + content
                except Exception as e:
                    print(f"AgentService: context injection failed (non-fatal): {e}")

            content += f"\n\n(Focus search on repo: {repo_filter})"
        else:
            # Cross-repo mode: tell the agent it can search across all indexed repos
            # and should explicitly mention which repo each finding comes from.
            content += "\n\n(Searching across all indexed repos. For each finding, mention which repo it comes from.)"

        messages.append({"role": "user", "content": content})
        return messages

    def _get_readme_summary(self, repo: str) -> str:
        """
        Extract the project purpose sentence from the indexed README.

        Strategy: strip markdown noise (badges, links, headings), then return
        the first substantive sentence — typically the one-liner that says what
        the project does. Cap at 200 chars.

        Why 200 chars (not 400)?
        The README's first meaningful sentence is almost always under 150 chars.
        400 chars frequently captures CI badge rows, table-of-contents links,
        or boilerplate that precedes the actual description. We want the purpose
        statement, not the decoration around it.

        The full README goes to the tour agent's Phase 0; this is just a
        grounding hint so the agent knows the repo's intent before searching.
        """
        if not self._repo_map or not hasattr(self._repo_map, '_store'):
            return ""
        try:
            import re as _re
            store = self._repo_map._store
            all_chunks = store.scroll_repo(repo)
            readme_chunks = []
            for p in all_chunks:
                fp = (p.get("filepath") or "").lower()
                fname = fp.split("/")[-1]
                if fname.startswith("readme") or fname in ("index.md", "overview.md"):
                    readme_chunks.append(p)

            # Prefer root-level README over nested documentation files
            readme_chunks.sort(key=lambda c: c.get("filepath", "").count("/"))
            if not readme_chunks:
                return ""

            text = (readme_chunks[0].get("text") or "").strip()

            # Remove markdown badge lines: [![...](...)(...)] and [!badge] patterns
            text = _re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
            text = _re.sub(r'!\[.*?\]\(.*?\)', '', text)
            # Remove bare markdown links [text](url) — keep the text
            text = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            # Strip heading markers
            text = _re.sub(r'^#+\s+', '', text, flags=_re.MULTILINE)
            # Collapse multiple blank lines
            text = _re.sub(r'\n{3,}', '\n\n', text)

            # Find the first line with ≥20 chars that looks like a description
            # (not a badge row, not a pure URL, not just punctuation/whitespace)
            for line in text.splitlines():
                line = line.strip()
                if len(line) >= 20 and not line.startswith('http') and not line.startswith('|'):
                    return line[:200]

            # Fallback: return whatever is left, capped at 200
            return text.strip()[:200]
        except Exception:
            return ""

    def _build_tool_result(self, tool_id: str, tool_name: str, result: str) -> dict:
        """
        Format a tool result for the conversation history.

        The two providers expect completely different formats for tool results:

        Groq/OpenAI: one message per result, role="tool"
          {"role": "tool", "tool_call_id": id, "content": result}

        Anthropic: all results in one user turn, role="user"
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, ...}]}
        """
        if self._provider in ("cerebras", "groq", "gemini", "openrouter"):
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

"""
agent.py — Agentic RAG using tool use (Groq or Anthropic).

═══════════════════════════════════════════════════════════════
WHAT IS AN AGENT? (vs plain RAG)
═══════════════════════════════════════════════════════════════

Plain RAG (what we had before):
  Query → single retrieval → LLM → answer

  The problem: one retrieval step may miss critical context.
  "How does the training loop work?" retrieves train() but misses
  the DataLoader, the gradient accumulation, the optimizer step.
  One shot, then done.

Agentic RAG (what we're building):
  Query → think → search → observe → think → search → observe → answer

  The LLM DECIDES when it has enough information.
  It can call tools multiple times, from different angles,
  until it's confident in its answer.

This is called a ReAct loop (Reason + Act):
  1. REASON: "I need to find the backward() implementation"
  2. ACT:    call search_code("backward implementation")
  3. OBSERVE: "I see relu._backward, but not the main backward()"
  4. REASON: "Let me search specifically for Value.backward"
  5. ACT:    call find_callers("backward")
  6. OBSERVE: "Found it — it does topological sort first"
  7. REASON: "I have enough to answer"
  8. RESPOND: full answer with citations

═══════════════════════════════════════════════════════════════
TWO PROVIDERS, SAME INTERFACE
═══════════════════════════════════════════════════════════════

This agent supports two LLM providers, both supporting tool use:

  GROQ (primary — free tier)
    Model: llama-3.3-70b-versatile
    API format: OpenAI-compatible (same as openai SDK)
    Tool format: {"type": "function", "function": {"name": ..., "parameters": ...}}
    Tool results: role="tool", one message per result

  ANTHROPIC (fallback — paid)
    Model: claude-haiku-4-5-20251001
    API format: Anthropic Messages API
    Tool format: {"name": ..., "input_schema": ...}
    Tool results: role="user", content=[{type: "tool_result", ...}]

The two APIs differ significantly in wire format but the logic is identical.
We abstract the differences behind _run_loop() which handles both.

Why Groq for tool use?
  Groq runs Llama 3.3 70B on custom inference chips (LPUs) at very high speed.
  It's free up to generous rate limits. Llama 3.3 supports OpenAI-compatible
  function calling, which means the same tool definitions work for both
  Groq and any other OpenAI-compatible provider.

═══════════════════════════════════════════════════════════════
OPENAI-COMPATIBLE TOOL USE (Groq format)
═══════════════════════════════════════════════════════════════

Tool definition:
  {
    "type": "function",
    "function": {
      "name": "search_code",
      "description": "...",
      "parameters": { "type": "object", "properties": {...}, "required": [...] }
    }
  }

Making a call:
  response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=messages,
      tools=TOOLS_OPENAI,
      tool_choice="auto",
  )

Checking if done:
  msg = response.choices[0].message
  if response.choices[0].finish_reason == "stop" or not msg.tool_calls:
      answer = msg.content  # final answer

Getting tool calls:
  for tc in msg.tool_calls:
      name = tc.function.name
      args = json.loads(tc.function.arguments)  # NOTE: JSON string, not dict
      id   = tc.id

Tool result message:
  {"role": "tool", "tool_call_id": tc.id, "content": result_string}
  One message per tool call (unlike Anthropic where all results go in one user turn).

═══════════════════════════════════════════════════════════════
STOPPING CONDITIONS
═══════════════════════════════════════════════════════════════

The loop ends when:
  1. The model returns finish_reason="stop" (or "end_turn" for Anthropic)
  2. The model returns text with no tool calls
  3. We hit MAX_ITERATIONS (safety cap — prevents infinite loops)

We cap at 8 iterations. Most questions need 2–4 hops.
"""

import json
from pathlib import Path
from typing import Iterator
import sys

import requests as http_requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings
from retrieval.retrieval import RetrievalService


# ── Tool definitions ───────────────────────────────────────────────────────────
# Defined once in Anthropic format; converted to OpenAI format for Groq.
# The description is what the LLM reads to decide WHEN to call each tool.

_TOOLS_ANTHROPIC = [
    {
        "name": "search_code",
        "description": (
            "Search the indexed GitHub repositories for code relevant to a query. "
            "Uses hybrid BM25 + semantic search. Returns ranked code chunks with "
            "file paths, function names, and line numbers. "
            "Call this first when answering any question about the codebase. "
            "You can call it multiple times with different queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "repo":  {"type": "string", "description": "Optional: 'owner/repo' to restrict search"},
                "mode":  {
                    "type": "string",
                    "enum": ["hybrid", "semantic", "keyword"],
                    "description": "hybrid=default, keyword=exact identifiers, semantic=concepts",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_file_chunk",
        "description": (
            "Fetch raw content of a specific file section from GitHub. "
            "Use when search returns a function but you need more context: "
            "the lines above (docstring) or below (what follows)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":       {"type": "string", "description": "'owner/repo'"},
                "filepath":   {"type": "string", "description": "path within the repo"},
                "start_line": {"type": "integer"},
                "end_line":   {"type": "integer"},
            },
            "required": ["repo", "filepath", "start_line", "end_line"],
        },
    },
    {
        "name": "find_callers",
        "description": (
            "Find all places in the codebase that call a specific function or class. "
            "Essential for understanding HOW something is used, not just what it does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {"type": "string"},
                "repo":          {"type": "string", "description": "Optional: restrict to one repo"},
            },
            "required": ["function_name"],
        },
    },
]

# Convert Anthropic tool format → OpenAI/Groq format.
# The only structural difference: "input_schema" → "parameters", wrapped in "function".
_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name":        t["name"],
            "description": t["description"],
            "parameters":  t["input_schema"],
        },
    }
    for t in _TOOLS_ANTHROPIC
]


SYSTEM_PROMPT = """You are an expert code assistant with access to a searchable index of GitHub repositories.

When answering questions about code:
1. Start by calling search_code to find relevant code
2. If the initial results don't fully answer the question, search again with a different query
3. Use get_file_chunk to see more context around a result (e.g., the full class or surrounding code)
4. Use find_callers to understand how functions are used, not just defined
5. Only answer when you have enough evidence from the actual code

Always cite your sources: mention the file path and line numbers.
Be precise — if the code doesn't show what you're looking for, say so rather than guessing."""


class AgentService:
    """
    Runs a ReAct (Reason + Act) loop using tool use.

    Provider auto-detection:
      - Groq available → use Llama 3.3 70B (free, fast)
      - Anthropic available → use Claude Haiku (paid fallback)
      - Neither → raises ValueError at init time

    The same run() / stream() interface works regardless of provider.
    """

    MAX_ITERATIONS = 8

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval = retrieval_service

        if settings.groq_api_key:
            from groq import Groq
            self._client   = Groq(api_key=settings.groq_api_key)
            self._provider = "groq"
            self._model    = "llama-3.3-70b-versatile"
            print("AgentService: using Groq (llama-3.3-70b-versatile)")
        elif settings.anthropic_api_key:
            import anthropic
            self._client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._provider = "anthropic"
            self._model    = "claude-haiku-4-5-20251001"
            print("AgentService: using Anthropic (claude-haiku)")
        else:
            raise ValueError("Agent requires GROQ_API_KEY or ANTHROPIC_API_KEY in .env")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, question: str, repo_filter: str | None = None) -> dict:
        """Run the full ReAct loop and return the final answer + trace."""
        messages = self._build_initial_messages(question, repo_filter)
        tool_trace = []

        for iteration in range(self.MAX_ITERATIONS):
            step = self._call_llm(messages)

            if step["done"]:
                return {
                    "answer":     step["answer"],
                    "tool_calls": tool_trace,
                    "iterations": iteration + 1,
                }

            # Append the assistant's tool-calling turn to history
            messages.append(step["assistant_message"])

            # Execute each tool call and append results
            for tc in step["tool_calls"]:
                try:
                    result = self._execute_tool(tc["name"], tc["input"])
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

    def stream(self, question: str, repo_filter: str | None = None) -> Iterator[dict]:
        """
        Stream agent progress as it happens.

        Yields dicts:
          {"type": "tool_call",   "tool": "search_code", "input": {...}}
          {"type": "tool_result", "tool": "search_code", "output": "..."}
          {"type": "token",       "text": "According..."}
          {"type": "done",        "iterations": 3}
        """
        messages = self._build_initial_messages(question, repo_filter)

        for iteration in range(self.MAX_ITERATIONS):
            step = self._call_llm(messages)

            if step["done"]:
                for word in step["answer"].split(" "):
                    yield {"type": "token", "text": word + " "}
                yield {"type": "done", "iterations": iteration + 1}
                return

            messages.append(step["assistant_message"])

            for tc in step["tool_calls"]:
                yield {"type": "tool_call", "tool": tc["name"], "input": tc["input"]}

                try:
                    result = self._execute_tool(tc["name"], tc["input"])
                except Exception as e:
                    result = f"Tool error: {e}"

                yield {"type": "tool_result", "tool": tc["name"], "output": result[:300]}

                messages.append(self._build_tool_result(tc["id"], tc["name"], result))

        yield {"type": "done", "iterations": self.MAX_ITERATIONS}

    # ── LLM call (provider-agnostic) ───────────────────────────────────────────

    def _call_llm(self, messages: list) -> dict:
        """
        Make one LLM call and return a normalised step dict:
          {
            "done":              bool,
            "answer":            str | None,     # if done
            "tool_calls":        list[dict],     # [{name, input, id}] if not done
            "assistant_message": dict,           # formatted for this provider's history
          }
        """
        if self._provider == "groq":
            return self._call_groq(messages)
        else:
            return self._call_anthropic(messages)

    def _call_groq(self, messages: list) -> dict:
        """
        Call Groq (OpenAI-compatible API) and normalise the response.

        Key differences from Anthropic:
          - tool arguments come as JSON strings (need json.loads)
          - finish_reason="tool_calls" means more tools to run
          - finish_reason="stop" means done
          - tool results go as role="tool" messages (one per call)
        """
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=_TOOLS_OPENAI,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg    = choice.message

        # Done: no tool calls or finish_reason is "stop"
        if not msg.tool_calls or choice.finish_reason == "stop":
            return {
                "done":              True,
                "answer":            msg.content or "",
                "tool_calls":        [],
                "assistant_message": None,
            }

        # Build normalised tool_calls list
        tool_calls = [
            {
                "id":    tc.id,
                "name":  tc.function.name,
                "input": json.loads(tc.function.arguments),
            }
            for tc in msg.tool_calls
        ]

        # Build the assistant message to append to history.
        # Groq/OpenAI expects the raw tool_calls objects, not our normalised dicts.
        assistant_message = {
            "role":       "assistant",
            "content":    msg.content,   # may be None
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

    def _call_anthropic(self, messages: list) -> dict:
        """
        Call Anthropic Messages API and normalise the response.

        Key differences from Groq:
          - tool arguments are already dicts (no json.loads needed)
          - stop_reason="end_turn" means done
          - stop_reason="tool_use" means more tools to run
          - tool results go into a single "user" turn as a list
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=_TOOLS_ANTHROPIC,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            answer = "".join(b.text for b in response.content if hasattr(b, "text"))
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
            content += f"\n\n(Search in repo: {repo_filter})"
        # Groq includes system prompt separately in the API call; Anthropic via `system=`
        # Both use the same user message format here.
        return [{"role": "user", "content": content}]

    def _build_tool_result(self, tool_id: str, tool_name: str, result: str) -> dict:
        """
        Format a tool result for the conversation history.

        Groq/OpenAI: role="tool" with tool_call_id, one message per call
        Anthropic:   role="user" with content=[{type: tool_result, ...}]
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
                "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": result}],
            }

    # ── Tool execution ─────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "search_code":
            return self._tool_search_code(args)
        elif name == "get_file_chunk":
            return self._tool_get_file_chunk(args)
        elif name == "find_callers":
            return self._tool_find_callers(args)
        return f"Unknown tool: {name}"

    def _tool_search_code(self, args: dict) -> str:
        results = self.retrieval.search(
            query=args["query"],
            top_k=int(args.get("top_k", 5)),  # Groq sometimes emits "5" (string)
            repo_filter=args.get("repo"),
            mode=args.get("mode", "hybrid"),
        )
        if not results:
            return "No results found."
        return self.retrieval.format_context(results)

    def _tool_get_file_chunk(self, args: dict) -> str:
        repo     = args["repo"]
        filepath = args["filepath"]
        start    = args["start_line"]
        end      = args["end_line"]
        owner, name = repo.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{name}/contents/{filepath}"
        headers = {"Accept": "application/vnd.github.v3.raw"}
        if settings.github_token:
            headers["Authorization"] = f"token {settings.github_token}"
        resp = http_requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return f"File not found: {filepath}"
        resp.raise_for_status()
        lines = resp.text.splitlines()
        start = max(1, start)
        end   = min(len(lines), end)
        chunk = "\n".join(f"{i+start}: {line}" for i, line in enumerate(lines[start-1:end]))
        return f"# {repo} — {filepath} (lines {start}–{end})\n\n{chunk}"

    def _tool_find_callers(self, args: dict) -> str:
        name = args["function_name"]
        results = self.retrieval.search(
            query=name,
            top_k=8,
            repo_filter=args.get("repo"),
            mode="keyword",
        )
        callers = [r for r in results if name in r["text"]]
        if not callers:
            return f"No call sites found for '{name}'."
        return self.retrieval.format_context(callers)

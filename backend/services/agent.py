"""
agent.py — Agentic RAG using Anthropic tool use.

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
HOW ANTHROPIC TOOL USE WORKS
═══════════════════════════════════════════════════════════════

Normal message:
  You → [message] → Claude → [answer text]

With tools:
  You → [message + tool_definitions] → Claude
    → either: [answer text]   (done, no tools needed)
    → or:     [tool_use block] (Claude wants to call a tool)
  You run the tool → [tool_result] → Claude
    → either: [answer text]
    → or:     [another tool_use block]
  ... repeat until Claude returns text

The conversation history grows:
  messages = [
    {"role": "user",      "content": "How does backward() work?"},
    {"role": "assistant", "content": [{"type": "tool_use", "name": "search_code", ...}]},
    {"role": "user",      "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]},
    {"role": "assistant", "content": [{"type": "tool_use", "name": "find_callers", ...}]},
    {"role": "user",      "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]},
    {"role": "assistant", "content": "According to Source 4, backward() works by..."},
  ]

The key insight: tool results are fed back as "user" messages.
The model never "runs" the tool — YOU do, and report back.

═══════════════════════════════════════════════════════════════
STOPPING CONDITIONS
═══════════════════════════════════════════════════════════════

The loop ends when:
  1. Claude returns stop_reason="end_turn" (it's satisfied)
  2. We hit max_iterations (safety cap — prevents infinite loops)
  3. Claude returns text with no tool calls (it has its answer)

We cap at 8 iterations. Each iteration is one Claude API call + one
tool execution. This bounds cost and latency while allowing real
multi-hop reasoning (most questions need 2–4 hops).
"""

import json
from pathlib import Path
from typing import Iterator
import sys

import requests as http_requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings
from retrieval.retrieval import RetrievalService


# ── Tool definitions (Anthropic format) ───────────────────────────────────────
# These are the same tools as the MCP server but defined in Anthropic's
# tool schema format. Same capabilities, different wire format.
#
# Notice the pattern: name, description (LLM reads this!), input_schema.
# The description tells the LLM WHEN to use the tool. Write it like a
# docstring for the model's benefit, not yours.

TOOLS = [
    {
        "name": "search_code",
        "description": (
            "Search the indexed GitHub repositories for code relevant to a query. "
            "Uses hybrid BM25 + semantic search. Returns ranked code chunks with "
            "file paths, function names, and line numbers. "
            "Call this first when answering any question about the codebase. "
            "You can call it multiple times with different queries to explore different aspects."
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
                "top_k": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_file_chunk",
        "description": (
            "Fetch the raw content of a specific section of a file from GitHub. "
            "Use this when a search result shows a function but you need more context: "
            "the lines above (docstring, decorators) or below (what comes after). "
            "Also useful to see the full class when search only returned one method."
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
            "Essential for understanding HOW something is used, not just what it does. "
            "Example: after finding the definition of Value.__mul__, call find_callers "
            "to see where multiplication is actually performed in training code."
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


class AgentService:
    """
    Runs a ReAct (Reason + Act) loop using Anthropic tool use.

    The agent has access to three tools: search_code, get_file_chunk, find_callers.
    It runs until either it produces an answer or hits max_iterations.

    Each call to `run()` returns a structured result including:
    - The final answer
    - The tool call trace (what it searched, what it found)
    - The sources actually used in the answer
    """

    MAX_ITERATIONS = 8

    SYSTEM_PROMPT = """You are an expert code assistant with access to a searchable index of GitHub repositories.

When answering questions about code:
1. Start by calling search_code to find relevant code
2. If the initial results don't fully answer the question, search again with a different query
3. Use get_file_chunk to see more context around a result (e.g., the full class or surrounding code)
4. Use find_callers to understand how functions are used, not just defined
5. Only answer when you have enough evidence from the actual code

Always cite your sources: mention the file path and line numbers.
Be precise — if the code doesn't show what you're looking for, say so rather than guessing."""

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval = retrieval_service
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for agentic queries")
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def run(self, question: str, repo_filter: str | None = None) -> dict:
        """
        Run the agent loop synchronously.

        Returns:
            {
                "answer":     str,           # final LLM answer
                "tool_calls": list[dict],    # trace: [{tool, input, output}, ...]
                "iterations": int,           # how many reasoning steps it took
            }
        """
        # The conversation starts with just the user question.
        # Tool results will be appended as the loop progresses.
        messages = [{"role": "user", "content": question}]

        # If the user selected a specific repo, hint the agent
        if repo_filter:
            messages[0]["content"] += f"\n\n(Search in repo: {repo_filter})"

        tool_trace = []

        for iteration in range(self.MAX_ITERATIONS):
            # ── Ask Claude (with tools available) ─────────────────────────────
            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=self.SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # ── Did Claude give us a final answer? ────────────────────────────
            # stop_reason="end_turn" means Claude is done — no more tool calls.
            if response.stop_reason == "end_turn":
                answer = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        answer += block.text
                return {
                    "answer":     answer,
                    "tool_calls": tool_trace,
                    "iterations": iteration + 1,
                }

            # ── Claude wants to call tools ────────────────────────────────────
            # The response content may have multiple blocks:
            # - TextContent blocks (thinking out loud)
            # - ToolUseContent blocks (actual tool calls)

            # Append Claude's response to the conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name   = block.name
                tool_input  = block.input
                tool_use_id = block.id

                # ── Execute the tool ──────────────────────────────────────────
                try:
                    result = self._execute_tool(tool_name, tool_input)
                except Exception as e:
                    result = f"Tool error: {e}"

                # Record for the trace
                tool_trace.append({
                    "tool":   tool_name,
                    "input":  tool_input,
                    "output": result[:500] + "..." if len(result) > 500 else result,
                })

                # ── Build the tool_result message ─────────────────────────────
                # This goes back to Claude as a "user" turn.
                # Claude reads these results and decides what to do next.
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,
                    "content":     result,
                })

            # Add all tool results to the conversation
            messages.append({"role": "user", "content": tool_results})

        # Hit max iterations — return what we have
        return {
            "answer":     "I was unable to fully answer this question within the allowed reasoning steps.",
            "tool_calls": tool_trace,
            "iterations": self.MAX_ITERATIONS,
        }

    def stream(self, question: str, repo_filter: str | None = None) -> Iterator[dict]:
        """
        Stream agent progress as it happens.

        Yields dicts with type:
          {"type": "tool_call",  "tool": "search_code", "input": {...}}
          {"type": "tool_result","tool": "search_code", "output": "..."}
          {"type": "token",      "text": "According..."}
          {"type": "done",       "iterations": 3}
        """
        messages = [{"role": "user", "content": question}]
        if repo_filter:
            messages[0]["content"] += f"\n\n(Search in repo: {repo_filter})"

        for iteration in range(self.MAX_ITERATIONS):
            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=self.SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                # Stream the final answer token by token
                for block in response.content:
                    if hasattr(block, "text"):
                        # Yield word-by-word for a streaming feel
                        for word in block.text.split(" "):
                            yield {"type": "token", "text": word + " "}
                yield {"type": "done", "iterations": iteration + 1}
                return

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                yield {"type": "tool_call", "tool": block.name, "input": block.input}

                try:
                    result = self._execute_tool(block.name, block.input)
                except Exception as e:
                    result = f"Tool error: {e}"

                yield {"type": "tool_result", "tool": block.name, "output": result[:300]}

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

        yield {"type": "done", "iterations": self.MAX_ITERATIONS}

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
            top_k=args.get("top_k", 5),
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

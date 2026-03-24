// In development: http://localhost:8000
// In production:  set VITE_API_URL in Vercel environment variables
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchRepos() {
  const res = await fetch(`${BASE}/repos`);
  if (!res.ok) throw new Error("Failed to fetch repos");
  return res.json();
}

export async function ingestRepo(repoUrl, force = false) {
  const res = await fetch(`${BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_url: repoUrl, force }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Ingestion failed");
  return data;
}

export async function fetchGraph(slug) {
  const [owner, name] = slug.split("/");
  const res = await fetch(`${BASE}/repos/${owner}/${name}/graph`);
  if (!res.ok) throw new Error("Failed to fetch graph");
  return res.json();
}

export async function fetchMcpPrompt(name, args = {}) {
  const res = await fetch(
    `${BASE}/mcp-prompt?name=${encodeURIComponent(name)}&arguments=${encodeURIComponent(JSON.stringify(args))}`
  );
  if (!res.ok) throw new Error("Failed to fetch prompt");
  return res.json(); // { name, text }
}

export async function fetchMcpStatus() {
  const res = await fetch(`${BASE}/mcp-status`);
  if (!res.ok) throw new Error("Failed to fetch MCP status");
  return res.json();
}

export async function deleteRepo(slug) {
  const [owner, name] = slug.split("/");
  const res = await fetch(`${BASE}/repos/${owner}/${name}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete repo");
  return res.json();
}

/**
 * Stream a query response via SSE.
 *
 * The server sends two event types:
 *   event: meta   → JSON with { sources, query_type } (arrives before tokens)
 *   (default)     → token text, or "[DONE]" to signal completion
 *
 * This avoids the previous double-LLM-call pattern where we fired both
 * POST /query and GET /query/stream simultaneously. Now one connection does both.
 */
export function streamQuery({ question, repo, mode, onToken, onSources, onDone, onError }) {
  const params = new URLSearchParams({
    question,
    mode: mode || "hybrid",
    top_k: 6,
    ...(repo ? { repo } : {}),
  });

  const es = new EventSource(`${BASE}/query/stream?${params}`);

  // Named event: sources + query_type arrive in the first frame
  es.addEventListener("meta", (e) => {
    const { sources, query_type } = JSON.parse(e.data);
    onSources(sources || [], query_type || "technical");
  });

  // Default events: token text
  es.onmessage = (e) => {
    if (e.data === "[DONE]") {
      es.close();
      onDone();
      return;
    }
    const token = e.data.replace(/\\n/g, "\n");
    onToken(token);
  };

  es.onerror = () => {
    es.close();
    onError("Connection lost");
  };

  return () => es.close();
}

/**
 * Stream the agentic RAG loop via SSE.
 *
 * Unlike streamQuery (one retrieval → tokens), this endpoint shows the
 * agent's full ReAct reasoning loop in real time:
 *
 *   1. agent decides to search → event: tool_call
 *   2. result comes back       → event: tool_result
 *   3. agent decides to search again (or answer)
 *   4. when done, answer streams token-by-token (default events)
 *   5. event: done signals completion with iteration count
 *
 * Callbacks:
 *   onToolCall(tool, input)    — agent is calling a tool
 *   onToolResult(tool, output) — tool returned a result
 *   onToken(text)              — token of the final answer
 *   onDone(iterations)         — agent finished
 *   onError(msg)               — connection or server error
 */
export function streamAgentQuery({ question, repo, onToolCall, onToolResult, onToken, onDone, onError }) {
  const params = new URLSearchParams({
    question,
    ...(repo ? { repo } : {}),
  });

  const es = new EventSource(`${BASE}/agent/stream?${params}`);

  // Named event: agent is about to call a tool
  es.addEventListener("tool_call", (e) => {
    const { tool, input } = JSON.parse(e.data);
    onToolCall?.(tool, input);
  });

  // Named event: tool returned a result
  es.addEventListener("tool_result", (e) => {
    const { tool, output } = JSON.parse(e.data);
    onToolResult?.(tool, output);
  });

  // Named event: agent finished
  es.addEventListener("done", (e) => {
    const { iterations } = JSON.parse(e.data);
    onDone?.(iterations);
  });

  // Named event: server sent a clean error (API credits, missing key, etc.)
  // Using "agent_error" not "error" — the browser reserves the name "error"
  // for connection failures and it would conflict with onerror below.
  es.addEventListener("agent_error", (e) => {
    es.close();  // close before onerror can also fire
    try {
      const { message } = JSON.parse(e.data);
      onError?.(message);
    } catch {
      onError?.("Agent error — check server logs.");
    }
  });

  // Default events: token text (or [DONE] sentinel)
  es.onmessage = (e) => {
    if (e.data === "[DONE]") {
      es.close();
      return;
    }
    const token = e.data.replace(/\\n/g, "\n");
    onToken?.(token);
  };

  es.onerror = () => {
    es.close();
    onError?.("Could not connect to the agent. Is the backend running?");
  };

  return () => es.close();
}

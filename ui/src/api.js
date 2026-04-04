// In development: http://localhost:8000
// In production:  set VITE_API_URL in Vercel environment variables
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchAgentModels() {
  const res = await fetch(`${BASE}/agent/models`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.models || [];
}

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

export async function fetchTour(slug) {
  const [owner, name] = slug.split("/");
  const res = await fetch(`${BASE}/repos/${owner}/${name}/tour`);
  if (!res.ok) throw new Error("Failed to generate tour");
  return res.json(); // { summary, entry_point, concepts: [...] }
}

export async function fetchDiagram(slug, type = "architecture") {
  const [owner, name] = slug.split("/");
  const res = await fetch(`${BASE}/repos/${owner}/${name}/diagram?type=${type}`);
  if (!res.ok) throw new Error("Failed to generate diagram");
  return res.json(); // { diagram: "<mermaid syntax>", type } or { error: "..." }
}

/**
 * Stream codebase tour generation with live progress events.
 *
 * Replaces the blank spinner with real progress stages:
 *   loading → analysing → generating → parsing → done
 *
 * onProgress({ stage, progress, message }) — called for each intermediate event
 * onDone(tourData)                          — called with the full tour on completion
 * onError(msg)                              — called on failure
 *
 * Returns a cancel() function.
 */
export function streamTour(slug, { onProgress, onDone, onError, force = false }) {
  const [owner, name] = slug.split("/");
  const controller = new AbortController();
  const url = force
    ? `${BASE}/repos/${owner}/${name}/tour/stream?force=true`
    : `${BASE}/repos/${owner}/${name}/tour/stream`;

  fetch(url, { signal: controller.signal })
    .then(async (res) => {
      if (!res.ok) { onError?.(`Server error ${res.status}`); return; }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
          if (!part.trim()) continue;
          const line = part.split("\n").find(l => l.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));

          if (event.stage === "done") {
            const { stage, progress, ...tourData } = event;
            onDone?.(tourData);
          } else if (event.stage === "error") {
            onError?.(event.error || "Failed to generate tour");
          } else {
            onProgress?.(event);
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") onError?.(err.message || "Connection lost");
    });

  return () => controller.abort();
}

/**
 * Stream diagram generation with live progress events.
 *
 * Progress stages: loading → building → enriching → done
 *
 * onProgress({ stage, progress, message })
 * onDone({ diagram, type })
 * onError(msg)
 *
 * Returns a cancel() function.
 */
export function streamDiagram(slug, type = "architecture", { onProgress, onDone, onError, force = false }) {
  const [owner, name] = slug.split("/");
  const controller = new AbortController();
  const url = `${BASE}/repos/${owner}/${name}/diagram/stream?type=${type}${force ? "&force=true" : ""}`;

  fetch(url, { signal: controller.signal })
    .then(async (res) => {
      if (!res.ok) { onError?.(`Server error ${res.status}`); return; }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
          if (!part.trim()) continue;
          const line = part.split("\n").find(l => l.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));

          if (event.stage === "done") {
            onDone?.({ diagram: event.diagram, type: event.type });
          } else if (event.stage === "error") {
            onError?.(event.error || "Failed to generate diagram");
          } else {
            onProgress?.(event);
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") onError?.(err.message || "Connection lost");
    });

  return () => controller.abort();
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
 * Low-level POST SSE helper.
 *
 * EventSource only supports GET, so we can't send a request body (e.g. history).
 * Instead we use fetch() with a ReadableStream response and parse the SSE format
 * manually. The returned cancel() function aborts the in-flight request.
 *
 * SSE wire format (per spec):
 *   event: <type>\ndata: <json>\n\n   ← named event
 *   data: <text>\n\n                  ← default event
 */
async function postSSE(path, body, handlers) {
  const controller = new AbortController();

  try {
    const res = await fetch(`${BASE}${path}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
      signal:  controller.signal,
    });

    if (!res.ok) {
      handlers.onError?.(`Server error ${res.status}`);
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by blank lines (\n\n)
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep any incomplete trailing chunk

      for (const part of parts) {
        if (!part.trim()) continue;
        let eventType = "message";
        let data      = "";

        for (const line of part.split("\n")) {
          if (line.startsWith("event: "))      eventType = line.slice(7).trim();
          else if (line.startsWith("data: "))  data      = line.slice(6);
        }

        if (data) handlers[eventType]?.(data);
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") handlers.onError?.(err.message || "Connection lost");
  }

  return () => controller.abort();
}

/**
 * Stream a query response via SSE (POST so we can send conversation history).
 *
 * The server sends two event types:
 *   event: meta   → JSON with { sources, query_type } (arrives before tokens)
 *   (default)     → token text, or "[DONE]" to signal completion
 */
export function streamQuery({ question, repo, mode, history, onToken, onSources, onGrade, onDone, onError }) {
  const controller = new AbortController();

  fetch(`${BASE}/query/stream`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      question,
      mode:    mode || "hybrid",
      top_k:   6,
      repo:    repo || null,
      history: history || [],
    }),
    signal:  controller.signal,
  }).then(async (res) => {
    if (!res.ok) { onError(`Server error ${res.status}`); return; }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.trim()) continue;
        let eventType = "message";
        let data      = "";
        for (const line of part.split("\n")) {
          if (line.startsWith("event: "))     eventType = line.slice(7).trim();
          else if (line.startsWith("data: ")) data      = line.slice(6);
        }
        if (!data) continue;

        if (eventType === "meta") {
          const { sources, query_type, pipeline, model } = JSON.parse(data);
          onSources(sources || [], query_type || "technical", pipeline || {}, model || "");
        } else if (eventType === "grade") {
          onGrade?.(JSON.parse(data));
        } else {
          // default event: token or [DONE]
          if (data === "[DONE]") { onDone(); return; }
          onToken(data.replace(/\\n/g, "\n"));
        }
      }
    }
  }).catch((err) => {
    if (err.name !== "AbortError") onError(err.message || "Connection lost");
  });

  return () => controller.abort();
}

/**
 * Stream the agentic RAG loop via SSE (POST so we can send conversation history).
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
export function streamAgentQuery({ question, repo, model_id, history, onThought, onToolCall, onToolResult, onToken, onSources, onDone, onError }) {
  const controller = new AbortController();

  fetch(`${BASE}/agent/stream`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question, repo: repo || null, model_id: model_id || null, history: history || [] }),
    signal:  controller.signal,
  }).then(async (res) => {
    if (!res.ok) { onError?.(`Server error ${res.status}`); return; }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.trim()) continue;
        let eventType = "message";
        let data      = "";
        for (const line of part.split("\n")) {
          if (line.startsWith("event: "))     eventType = line.slice(7).trim();
          else if (line.startsWith("data: ")) data      = line.slice(6);
        }
        if (!data) continue;

        if (eventType === "thought") {
          const { text } = JSON.parse(data);
          onThought?.(text);
        } else if (eventType === "tool_call") {
          const { tool, input } = JSON.parse(data);
          onToolCall?.(tool, input);
        } else if (eventType === "tool_result") {
          const { tool, output } = JSON.parse(data);
          onToolResult?.(tool, output);
        } else if (eventType === "sources") {
          const { sources } = JSON.parse(data);
          onSources?.(sources || []);
        } else if (eventType === "done") {
          const { iterations, model } = JSON.parse(data);
          onDone?.(iterations, model);
        } else if (eventType === "agent_error") {
          const { message } = JSON.parse(data);
          onError?.(message);
          return;
        } else {
          // default: token or [DONE]
          if (data === "[DONE]") return;
          onToken?.(data.replace(/\\n/g, "\n"));
        }
      }
    }
  }).catch((err) => {
    if (err.name !== "AbortError") onError?.("Could not connect to the agent. Is the backend running?");
  });

  return () => controller.abort();
}

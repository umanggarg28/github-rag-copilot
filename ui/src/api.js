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

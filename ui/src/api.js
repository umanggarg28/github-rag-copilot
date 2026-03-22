const BASE = "http://localhost:8000";

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
 * Calls onToken(token) for each chunk, onSources(sources) when done,
 * and onDone(queryType) at the end.
 */
export function streamQuery({ question, repo, mode, onToken, onSources, onDone, onError }) {
  const params = new URLSearchParams({
    question,
    mode: mode || "hybrid",
    top_k: 6,
    ...(repo ? { repo } : {}),
  });

  // First fetch sources via POST /query (non-streaming) to get structured data,
  // then stream the answer via GET /query/stream for the text tokens.
  // We run both in parallel — sources arrive slightly later but the stream starts immediately.

  let queryType = "technical";

  // Kick off the source fetch
  fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, repo: repo || null, mode: mode || "hybrid", top_k: 6 }),
  })
    .then((r) => r.json())
    .then((data) => {
      onSources(data.sources || [], data.query_type || "technical");
    })
    .catch(() => onSources([], "technical"));

  // Stream the answer tokens
  const es = new EventSource(`${BASE}/query/stream?${params}`);

  es.onmessage = (e) => {
    if (e.data === "[DONE]") {
      es.close();
      onDone(queryType);
      return;
    }
    // Unescape newlines that were escaped server-side
    const token = e.data.replace(/\\n/g, "\n");
    onToken(token);
  };

  es.onerror = () => {
    es.close();
    onError("Connection lost");
  };

  return () => es.close(); // return cleanup fn
}

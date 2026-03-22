import { useState } from "react";
import { ingestRepo, deleteRepo } from "../api";

export default function Sidebar({ repos, activeRepo, onSelectRepo, onReposChange, mode, onModeChange }) {
  const [url, setUrl]         = useState("");
  const [status, setStatus]   = useState(null); // {type, text}
  const [loading, setLoading] = useState(false);

  async function handleIngest(e) {
    e.preventDefault();
    if (!url.trim()) return;
    setLoading(true);
    setStatus({ type: "info", text: "Ingesting…" });
    try {
      const result = await ingestRepo(url.trim());
      setStatus({ type: "success", text: `✓ ${result.chunks_stored} chunks indexed` });
      setUrl("");
      onReposChange();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(e, slug) {
    e.stopPropagation();
    if (!confirm(`Delete all chunks for ${slug}?`)) return;
    try {
      await deleteRepo(slug);
      if (activeRepo === slug) onSelectRepo(null);
      onReposChange();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    }
  }

  return (
    <div className="sidebar">
      <h1><span>⚡</span> GitHub RAG</h1>

      {/* ── Ingest ── */}
      <div>
        <div className="section-label">Add Repository</div>
        <form className="ingest-form" onSubmit={handleIngest}>
          <input
            type="text"
            placeholder="github.com/owner/repo"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={loading}
          />
          <button className="btn" type="submit" disabled={loading || !url.trim()}>
            {loading ? "Indexing…" : "Index Repo"}
          </button>
        </form>
        {status && (
          <div className={`status-bar ${status.type}`} style={{ marginTop: 8 }}>
            {status.text}
          </div>
        )}
      </div>

      {/* ── Search mode ── */}
      <div>
        <div className="section-label">Search Mode</div>
        <div className="mode-pills">
          {["hybrid", "semantic", "keyword"].map((m) => (
            <button
              key={m}
              className={`pill ${mode === m ? "active" : ""}`}
              onClick={() => onModeChange(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* ── Repos ── */}
      <div style={{ flex: 1 }}>
        <div className="section-label">Indexed Repos ({repos.length})</div>
        {repos.length === 0 ? (
          <p style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.5 }}>
            No repos indexed yet. Add one above.
          </p>
        ) : (
          <div className="repo-list">
            <div
              className={`repo-item ${activeRepo === null ? "active" : ""}`}
              onClick={() => onSelectRepo(null)}
            >
              <span className="repo-slug" style={{ color: "var(--muted)" }}>All repos</span>
            </div>
            {repos.map((r) => (
              <div
                key={r.slug}
                className={`repo-item ${activeRepo === r.slug ? "active" : ""}`}
                onClick={() => onSelectRepo(r.slug)}
              >
                <span className="repo-slug">{r.slug}</span>
                <span className="repo-count">{r.chunks}</span>
                <button
                  className="repo-delete"
                  onClick={(e) => handleDelete(e, r.slug)}
                  title="Remove from index"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

import { useState, useEffect } from "react";
import { ingestRepo, deleteRepo, fetchMcpStatus } from "../api";

export default function Sidebar({ repos, activeRepo, onSelectRepo, onReposChange, mode, onModeChange, agentMode, onAgentModeChange, isOpen, onClose }) {
  const [url, setUrl]         = useState("");
  const [status, setStatus]   = useState(null); // {type, text}
  const [loading, setLoading] = useState(false);
  const [mcpInfo, setMcpInfo] = useState(null); // MCP server status
  const [mcpOpen, setMcpOpen] = useState(false); // expand/collapse panel
  const [confirming, setConfirming] = useState(null); // slug being confirmed for delete

  // Load MCP status once on mount
  useEffect(() => {
    fetchMcpStatus().then(setMcpInfo).catch(() => setMcpInfo({ connected: false }));
  }, []);

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
      // Auto-select the newly indexed repo
      if (result.repo) onSelectRepo(result.repo);
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(e, slug) {
    e.stopPropagation();
    try {
      await deleteRepo(slug);
      if (activeRepo === slug) onSelectRepo(null);
      onReposChange();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    }
  }

  const SEARCH_MODE_TITLES = {
    hybrid: "Combines text matching + semantic similarity (recommended)",
    semantic: "Finds conceptually similar code",
    keyword: "Exact identifier matching",
  };

  return (
    <div className={`sidebar ${isOpen ? "open" : ""}`}>
      {/* ── Brand ── */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">⚡</div>
        <div>
          <div className="sidebar-brand-name">GitHub RAG</div>
          <div className="sidebar-brand-tag">Code Copilot</div>
        </div>
      </div>

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

      {/* ── Query mode (RAG vs Agent) ── */}
      <div>
        <div className="section-label">Query Mode</div>
        {/* Agent mode toggle — switches between plain RAG and agentic ReAct loop */}
        <div className="mode-pills">
          <button
            className={`pill ${!agentMode ? "active" : ""}`}
            onClick={() => onAgentModeChange(false)}
            title="Single retrieval, fast answer"
            aria-pressed={!agentMode}
          >
            RAG
          </button>
          <button
            className={`pill ${agentMode ? "active" : ""}`}
            onClick={() => onAgentModeChange(true)}
            title="Multi-step reasoning, more thorough"
            aria-pressed={agentMode}
          >
            Agent ✦
          </button>
        </div>
      </div>

      {/* ── Search mode (only visible in RAG mode) ── */}
      {!agentMode && (
        <div>
          <div className="section-label">Search Mode</div>
          <div className="mode-pills">
            {["hybrid", "semantic", "keyword"].map((m) => (
              <button
                key={m}
                className={`pill ${mode === m ? "active" : ""}`}
                onClick={() => onModeChange(m)}
                aria-pressed={mode === m}
                title={SEARCH_MODE_TITLES[m]}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      )}

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
                {confirming === r.slug ? (
                  <span style={{ display: "flex", gap: 4, alignItems: "center", flexShrink: 0 }}>
                    <button
                      className="repo-delete"
                      style={{ color: "var(--red)", fontWeight: 600, fontSize: 11 }}
                      onClick={(e) => { e.stopPropagation(); handleDelete(e, r.slug); setConfirming(null); }}
                    >Delete</button>
                    <button
                      className="repo-delete"
                      onClick={(e) => { e.stopPropagation(); setConfirming(null); }}
                    >Cancel</button>
                  </span>
                ) : (
                  <button
                    className="repo-delete"
                    onClick={(e) => { e.stopPropagation(); setConfirming(r.slug); }}
                    title="Remove from index"
                    aria-label={`Remove ${r.slug} from index`}
                  >×</button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── MCP Server Status — infrastructure panel, kept below primary UX ── */}
      <div className="mcp-panel">
        <button
          className="mcp-panel-header"
          onClick={() => setMcpOpen(o => !o)}
          aria-expanded={mcpOpen}
          aria-controls="mcp-panel-body"
        >
          <span className={`mcp-dot ${mcpInfo?.connected ? "connected" : "disconnected"}`} />
          <span className="mcp-panel-title">MCP Server</span>
          {mcpInfo?.connected && (
            <span className="mcp-counts">
              {mcpInfo.tools.length}T · {mcpInfo.resources.length}R · {mcpInfo.prompts.length}P
            </span>
          )}
          <span className="mcp-chevron">{mcpOpen ? "▴" : "▾"}</span>
        </button>

        {mcpOpen && mcpInfo && (
          <div id="mcp-panel-body" className="mcp-panel-body">
            {!mcpInfo.connected ? (
              <p className="mcp-error">Not connected — is the backend running?</p>
            ) : (
              <>
                {mcpInfo.tools.length > 0 && (
                  <div className="mcp-section">
                    <div className="mcp-section-label">Tools</div>
                    {mcpInfo.tools.map(t => (
                      <div key={t.name} className="mcp-item">
                        <span className="mcp-item-name">{t.name}</span>
                      </div>
                    ))}
                  </div>
                )}
                {mcpInfo.resources.length > 0 && (
                  <div className="mcp-section">
                    <div className="mcp-section-label">Resources</div>
                    {mcpInfo.resources.map(r => (
                      <div key={r.uri} className="mcp-item">
                        <span className="mcp-item-name mcp-uri">{r.uri}</span>
                      </div>
                    ))}
                  </div>
                )}
                {mcpInfo.prompts.length > 0 && (
                  <div className="mcp-section">
                    <div className="mcp-section-label">Prompts</div>
                    {mcpInfo.prompts.map(p => (
                      <div key={p.name} className="mcp-item">
                        <span className="mcp-item-name">/{p.name}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

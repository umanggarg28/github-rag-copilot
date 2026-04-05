import { useState, useEffect, useRef } from "react";
import { deleteRepo, fetchMcpStatus, ingestRepo } from "../api";

function ContextualTip() {
  const [open, setOpen] = useState(false);
  return (
    <div className="ctip">
      <button className="ctip-trigger" onClick={() => setOpen(o => !o)}>
        <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.5, flexShrink: 0 }}>
          <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-.5 4.5h1v1.5h-1zm0 3h1v4h-1z"/>
        </svg>
        <span>Improve search quality</span>
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginLeft: "auto", opacity: 0.4, transition: "transform 0.2s", transform: open ? "rotate(180deg)" : "none" }}>
          <path d="m4 6 4 4 4-4"/>
        </svg>
      </button>
      {open && (
        <p className="ctip-body">
          Hit <span className="quality-tip-key">⟳</span> on any repo to re-index with <strong>contextual retrieval</strong> — the AI prepends a description to each key chunk before embedding. Searches, diagrams, and the semantic map all improve.
        </p>
      )}
    </div>
  );
}

function SessionItem({ sess, onLoad, onDelete, onRename, isActive }) {
  const [confirming, setConfirming] = useState(false);
  const [editing, setEditing]       = useState(false);
  const [editVal, setEditVal]       = useState(sess.title);
  const inputRef = useRef(null);

  // Focus the input when entering edit mode
  useEffect(() => {
    if (editing && inputRef.current) inputRef.current.focus();
  }, [editing]);

  function startEdit(e) {
    e.stopPropagation();
    setEditVal(sess.title);
    setEditing(true);
  }

  function commitEdit() {
    const trimmed = editVal.trim();
    if (trimmed && trimmed !== sess.title) onRename(sess.id, trimmed);
    setEditing(false);
  }

  function handleEditKey(e) {
    if (e.key === "Enter") { e.preventDefault(); commitEdit(); }
    if (e.key === "Escape") { setEditing(false); }
  }

  return (
    <div className={`session-item${isActive ? " active" : ""}`}>
      {editing ? (
        <input
          ref={inputRef}
          className="session-title-input"
          value={editVal}
          onChange={e => setEditVal(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={handleEditKey}
          onClick={e => e.stopPropagation()}
          maxLength={80}
          aria-label="Edit session title"
        />
      ) : (
        <button className="session-btn" onClick={() => onLoad(sess)} onDoubleClick={startEdit} title={`${sess.title}\n(double-click to rename)`}>
          <span className="session-title">{sess.title}</span>
          <span style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
            {sess.agentMode && <span className="session-mode-badge" title="Agent mode session">✦</span>}
            <span className="session-time">{timeAgo(sess.timestamp)}</span>
          </span>
        </button>
      )}
      {confirming ? (
        <span className="session-confirm">
          <button className="session-confirm-yes" onClick={() => { onDelete(sess.id); setConfirming(false); }}>Remove</button>
          <button className="session-confirm-no"  onClick={() => setConfirming(false)}>Cancel</button>
        </span>
      ) : (
        <button className="session-delete" onClick={() => setConfirming(true)} title="Delete session" aria-label="Delete session">×</button>
      )}
    </div>
  );
}

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1)  return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// Staleness thresholds: warn if index is older than 3 days, stale if > 7 days.
function stalenessLevel(isoTimestamp) {
  if (!isoTimestamp) return null;
  const days = (Date.now() - new Date(isoTimestamp).getTime()) / (1000 * 60 * 60 * 24);
  if (days < 3) return null;       // fresh — no indicator
  if (days < 7) return "warn";     // getting old
  return "stale";                  // definitely stale
}

export default function Sidebar({ repos, reposLoading, activeRepo, onSelectRepo, onReposChange, mode, onModeChange, agentMode, onAgentModeChange, sessions, currentSessionId, onLoadSession, onDeleteSession, onRenameSession, isOpen, onClose, collapsed, onToggleCollapse }) {
  const [url, setUrl]                   = useState("");
  const [status, setStatus]             = useState(null); // {type, text}
  const [loading, setLoading]           = useState(false);
  const [mcpInfo, setMcpInfo]           = useState(null); // MCP server status
  const [mcpOpen, setMcpOpen]           = useState(false); // expand/collapse panel
  const [confirming, setConfirming]     = useState(null); // slug being confirmed for delete
  const [ingestProgress, setIngestProgress] = useState([]); // [{step, detail, done}]
  const [isIngesting, setIsIngesting]   = useState(false);
  const [reindexing, setReindexing]     = useState(null);  // slug currently re-indexing
  const [reindexDone, setReindexDone]   = useState({});    // slug → bool (just finished)
  const [sessionSearch, setSessionSearch] = useState("");  // filter text for sessions list

  // Load MCP status once on mount
  useEffect(() => {
    fetchMcpStatus().then(setMcpInfo).catch(() => setMcpInfo({ connected: false }));
  }, []);

  function handleIngest(e) {
    e.preventDefault();
    if (!url.trim() || isIngesting) return;

    setIsIngesting(true);
    setIngestProgress([]);
    setStatus(null);

    // Connect to the SSE stream — the server pushes step events as it progresses
    // through fetching → filtering → chunking → embedding → storing → done.
    // EventSource handles reconnection automatically on network blips, so we
    // explicitly close it once we receive "done" or "error" to prevent that.
    const streamUrl = `http://localhost:8000/ingest/stream?repo=${encodeURIComponent(url.trim())}`;
    const es = new EventSource(streamUrl);

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);

      setIngestProgress(prev => {
        // Mark all previous steps as completed, then append the new active step.
        const updated = prev.map(s => ({ ...s, done: true }));
        return [...updated, { step: event.step, detail: event.detail, done: false }];
      });

      if (event.step === "done" || event.step === "error") {
        // The final step was appended as active (done: false). Mark it done now —
        // no subsequent event will arrive to flip it, so we do it explicitly.
        setIngestProgress(prev => prev.map(s => ({ ...s, done: true })));
        es.close();
        setIsIngesting(false);
        if (event.step === "done") {
          // Extract owner/repo slug from the URL the user typed.
          // Handles both "github.com/owner/repo" and "https://github.com/owner/repo".
          const match = url.match(/github\.com\/([^/]+\/[^/]+)/);
          if (match && onSelectRepo) onSelectRepo(match[1]);
          setUrl("");
          onReposChange();
          // Collapse the progress list after 3s so the card returns to normal size
          setTimeout(() => setIngestProgress([]), 3000);
        }
      }
    };

    es.onerror = () => {
      es.close();
      setIsIngesting(false);
      setIngestProgress(prev => [
        ...prev,
        { step: "error", detail: "Connection failed — is the backend running?", done: false },
      ]);
    };
  }

  async function handleDelete(e, slug) {
    e.stopPropagation();
    try {
      await deleteRepo(slug);
      if (activeRepo === slug) onSelectRepo("all");
      onReposChange();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    }
  }

  async function handleReindex(e, slug) {
    e.stopPropagation();
    if (reindexing) return;
    setReindexing(slug);
    setReindexDone(prev => ({ ...prev, [slug]: false }));
    try {
      // force=true deletes and re-ingests from scratch so the index is fully fresh
      await ingestRepo(`https://github.com/${slug}`, true);
      setReindexDone(prev => ({ ...prev, [slug]: true }));
      onReposChange();
      // Clear the "Re-indexed" badge after 3s
      setTimeout(() => setReindexDone(prev => { const n = {...prev}; delete n[slug]; return n; }), 3000);
    } catch (err) {
      setStatus({ type: "error", text: `Re-index failed: ${err.message}` });
    } finally {
      setReindexing(null);
    }
  }

  const SEARCH_MODE_TITLES = {
    hybrid: "Combines text matching + semantic similarity (recommended)",
    semantic: "Finds conceptually similar code",
    keyword: "Exact identifier matching",
  };

  // ── Collapsed rail ────────────────────────────────────────────────────────
  // When collapsed, show a slim 52px icon strip with key counts + expand button.
  // Same pattern as rag-research-copilot: two separate JSX trees, no CSS trickery.
  if (collapsed) {
    return (
      <div className="sidebar sidebar-collapsed">
        {/* Brand icon */}
        <div className="sidebar-brand-icon sidebar-collapsed-brand" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M5.5 5L2 9l3.5 4" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" strokeOpacity="0.95"/>
            <path d="M12.5 5L16 9l-3.5 4" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" strokeOpacity="0.95"/>
            <circle cx="9" cy="9" r="1.2" fill="white" fillOpacity="0.7"/>
          </svg>
        </div>

        {/* Repo count */}
        {repos.length > 0 && (
          <div className="sidebar-collapsed-item" title={`${repos.length} repo${repos.length !== 1 ? 's' : ''} indexed`}>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.5 }}>
              <path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Z"/>
            </svg>
            <span className="sidebar-collapsed-badge">{repos.length}</span>
          </div>
        )}

        {/* Session count */}
        {sessions && sessions.length > 0 && (
          <div className="sidebar-collapsed-item" title={`${sessions.length} saved chat${sessions.length !== 1 ? 's' : ''}`}>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.5 }}>
              <path d="M1 2.75C1 1.784 1.784 1 2.75 1h10.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0 1 13.25 12H9.06l-2.573 2.573A1.457 1.457 0 0 1 4 13.543V12H2.75A1.75 1.75 0 0 1 1 10.25Z"/>
            </svg>
            <span className="sidebar-collapsed-badge">{sessions.length}</span>
          </div>
        )}

        {/* Expand button — pinned to bottom */}
        <button
          className="sidebar-collapsed-expand"
          onClick={onToggleCollapse}
          title="Expand sidebar"
          aria-label="Expand sidebar"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m6 4 4 4-4 4"/>
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className={`sidebar ${isOpen ? "open" : ""}`}>
      {/* ── Scrollable top section ── */}
      <div className="sidebar-scroll">

      {/* ── Brand ── */}
      <div className="sidebar-brand">
        {/* Compass — hand-drawn marker style, like a child sketched it on a treasure map */}
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
          {/* The compass housing — thick circle drawn with a round marker */}
          <circle cx="12" cy="12" r="9" stroke="var(--accent)" strokeWidth="1.8" strokeOpacity="0.35"/>
          {/* N needle — two angled strokes meeting at the top, like a fat marker arrow */}
          <path d="M9.5 13.5 L12 4 L14.5 13.5"
                stroke="var(--accent)" strokeWidth="2.6"
                strokeLinecap="round" strokeLinejoin="round"/>
          {/* S tail — shorter, dimmer, same drawn feel */}
          <path d="M10.5 13.5 L12 20 L13.5 13.5"
                stroke="var(--accent)" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round"
                strokeOpacity="0.3"/>
          {/* Center pivot — chunky dot */}
          <circle cx="12" cy="12" r="2.2" fill="var(--accent)"/>
        </svg>
        <div style={{ flex: 1 }}>
          <div className="sidebar-brand-name">Cartographer</div>
        </div>
        <button
          className="sidebar-collapse-btn"
          onClick={onToggleCollapse}
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m10 4-4 4 4 4"/>
          </svg>
        </button>
      </div>

      {/* ── Ingest ── */}
      <div className="sidebar-section">
        <div className="section-label">Add Repository</div>
        <div className="ingest-card">
        <form className="ingest-form" onSubmit={handleIngest}>
          <input
            type="text"
            placeholder="github.com/karpathy/nanoGPT"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={isIngesting}
          />
          <button className="btn" type="submit" disabled={isIngesting || !url.trim()}>
            {isIngesting ? "Indexing…" : "Index Repo"}
          </button>
        </form>
        {/* Curated repos — quick-start for new users */}
        <div style={{ marginTop: 10 }}>
          <div style={{ marginBottom: 5, color: "var(--faint)", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>Try these</div>
          <div className="try-repo-chips">
            {[
              { slug: "karpathy/nanoGPT", label: "GPT from scratch" },
              { slug: "karpathy/micrograd", label: "autograd engine" },
              { slug: "langchain-ai/langchain", label: "LLM framework" },
            ].map(({ slug, label }) => (
              <button
                key={slug}
                className="try-repo-chip"
                onClick={() => setUrl(`github.com/${slug}`)}
                title={label}
              >
                <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.6, flexShrink: 0 }}>
                  <path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Z"/>
                </svg>
                {slug.split("/")[1]}
              </button>
            ))}
          </div>
        </div>
        {status && (
          <div className={`status-bar ${status.type}`} style={{ marginTop: 8 }}>
            {status.text}
          </div>
        )}
        {ingestProgress.length > 0 && (
          <div className="ingest-progress">
            {ingestProgress.map((p, i) => (
              <div
                key={i}
                className={`ingest-step ${p.done ? "done" : "active"} ${p.step === "error" ? "error" : ""}`}
              >
                <span className="ingest-step-icon">
                  {p.step === "error" ? (
                    /* X circle */
                    <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                      <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.03 10.97L10.03 12 8 9.97 5.97 12l-1-1.03L7 8.97 5 6.97l1-1 2 2 2-2 1 1-2 2z"/>
                    </svg>
                  ) : p.done ? (
                    /* Check circle */
                    <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                      <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.78 6.22-4.5 4.5a.75.75 0 0 1-1.06 0l-2-2a.75.75 0 1 1 1.06-1.06l1.47 1.47 3.97-3.97a.75.75 0 1 1 1.06 1.06z"/>
                    </svg>
                  ) : (
                    /* Spinner dots — three dots for "in progress" */
                    <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                      <circle cx="2" cy="8" r="1.5"/><circle cx="8" cy="8" r="1.5"/><circle cx="14" cy="8" r="1.5"/>
                    </svg>
                  )}
                </span>
                <span className="ingest-step-detail">{p.detail}</span>
              </div>
            ))}
          </div>
        )}
        </div>{/* end ingest-card */}
      </div>

      {/* ── Query mode (RAG vs Agent) ── */}
      <div className="sidebar-section">
        <div className="section-label">Query Mode</div>
        <div className="mode-pills">
          <button
            className={`pill ${!agentMode ? "active" : ""}`}
            onClick={() => onAgentModeChange(false)}
            aria-pressed={!agentMode}
          >RAG</button>
          <button
            className={`pill ${agentMode ? "active" : ""}`}
            onClick={() => onAgentModeChange(true)}
            aria-pressed={agentMode}
          >Agent <span style={{ fontSize: 8, verticalAlign: "middle", color: "var(--accent-soft)", marginLeft: 2 }}>●</span></button>
        </div>
        <p className="mode-description">
          {agentMode
            ? "Searches → reads → searches again. Slower but thorough."
            : "Retrieves code once, streams an answer. Fast."}
        </p>
      </div>

      {/* ── Search mode (only visible in RAG mode) ── */}
      {!agentMode && (
        <div className="sidebar-section">
          <div className="section-label">Search Mode</div>
          <div className="mode-pills">
            {["hybrid", "semantic", "keyword"].map((m) => (
              <button
                key={m}
                className={`pill ${mode === m ? "active" : ""}`}
                onClick={() => onModeChange(m)}
                aria-pressed={mode === m}
              >{m}</button>
            ))}
          </div>
          <p className="mode-description">
            {mode === "hybrid"   && "Text + semantic combined. Best for most questions."}
            {mode === "semantic" && "Finds conceptually similar code, even without exact terms."}
            {mode === "keyword"  && "Exact identifier matching. Best for function or class names."}
          </p>
        </div>
      )}

      {/* ── Repos ── */}
      <div className="sidebar-section">
        <div className="section-label">Indexed Repos ({reposLoading ? "…" : repos.length})</div>
        {reposLoading ? (
          // Skeleton while the first fetch is in flight — backend can take a moment on cold start
          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 4 }}>
            {[1, 2].map(i => (
              <div key={i} style={{
                height: 34, borderRadius: "var(--radius-sm)",
                background: "var(--surface-3)",
                animation: "pulse 1.4s ease-in-out infinite",
                animationDelay: `${i * 0.15}s`,
              }} />
            ))}
          </div>
        ) : repos.length === 0 ? (
          <p style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.5 }}>
            No repos indexed yet. Add one above.
          </p>
        ) : (
          <div className="repo-list">
            <div
              className={`repo-item ${activeRepo === "all" ? "active" : ""}`}
              onClick={() => onSelectRepo(activeRepo === "all" ? null : "all")}
            >
              <span className="repo-slug">All repos</span>
            </div>
            {repos.map((r) => {
              const staleness = stalenessLevel(r.indexed_at);
              const isReindexingThis = reindexing === r.slug;
              const justDone = reindexDone[r.slug];
              return (
                <div
                  key={r.slug}
                  className={`repo-item ${activeRepo === r.slug ? "active" : ""}`}
                  onClick={() => onSelectRepo(activeRepo === r.slug ? null : r.slug)}
                >
                  <div className="repo-item-main">
                    <span className="repo-slug">{r.slug}</span>
                    <div className="repo-item-meta">
                      {/* Staleness indicator — shown when index is > 3 days old */}
                      {staleness && !justDone && (
                        <span className={`repo-staleness repo-staleness--${staleness}`} title={`Indexed ${timeAgo(r.indexed_at)}`}>
                          {staleness === "warn" ? "~old" : "stale"}
                        </span>
                      )}
                      {justDone && (
                        <span className="repo-staleness repo-staleness--fresh">updated</span>
                      )}
                      {r.contextual_at && (
                        <span className="repo-contextual" title={`Contextual retrieval applied — re-indexed ${timeAgo(r.contextual_at)}`}>✦</span>
                      )}
                      <span className="repo-count">{r.chunks}</span>
                    </div>
                  </div>
                  <div className="repo-item-actions">
                    {/* Re-index button — one click re-ingests from scratch */}
                    <button
                      className={`repo-reindex${isReindexingThis ? " spinning" : ""}`}
                      onClick={(e) => handleReindex(e, r.slug)}
                      disabled={!!reindexing}
                      title={isReindexingThis ? "Re-indexing…" : "Re-index with contextual retrieval — adds AI-generated descriptions to key chunks before embedding, improving search precision"}
                      aria-label={`Re-index ${r.slug}`}
                    >
                      ⟳
                    </button>
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
                </div>
              );
            })}
          </div>
        )}
        {repos.length > 0 && <ContextualTip />}
      </div>

      {/* ── Recent chats ── */}
      {sessions && sessions.length > 0 && (
        <div className="sidebar-section">
          <div className="section-label">Recent chats</div>
          {/* Session search — visible when there are enough sessions to warrant filtering */}
          {sessions.length >= 3 && (
            <input
              className="session-search"
              type="text"
              placeholder="Search chats…"
              value={sessionSearch}
              onChange={e => setSessionSearch(e.target.value)}
              aria-label="Search sessions"
            />
          )}
          <div className="session-list">
            {sessions
              .filter(sess => !sessionSearch || sess.title.toLowerCase().includes(sessionSearch.toLowerCase()))
              .map(sess => (
                <SessionItem
                  key={sess.id}
                  sess={sess}
                  isActive={sess.id === currentSessionId}
                  onLoad={onLoadSession}
                  onDelete={onDeleteSession}
                  onRename={onRenameSession}
                />
              ))
            }
            {sessionSearch && sessions.filter(s => s.title.toLowerCase().includes(sessionSearch.toLowerCase())).length === 0 && (
              <div style={{ fontSize: 12, color: "var(--muted)", padding: "6px 0" }}>No chats match "{sessionSearch}"</div>
            )}
          </div>
        </div>
      )}

      </div>{/* end sidebar-scroll */}

      {/* ── MCP Server Status — pinned at bottom, does not scroll with sidebar ── */}
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
          <svg
            className="mcp-chevron"
            width="10" height="10" viewBox="0 0 16 16"
            fill="none" stroke="currentColor" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round"
            style={{ transform: mcpOpen ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}
            aria-hidden="true"
          >
            <path d="m4 6 4 4 4-4"/>
          </svg>
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

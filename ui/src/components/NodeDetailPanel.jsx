/**
 * NodeDetailPanel.jsx — Bottom-tray detail panel for diagram nodes and edges.
 *
 * Layout: sits below the diagram canvas, collapses vertically.
 * Collapse = shrinks to a 44px header strip; expand = 280px tray.
 *
 * Features:
 *  - Module-level summary cache (Map) — AI answers are stable for a session,
 *    so we never re-fetch the same node in the same browser session.
 *  - Auto-generates a concise AI summary on first open for each unique node.
 *  - Inline mini-chat scoped to the selected node.
 *  - "Open in full chat →" escape hatch.
 *  - Consistent design: uses CSS custom properties (--surface-2, --border, etc.)
 *    and the same button classes as the rest of the app (diagram-ask-btn, session-delete).
 */
import { useState, useEffect, useRef } from "react";
import { streamQuery } from "../api";

// ── Module-level cache ────────────────────────────────────────────────────────
// Keyed by `repo::label` — stable for the browser session since indexed
// repo content doesn't change between diagram interactions.
const summaryCache = new Map();

// ── Type colours (matches GraphDiagram.jsx TYPE_STYLE dots) ──────────────────
const TYPE_COLOR = {
  module:    "#fbbf24", class:    "#a78bfa", abstract: "#c4b5fd",
  mixin:     "#ddd6fe", service:  "#2dd4bf", database: "#fbbf24",
  external:  "#9CA3AF", input:    "#2dd4bf", transform: "#818CF8",
  output:    "#f472b6", edge:     "#6b7280",
};

// ── Chevron SVGs — same viewport as DiagramView fullscreen icons ──────────────
function ChevronDown() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
      <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
    </svg>
  );
}
function ChevronUp() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
      <path d="M7.646 4.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1-.708.708L8 5.707l-5.646 5.647a.5.5 0 0 1-.708-.708l6-6z"/>
    </svg>
  );
}

export default function NodeDetailPanel({ subject, repo, onClose, onOpenInChat }) {
  const { kind = "node", label, type, file, description, items = [], autoQuestion } = subject;
  const typeColor = TYPE_COLOR[type] || "#818CF8";
  const cacheKey  = `${repo}::${label}`;

  const [summary, setSummary]     = useState(() => summaryCache.get(cacheKey) || "");
  const [streaming, setStreaming] = useState(false);
  const [input, setInput]         = useState("");
  const [messages, setMessages]   = useState([]);
  const [collapsed, setCollapsed] = useState(false);
  const stopRef   = useRef(null);
  const bottomRef = useRef(null);

  // ── Auto-summary on open — skipped if already cached ─────────────────────
  useEffect(() => {
    setSummary(summaryCache.get(cacheKey) || "");
    setMessages([]);
    setInput("");
    setCollapsed(false); // always expand when switching to a new subject

    if (summaryCache.has(cacheKey)) {
      setStreaming(false);
      return;
    }

    setStreaming(true);
    const q = autoQuestion ||
      `Give a concise 3–4 sentence explanation of "${label}" in ${repo}: what it does, its key responsibilities, and what other parts of the codebase depend on it. Be specific.`;

    let content = "";
    const stop = streamQuery({
      question: q, repo, mode: "hybrid",
      onToken:   (t) => { content += t; setSummary(content); },
      onSources: () => {},
      onDone:    () => {
        summaryCache.set(cacheKey, content); // cache on completion
        setStreaming(false);
        stopRef.current = null;
      },
      onError:   () => { setStreaming(false); stopRef.current = null; },
    });
    stopRef.current = stop;
    return () => { stopRef.current?.(); };
  }, [label, repo]); // re-run only when a different node is selected

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, summary]);

  // ── Inline chat ───────────────────────────────────────────────────────────
  function handleAsk(e) {
    e?.preventDefault();
    const q = input.trim();
    if (!q || streaming) return;
    setInput("");

    const assistantId = Date.now();
    setMessages(prev => [
      ...prev,
      { role: "user", content: q },
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setStreaming(true);
    // Expand tray if collapsed so user can see the answer
    setCollapsed(false);

    const history = messages.map(m => ({ role: m.role, content: m.content }));
    let content = "";
    const stop = streamQuery({
      question: `Regarding "${label}" (${file || type}) in ${repo}: ${q}`,
      repo, mode: "hybrid", history,
      onToken: (t) => {
        content += t;
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content } : m));
      },
      onSources: () => {},
      onDone:  () => { setStreaming(false); stopRef.current = null; },
      onError: (err) => {
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, content: `Error: ${err}` } : m
        ));
        setStreaming(false);
        stopRef.current = null;
      },
    });
    stopRef.current = stop;
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); }
  }

  return (
    <div style={{
      width: "100%",
      height: collapsed ? 44 : 280,
      flexShrink: 0,
      overflow: "hidden",
      background: "var(--surface-2)",
      borderTop: "1px solid var(--border)",
      display: "flex", flexDirection: "column",
      fontFamily: "var(--mono)",
      transition: "height var(--transition-slow)",
    }}>

      {/* ── Header strip (always visible) ── */}
      <div style={{
        height: 44, minHeight: 44,
        padding: "0 12px",
        borderBottom: collapsed ? "none" : "1px solid var(--border-subtle)",
        background: "var(--surface-3)",
        display: "flex", alignItems: "center", gap: 8,
        cursor: "default",
      }}>
        {/* Type pill */}
        <span style={{
          fontSize: 8, fontWeight: 700, letterSpacing: "0.12em",
          textTransform: "uppercase", color: typeColor,
          background: `${typeColor}18`, border: `1px solid ${typeColor}38`,
          borderRadius: "var(--radius-sm)", padding: "2px 6px",
          flexShrink: 0,
        }}>{type || kind}</span>

        {/* Label */}
        <span style={{
          fontSize: 12, fontWeight: 700, color: "var(--text)",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          flex: 1,
        }}>{label}</span>

        {/* File path */}
        {file && (
          <span style={{
            fontSize: 9, color: "var(--faint)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            maxWidth: 160, flexShrink: 1,
          }}>{file}</span>
        )}

        {/* Streaming indicator */}
        {streaming && messages.length === 0 && (
          <span style={{
            width: 6, height: 6, borderRadius: "50%",
            background: "var(--accent-soft)",
            animation: "pulse 1.2s ease-in-out infinite",
            flexShrink: 0,
          }} />
        )}

        {/* Collapse/expand button */}
        <button
          onClick={() => setCollapsed(c => !c)}
          title={collapsed ? "Expand panel" : "Collapse panel"}
          className="session-delete"
          style={{ fontSize: "inherit", lineHeight: 1, padding: "3px 5px", opacity: 1 }}
        >
          {collapsed ? <ChevronUp /> : <ChevronDown />}
        </button>

        {/* Close button */}
        <button onClick={onClose} className="session-delete" style={{ fontSize: 16, padding: "0 3px", opacity: 1 }}>
          ×
        </button>
      </div>

      {/* ── Scrollable content (hidden when collapsed via parent height clip) ── */}
      <div style={{ flex: 1, overflowY: "auto", padding: "10px 14px 6px" }}>

        {/* Static description from AST */}
        {description && (
          <p style={{
            fontSize: 10.5, color: "var(--faint)", lineHeight: 1.6,
            margin: "0 0 10px",
          }}>{description}</p>
        )}

        {/* Method / export pills */}
        {items.length > 0 && (
          <div style={{ marginBottom: 10 }}>
            <div style={{
              fontSize: 8.5, color: "var(--faint)", letterSpacing: "0.1em",
              textTransform: "uppercase", marginBottom: 5,
            }}>Methods / Exports</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
              {items.map((item, i) => (
                <span key={i} style={{
                  fontSize: 9.5, background: "var(--accent-dim)",
                  border: "1px solid var(--accent-border)",
                  borderRadius: "var(--radius-sm)", padding: "2px 6px",
                  color: "var(--accent-soft)",
                }}>{item}</span>
              ))}
            </div>
          </div>
        )}

        <div style={{ borderTop: "1px solid var(--border-subtle)", margin: "0 0 10px" }} />

        {/* AI Summary */}
        <div>
          <div style={{
            fontSize: 8.5, color: "var(--faint)", letterSpacing: "0.1em",
            textTransform: "uppercase", marginBottom: 6,
          }}>AI Summary</div>
          {summary ? (
            <div style={{ fontSize: 11, color: "var(--accent-soft)", lineHeight: 1.65 }}>
              {summary}
              {streaming && messages.length === 0 && <span style={{ opacity: 0.4 }}>▋</span>}
            </div>
          ) : (
            <div style={{ fontSize: 11, color: "var(--faint)", opacity: 0.5 }}>Generating…</div>
          )}
        </div>

        {/* Follow-up messages */}
        {messages.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ borderTop: "1px solid var(--border-subtle)", marginBottom: 10 }} />
            {messages.map((m, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <div style={{
                  fontSize: 8.5, fontWeight: 700, letterSpacing: "0.08em",
                  textTransform: "uppercase", marginBottom: 2,
                  color: m.role === "user" ? "var(--text-2)" : "var(--accent-soft)",
                }}>
                  {m.role === "user" ? "You" : "Claude"}
                </div>
                <div style={{
                  fontSize: 11, lineHeight: 1.65, whiteSpace: "pre-wrap",
                  color: m.role === "user" ? "var(--muted)" : "var(--accent-soft)",
                }}>
                  {m.content}
                  {m.role === "assistant" && streaming && i === messages.length - 1 && (
                    <span style={{ opacity: 0.4 }}>▋</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Pinned footer: input + action (hidden when collapsed) ── */}
      {!collapsed && <div style={{
        flexShrink: 0,
        borderTop: "1px solid var(--border-subtle)",
        padding: "8px 12px",
        display: "flex", gap: 8, alignItems: "center",
        background: "var(--surface-3)",
      }}>
        <form onSubmit={handleAsk} style={{ display: "flex", gap: 5, flex: 1 }}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={streaming}
            placeholder={`Ask about ${label}…`}
            style={{
              flex: 1, background: "var(--accent-dim)",
              border: "1px solid var(--accent-border)",
              borderRadius: "var(--radius-sm)", color: "var(--text)", fontSize: 10.5,
              padding: "6px 8px", fontFamily: "var(--mono)",
              outline: "none", opacity: streaming ? 0.5 : 1,
            }}
          />
          <button type="submit" disabled={!input.trim() || streaming} className="diagram-ask-btn" style={{
            padding: "0 10px", fontSize: 13, fontWeight: 700,
            opacity: (!input.trim() || streaming) ? 0.35 : 1,
          }}>→</button>
        </form>
        <button
          onClick={() => onOpenInChat(
            `Explain "${label}" in ${repo} in detail — what does it do, what are its key methods or responsibilities, what calls it, and what does it depend on?`
          )}
          className="diagram-retry-btn"
          style={{ flexShrink: 0, padding: "5px 10px", fontSize: 9.5, letterSpacing: "0.04em" }}
        >
          Open in full chat →
        </button>
      </div>}
    </div>
  );
}

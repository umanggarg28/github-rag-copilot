/**
 * ReadmeView — on-demand README generator, rendered as a full-height view
 * alongside Chat and Diagram in the main content area.
 *
 * Design principles:
 *   - Matches the DiagramView pattern exactly: fills the main pane, no modal
 *   - Scrollable markdown panel with a sticky action bar at the top
 *   - Progress bar during generation, same visual language as tour/diagram loading
 *   - Copy + Regenerate in the action bar — accessible, not tucked in a corner
 */

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamReadme } from "../api";

/**
 * TerminalBlock — wraps every fenced code block in a macOS-style terminal window.
 * ReactMarkdown passes the <code> element as children of <pre>; we intercept
 * the <pre> render and fish the language out of code's className prop.
 */
function TerminalBlock({ children }) {
  const codeChild = Array.isArray(children) ? children[0] : children;
  const lang = codeChild?.props?.className?.replace("language-", "") ?? null;
  return (
    <div className="readme-terminal">
      <div className="readme-terminal-bar">
        <span className="readme-terminal-dots">
          <span /><span /><span />
        </span>
        {lang && <span className="readme-terminal-lang">{lang}</span>}
      </div>
      <pre className="readme-terminal-pre">{children}</pre>
    </div>
  );
}

const MD_COMPONENTS = { pre: TerminalBlock };

const STAGE_LABELS = {
  loading:    "Analysing repository…",
  generating: "Generating README…",
};

export default function ReadmeView({ repo, contextualAt, onClose }) {
  const [status,    setStatus]    = useState("idle");
  const [progress,  setProgress]  = useState(0);
  const [message,   setMessage]   = useState("");
  const [content,   setContent]   = useState(null);
  const [fromCache, setFromCache] = useState(false);
  const [error,     setError]     = useState(null);
  const [copied,    setCopied]    = useState(false);
  const [rawMode,   setRawMode]   = useState(false);  // preview vs markdown source
  const cancelRef = useRef(null);

  const generate = useCallback((force = false) => {
    cancelRef.current?.();
    setStatus("loading");
    setProgress(0);
    setContent(null);
    setError(null);

    cancelRef.current = streamReadme(repo, {
      force,
      onProgress: ({ stage, progress: p, message: m }) => {
        setStatus(stage || "loading");
        setProgress(p ?? 0);
        setMessage(m || STAGE_LABELS[stage] || "Working…");
      },
      onDone: ({ content: md, from_cache }) => {
        setContent(md);
        setFromCache(from_cache);
        setStatus("done");
        setProgress(1);
      },
      onError: (msg) => {
        setError(msg);
        setStatus("error");
      },
    });
  }, [repo]);

  useEffect(() => {
    generate(false);
    return () => cancelRef.current?.();
  }, [generate]);

  function handleCopy() {
    if (!content) return;
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }

  const isLoading = status === "loading" || status === "generating";

  return (
    <div className="readme-view">
      {/* Sticky action bar */}
      <div className="readme-view-bar">
        <div className="readme-view-bar-left">
          {onClose && (
            <button
              className="readme-bar-btn"
              onClick={onClose}
              title="Back"
              style={{ marginRight: 4 }}
            >
              <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M10 3L5 8l5 5"/>
              </svg>
            </button>
          )}
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-2)" }}>README.md</span>
          {fromCache && status === "done" && (
            <span className="readme-cache-badge" title="Loaded from cache — click Regenerate to refresh">
              cached
            </span>
          )}
          {isLoading && (
            <span className="readme-view-status">
              <span className="spinner" style={{ width: 10, height: 10 }} />
              {message || STAGE_LABELS[status]}
            </span>
          )}
        </div>

        <div className="readme-view-bar-right">
          {status === "done" && (
            <>
              {/* Preview / Markdown source toggle — reuses the app-wide view-toggle system */}
              <div className="view-toggle" style={{ padding: "2px", gap: "2px" }}>
                <button
                  className={`view-btn${!rawMode ? " active" : ""}`}
                  style={{ padding: "3px 12px", fontSize: 12 }}
                  onClick={() => setRawMode(false)}
                >Preview</button>
                <button
                  className={`view-btn${rawMode ? " active" : ""}`}
                  style={{ padding: "3px 12px", fontSize: 12 }}
                  onClick={() => setRawMode(true)}
                >Markdown</button>
              </div>
              <div style={{ width: 1, height: 14, background: "var(--border-strong)", margin: "0 6px" }} />
              <button
                className="readme-bar-btn"
                onClick={handleCopy}
                title={copied ? "Copied!" : "Copy markdown"}
              >
                {copied ? (
                  <>
                    <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                      <path d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/>
                    </svg>
                    Copied
                  </>
                ) : (
                  <>
                    <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                      <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25z"/>
                      <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25z"/>
                    </svg>
                    Copy
                  </>
                )}
              </button>
              <button
                className="readme-bar-btn"
                onClick={() => generate(true)}
                title="Regenerate README"
              >
                ↺ Regenerate
              </button>
            </>
          )}
        </div>
      </div>

      {/* Thin progress bar */}
      {isLoading && (
        <div className="readme-progress-wrap">
          <div className="readme-progress-bar" style={{ width: `${Math.max(progress * 100, 6)}%` }} />
        </div>
      )}

      {/* Quality tip — only shown when contextual retrieval hasn't been applied yet */}
      {status === "done" && !contextualAt && (
        <div className="readme-quality-note">
          <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" style={{ flexShrink: 0, opacity: 0.5 }}>
            <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 3.25a.75.75 0 110 1.5.75.75 0 010-1.5zM7.25 7h1.5v4.5h-1.5V7z"/>
          </svg>
          Quality improves when the repo is indexed with contextual retrieval enabled
          (<code>CONTEXTUAL_TOP_N&gt;0</code>). Re-index to get richer README output.
        </div>
      )}

      {/* Content area */}
      <div className="readme-view-body">
        {isLoading && (
          <div className="readme-view-placeholder">
            <div className="readme-view-placeholder-icon">
              <svg width="28" height="28" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.3" aria-hidden="true">
                <rect x="2" y="1" width="9" height="13" rx="1"/>
                <path d="M5 5h4M5 7h4M5 9h2"/>
                <path d="M9 1v3h3"/>
              </svg>
            </div>
            <p>{message || STAGE_LABELS[status]}</p>
          </div>
        )}

        {status === "error" && (
          <div className="readme-view-placeholder">
            <p style={{ color: "var(--text-2)" }}>{error}</p>
            <button className="readme-bar-btn" style={{ marginTop: 12 }} onClick={() => generate(true)}>
              Try again
            </button>
          </div>
        )}

        {content && (
          rawMode ? (
            /* Raw markdown source — monospace, line-numbered feel */
            <pre className="readme-raw">{content}</pre>
          ) : (
            <div className="readme-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>{content}</ReactMarkdown>
            </div>
          )
        )}
      </div>
    </div>
  );
}

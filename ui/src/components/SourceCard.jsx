import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const LANG_MAP = {
  python: "python", javascript: "javascript", typescript: "typescript",
  go: "go", rust: "rust", java: "java", cpp: "cpp", c: "c",
  markdown: "markdown", yaml: "yaml", json: "json", bash: "bash",
};

const CopyIcon = () => (
  <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25z"/>
    <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25z"/>
  </svg>
);

// showRepo=true when querying all repos — makes the source repo visible on every card
export default function SourceCard({ source, index, showRepo = false }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const lang = LANG_MAP[source.language] || "text";
  const name = source.name ? `${source.name}()` : null;

  // blob/HEAD always resolves to the default branch (main or master), avoiding
  // hardcoding "main" which breaks repos that still use "master".
  const githubUrl = `https://github.com/${source.repo}/blob/HEAD/${source.filepath}#L${source.start_line}`;
  const scorePercent = source.score != null ? `${Math.round(source.score * 100)}%` : null;
  // Color-code by confidence using design-system tokens (sage/warning/red)
  const scoreColor = !source.score ? null
    : source.score >= 0.90 ? "var(--green)"    // sage green — high confidence
    : source.score >= 0.70 ? "var(--warning)"  // warm amber — moderate
    : "var(--red)";                            // muted red — low confidence

  function handleCopy(e) {
    e.stopPropagation();
    navigator.clipboard.writeText(source.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <div className="source-item">
      <div
        className="source-header"
        onClick={() => setOpen((o) => !o)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" || e.key === " " ? setOpen(o => !o) : null}
        aria-expanded={open}
        aria-label={`Source ${index}: ${source.filepath}`}
      >
        <span className="source-num">{index}</span>
        {showRepo && source.repo && (
          <span className="source-repo-badge" title={source.repo}>
            {source.repo.split("/")[1]}
          </span>
        )}
        <span className="source-lang-badge" data-lang={lang}>{lang}</span>
        <a
          className="source-github-link"
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          title={`Open ${source.filepath} on GitHub`}
        >
          {source.filepath}
        </a>
        {name && <span className="source-name">{name}</span>}
        <span className="source-lines">L{source.start_line}–{source.end_line}</span>
        <a
          className="source-open-btn"
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          title={`Open on GitHub (L${source.start_line})`}
          aria-label="Open on GitHub"
        >
          <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M7 3H3a1 1 0 00-1 1v9a1 1 0 001 1h9a1 1 0 001-1V9"/>
            <path d="M13 3h-4m4 0v4m0-4L7 9"/>
          </svg>
        </a>
        {scorePercent && (
          <span
            className="source-score"
            title={`Relevance: ${source.score}`}
            style={scoreColor ? {
              color: scoreColor,
              borderColor: `${scoreColor}55`,
              background: `${scoreColor}18`,
            } : undefined}
          >{scorePercent}</span>
        )}
        <button
          className="source-copy-btn"
          onClick={handleCopy}
          title={copied ? "Copied!" : "Copy code"}
          aria-label="Copy code to clipboard"
        >
          {copied ? "✓" : <CopyIcon />}
        </button>
        <svg
          className={`source-chevron ${open ? "open" : ""}`}
          width="10" height="10" viewBox="0 0 16 16"
          fill="none" stroke="currentColor" strokeWidth="2"
          strokeLinecap="round" strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="m6 4 4 4-4 4"/>
        </svg>
      </div>

      {open && (
        <div className="source-code">
          <SyntaxHighlighter
            language={lang}
            style={oneDark}
            customStyle={{ fontSize: 12, margin: 0, background: '#141210', borderRadius: 0 }}
            lineNumberStyle={{ color: 'rgba(237,228,206,0.22)', fontSize: 11, minWidth: 36, paddingRight: 12 }}
            showLineNumbers
            startingLineNumber={source.start_line}
          >
            {source.text}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  );
}

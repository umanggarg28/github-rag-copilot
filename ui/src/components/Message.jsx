import { useState, useCallback, Suspense, lazy, forwardRef } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import SourceCard from "./SourceCard";

// Lazy-load MermaidBlock — deferred so mermaid.js doesn't bloat the initial bundle.
const MermaidBlock = lazy(() => import("./MermaidBlock"));

// ReactMarkdown renders fenced code blocks as <pre><code>...</code></pre>.
// If we override only `code`, ReactMarkdown wraps the whole thing in a <p>,
// giving <p><pre>...</pre></p> — invalid HTML (pre can't be inside p).
//
// Fix: override `pre` to render just its children (no wrapper), so
// SyntaxHighlighter's own <pre> is the only one. Then in `code` we check
// whether it has a language class (block code) or not (inline code).
const mdComponents = {
  // Strip the <pre> wrapper — SyntaxHighlighter adds its own
  pre({ children }) {
    return <>{children}</>;
  },
  code({ className, children, ...props }) {
    const lang = /language-(\w+)/.exec(className || "")?.[1];
    if (lang === "diagram" || lang === "mermaid") {
      // Agent drew a diagram — render as SVG via mermaid.js.
      // We intercept both "diagram" (our custom tag) and "mermaid" (model's natural tag).
      return (
        <Suspense fallback={
          <div style={{ padding: "12px 0", color: "var(--muted)", fontSize: 12 }}>
            <span className="spinner" style={{ marginRight: 8 }} /> Rendering diagram…
          </div>
        }>
          <MermaidBlock mermaid={String(children).replace(/\n$/, "")} />
        </Suspense>
      );
    }
    if (lang) {
      // Block code with a language tag → syntax-highlighted
      return (
        <SyntaxHighlighter
          language={lang}
          style={oneDark}
          customStyle={{ fontSize: 13, background: '#141210', borderRadius: 8, border: '1px solid rgba(237,228,206,0.08)', borderLeft: '2px solid rgba(212,132,90,0.45)', margin: '10px 0' }}
        >
          {String(children).replace(/\n$/, "")}
        </SyntaxHighlighter>
      );
    }
    // Inline code → plain <code>
    return <code className={className} {...props}>{children}</code>;
  },
};

// Thought bubble — shows the LLM's reasoning before a tool call.
// Displayed inline in the trace timeline so users can see WHY the agent
// chose each tool, not just WHAT it called.
function AgentThought({ text }) {
  return (
    <div className="agent-thought">
      <div className="agent-step-node">
        <span className="agent-step-dot thought-dot" />
      </div>
      <div className="agent-thought-text">{text}</div>
    </div>
  );
}

// Convert tool+input into a short human-readable label shown in the step header.
// Reads like a sentence fragment so the trace feels like watching the agent think,
// not like reading a JSON dump.
function formatStepQuery(tool, input) {
  if (!input) return "";
  switch (tool) {
    case "search_code":    return input.query        || JSON.stringify(input);
    case "search_symbol":  return input.symbol_name  || JSON.stringify(input);
    case "list_files":     return input.path ? `${input.repo}/${input.path}` : (input.repo || JSON.stringify(input));
    case "find_callers":   return input.function_name || JSON.stringify(input);
    case "get_file_chunk": return input.filepath
      ? `${input.filepath} (L${input.start_line}–${input.end_line})`
      : JSON.stringify(input);
    case "read_file":      return input.filepath || JSON.stringify(input);
    case "note":           return input.key ? `${input.key}: ${input.value}` : JSON.stringify(input);
    case "recall_notes":   return "checking notes";
    case "trace_calls":    return input.symbol_name || JSON.stringify(input);
    default:               return input.query || input.name || JSON.stringify(input);
  }
}

// Individual agent step — renders as a node in the connected timeline chain.
//
// Collapsed by default once a step is no longer the active one.
// isActive = this step is currently executing (isLast && streaming).
// Clicking a completed (non-active) step toggles its output open/closed.
function AgentStep({ step, isLast, icon, streaming }) {
  const isActive  = isLast && streaming;
  const isPending = !step.output && isActive;
  // manualExpand lets users re-open a completed step; resets when step becomes active again
  const [manualExpand, setManualExpand] = useState(false);
  const showOutput = isActive || manualExpand;

  const isLong = step.output && step.output.length > 300;
  const [outputExpanded, setOutputExpanded] = useState(false);

  const toggle = () => {
    if (!isActive) setManualExpand(v => !v);
  };

  return (
    <div className={`agent-step ${step.output ? "done" : "pending"}${isLast ? " last" : ""}${!showOutput && step.output ? " collapsed" : ""}`}>
      {/* Node dot on the vertical line + arrow connector */}
      <div className="agent-step-node">
        <span className="agent-step-dot" />
        <span className="agent-step-arrow">→</span>
      </div>

      {/* Step body */}
      <div className="agent-step-body">
        <div
          className="agent-step-header"
          onClick={toggle}
          style={{ cursor: !isActive && step.output ? "pointer" : "default" }}
        >
          <span className="agent-step-icon">{icon}</span>
          <span className="agent-step-tool">{step.tool}</span>
          <span className="agent-step-query">
            {formatStepQuery(step.tool, step.input)}
          </span>
          {isPending && <span className="spinner" style={{ marginLeft: "auto", flexShrink: 0, width: 10, height: 10 }} />}
          {!isActive && step.output && (
            <span className="agent-step-chevron" style={{ marginLeft: "auto", opacity: 0.4 }}>
              <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                {manualExpand ? <path d="m4 6 4 4 4-4"/> : <path d="m6 4 4 4-4 4"/>}
              </svg>
            </span>
          )}
        </div>

        {showOutput && step.output && (
          <>
            <div
              className={`agent-step-output${outputExpanded ? " expanded" : isLong ? " clipped" : ""}`}
              onClick={() => isLong && !outputExpanded && setOutputExpanded(true)}
            >
              {step.output}
            </div>
            {isLong && !outputExpanded && (
              <button className="agent-step-expand" onClick={() => setOutputExpanded(true)}>
                Show full output ↓
              </button>
            )}
            {isLong && outputExpanded && (
              <button className="agent-step-expand" onClick={() => setOutputExpanded(false)}>
                Collapse ↑
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ToolCallTrace shows the agent's reasoning steps as a connected timeline —
// visually similar to how Claude Code shows "Agent → Bash → Read" with
// vertical lines connecting each step.
//
// DURING streaming:  always expanded so user can watch the agent think live.
// AFTER completion:  collapsible via the toggle header.
function ToolCallTrace({ steps, streaming, iterations, model }) {
  const [expanded, setExpanded] = useState(true);
  if (!steps || steps.length === 0) return null;

  // Tool name → icon SVG for clean visual scanning (no emoji)
  const toolIcon = {
    search_code:    <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><circle cx="6.5" cy="6.5" r="4.5" stroke="currentColor" strokeWidth="1.5"/><path d="M10 10l3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>,
    search_symbol:  <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M3 4h10M3 8h7M3 12h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/><circle cx="12" cy="11" r="2.5" stroke="currentColor" strokeWidth="1.3"/><path d="M14 13l1.5 1.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>,
    list_files:     <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><rect x="1" y="2" width="14" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.5"/><path d="M4 6h8M4 9h5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>,
    get_file_chunk: <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><rect x="3" y="1" width="8" height="10" rx="1" stroke="currentColor" strokeWidth="1.5"/><path d="M5 5h4M5 7h3M9 1v3h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>,
    read_file:      <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><rect x="2" y="1" width="9" height="12" rx="1" stroke="currentColor" strokeWidth="1.5"/><path d="M4 5h5M4 7h5M4 9h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/><path d="M11 8l3 3-3 3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/></svg>,
    find_callers:   <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><circle cx="4" cy="4" r="2" stroke="currentColor" strokeWidth="1.4"/><circle cx="12" cy="12" r="2" stroke="currentColor" strokeWidth="1.4"/><path d="M6 4h2a2 2 0 012 2v2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/></svg>,
    note:           <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M3 2h10a1 1 0 011 1v8l-3 3H3a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.4"/><path d="M11 11v3l3-3h-3z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/><path d="M4 5h6M4 7h6M4 9h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/></svg>,
    recall_notes:   <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.4"/><path d="M8 5v3.5l2.5 1.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg>,
    trace_calls:    <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><circle cx="3" cy="8" r="2" stroke="currentColor" strokeWidth="1.4"/><circle cx="13" cy="4" r="2" stroke="currentColor" strokeWidth="1.4"/><circle cx="13" cy="12" r="2" stroke="currentColor" strokeWidth="1.4"/><path d="M5 8h3M8 8L11 5M8 8l3 3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>,
  };
  const defaultIcon = <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="5" stroke="currentColor" strokeWidth="1.5"/><path d="M8 5v3l2 1" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/></svg>;

  // Compute the index of the last non-thought step so we can pass isLast correctly
  const lastToolIdx = steps.reduce((acc, s, i) => s.type !== "thought" ? i : acc, -1);

  const stepsEl = (
    <div className="agent-trace-steps">
      {/* Vertical connector line running the full height */}
      <div className="agent-trace-line" />
      {steps.map((step, i) => {
        if (step.type === "thought") {
          return <AgentThought key={i} text={step.text} />;
        }
        return (
          <AgentStep
            key={i}
            step={step}
            isLast={i === lastToolIdx}
            icon={toolIcon[step.tool] || defaultIcon}
            streaming={streaming}
          />
        );
      })}
    </div>
  );

  return (
    <div className={`agent-trace${streaming ? " live" : ""}`}>
      <button
        className="agent-trace-toggle"
        onClick={() => !streaming && setExpanded(v => !v)}
        aria-expanded={expanded}
        style={{ cursor: streaming ? "default" : "pointer" }}
      >
        {/* "Agent" node — the root of the chain */}
        <span className="agent-trace-root-dot" />
        <span className="agent-trace-root-label">Agent</span>
        {streaming
          ? <span className="spinner" style={{ marginLeft: 6, width: 10, height: 10 }} />
          : (() => {
              // Use backend's authoritative iteration count when available.
              // steps.length = tool calls only; iterations = full ReAct turns
              // (includes the final answer turn, so it's always >= steps.length).
              const count = iterations ?? steps.length;
              const label = iterations ? "iteration" : "tool call";
              return (
                <>
                  <span className="agent-trace-count">{count} {label}{count !== 1 ? "s" : ""}</span>
                  {model && (
                    <span style={{ opacity: 0.4, fontStyle: "italic", fontSize: 10.5, marginLeft: 6 }}
                      title={`Model: ${model}`}>
                      {model.split("/").pop()}
                    </span>
                  )}
                </>
              );
            })()
        }
        {!streaming && (
          <span className="agent-trace-chevron">
            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              {expanded ? <path d="m4 6 4 4 4-4"/> : <path d="m6 4 4 4-4 4"/>}
            </svg>
          </span>
        )}
      </button>
      {expanded && stepsEl}
      {/* When collapsed, show the first thought as a one-line summary so users can still see the agent's reasoning intent */}
      {!expanded && !streaming && (() => {
        const firstThought = steps.find(s => s.type === "thought");
        if (!firstThought) return null;
        return (
          <div className="agent-trace-thought-summary" title={firstThought.text}>
            "{firstThought.text.length > 120 ? firstThought.text.slice(0, 120) + "…" : firstThought.text}"
          </div>
        );
      })()}
    </div>
  );
}


// ConfidenceBadge — rendered after model-based grading completes.
// high   = all claims confirmed in sources → green check (shown in pipeline bar only)
// medium = mostly supported, minor extrapolation → amber warning
// low    = claims not backed by sources → red warning
const CONFIDENCE_CONFIG = {
  high:   { color: "#10b981", bg: "rgba(16,185,129,0.10)", icon: "✓", label: "High confidence" },
  medium: { color: "#f59e0b", bg: "rgba(245,158,11,0.10)", icon: "◐", label: "Medium confidence" },
  low:    { color: "#ef4444", bg: "rgba(239,68,68,0.10)",  icon: "⚠", label: "Low confidence"   },
};

function ConfidenceBadge({ grade }) {
  const cfg = CONFIDENCE_CONFIG[grade.confidence] || CONFIDENCE_CONFIG.medium;
  return (
    <div style={{
      display: "inline-flex", alignItems: "flex-start", gap: 6,
      marginTop: 8,
      padding: "5px 10px",
      background: cfg.bg,
      border: `1px solid ${cfg.color}33`,
      borderRadius: 6,
      fontSize: 11.5,
      fontFamily: "var(--mono)",
      color: cfg.color,
      maxWidth: "100%",
    }}>
      <span style={{ flexShrink: 0, marginTop: 1 }}>{cfg.icon} {cfg.label}</span>
      {grade.note && (
        <span style={{ color: "var(--text-2)", fontFamily: "inherit", fontSize: 11 }}>
          — {grade.note}
        </span>
      )}
    </div>
  );
}

// Copy-answer button — appears on hover over the assistant message.
// Copies the raw markdown text so developers can paste it into docs/code.
function CopyAnswerButton({ content }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }, [content]);
  return (
    <button
      className="copy-answer-btn"
      onClick={handleCopy}
      title={copied ? "Copied!" : "Copy answer"}
      aria-label="Copy answer to clipboard"
    >
      {copied
        ? <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/></svg>
        : <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25z"/></svg>
      }
    </button>
  );
}

const Message = forwardRef(function Message({ msg, onDiagramThis, onRetry, showRepo = false }, ref) {
  const isUser = msg.role === "user";

  return (
    <div ref={ref} className={`message ${msg.role}`}>
      {isUser ? (
        <div className="bubble">{msg.content}</div>
      ) : (
        <>
          {/* Assistant avatar — ✦ for agent responses, code icon for RAG.
              This matches the ✦ badge on agent sessions in the sidebar,
              making the visual language consistent: ✦ = agent mode. */}
          <div className="message-avatar assistant" aria-hidden="true">
            {msg.iterations
              ? <span style={{ fontSize: 13, lineHeight: 1, color: "white", opacity: 0.9 }}>✦</span>
              : <svg width="14" height="14" viewBox="0 0 18 18" fill="none">
                  <path d="M5.5 5L2 9l3.5 4" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" strokeOpacity="0.95"/>
                  <path d="M12.5 5L16 9l-3.5 4" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" strokeOpacity="0.95"/>
                  <circle cx="9" cy="9" r="1.2" fill="white" fillOpacity="0.7"/>
                </svg>
            }
          </div>

          {/* All assistant content in a column wrapper */}
          <div className="message-content">
            {/* Agent reasoning trace */}
            {msg.toolCalls && msg.toolCalls.length > 0 && (
              <ToolCallTrace steps={msg.toolCalls} streaming={msg.streaming} iterations={msg.iterations} model={msg.model} />
            )}


            {/* "Thinking…" shown before first tool call in agent mode */}
            {msg.streaming && msg.currentTool === null && !msg.content && (!msg.toolCalls || msg.toolCalls.length === 0) && !msg.phase && (
              <div className="agent-thinking">
                <span className="spinner" role="status" aria-label="Thinking" />
                Thinking…
              </div>
            )}

            {/* RAG retrieval phase indicator — makes the invisible retrieval step visible.
                "searching" = waiting for vector search to return sources.
                "generating" = sources received, LLM is now streaming the answer. */}
            {msg.streaming && msg.phase && (
              <div className={`stream-phase stream-phase--${msg.phase}`}>
                <span className="stream-phase-dot" />
                {msg.phase === "searching"
                  ? "Searching code…"
                  : `Found ${msg.sourceCount ?? "?"} source${msg.sourceCount !== 1 ? "s" : ""} · Generating answer…`
                }
              </div>
            )}

            {/* Rate-limit countdown banner — shown instead of a hard error */}
            {msg.rateLimited && (
              <div className="rate-limit-banner">
                <span className="rate-limit-spinner" aria-hidden="true" />
                <span className="rate-limit-text">{msg.content}</span>
                {onRetry && msg.retryQuestion && (
                  <button
                    className="rate-limit-retry-btn"
                    onClick={() => onRetry(msg.retryQuestion)}
                  >
                    Retry now
                  </button>
                )}
              </div>
            )}

            {/* Answer bubble */}
            <div className="bubble" style={{ position: "relative", display: msg.rateLimited ? "none" : undefined }}>
              <ReactMarkdown components={mdComponents}>
                {msg.content || " "}
              </ReactMarkdown>
              {/* Show cursor whenever streaming, not just when no tool active */}
              {msg.streaming && <span className="cursor" aria-hidden="true" />}
              {/* Copy-answer button — visible on hover; lets devs paste the answer */}
              {!msg.streaming && msg.content && <CopyAnswerButton content={msg.content} />}
            </div>

            {/* Pipeline provenance — shows every retrieval stage that fired for this answer.
                Positioned HERE (before sources) so it's immediately visible after the answer,
                not buried below N source cards. Quality features only shown when they ran. */}
            {!msg.streaming && msg.queryType && !msg.iterations && (
              <div className="pipeline-provenance">
                {msg.pipeline?.hyde && (
                  <>
                    <span className="pipeline-stage pipeline-quality" title="Hypothetical Document Embeddings: a code snippet was generated from your question and used for retrieval instead of the raw query text">
                      HyDE
                    </span>
                    <span className="pipeline-sep">→</span>
                  </>
                )}
                {msg.pipeline?.expanded > 0 && (
                  <>
                    <span className="pipeline-stage pipeline-quality" title={`Query Expansion: ${msg.pipeline.expanded} alternative phrasings were searched and merged with RRF`}>
                      +{msg.pipeline.expanded} expansions
                    </span>
                    <span className="pipeline-sep">→</span>
                  </>
                )}
                <span className="pipeline-stage" title={`${msg.queryType === "hybrid" ? "Hybrid: dense semantic vectors + BM25 keyword search, fused with Reciprocal Rank Fusion" : msg.queryType === "semantic" ? "Dense semantic search: nearest-neighbour lookup in 768-dim embedding space" : "BM25 keyword search: exact and fuzzy term matching"}`}>
                  <svg width="9" height="9" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="6.5" cy="6.5" r="4.5"/><path d="M10 10l3.5 3.5"/></svg>
                  {msg.queryType} search
                </span>
                <span className="pipeline-sep">→</span>
                <span
                  className="pipeline-stage"
                  title={msg.pipeline?.reranker === "cohere" ? "Cohere rerank-v3.5: API cross-encoder re-scores every candidate against your question for maximum precision" : "Local ms-marco cross-encoder: re-scores candidates locally without an API call"}
                >
                  {msg.pipeline?.reranker === "cohere" ? "cohere re-ranked" : "re-ranked"}
                </span>
                <span className="pipeline-sep">→</span>
                <span className="pipeline-stage" title={`${msg.sources?.length ?? 0} code chunk${(msg.sources?.length ?? 0) !== 1 ? "s" : ""} were retrieved and passed as context to the LLM`}>{msg.sources?.length ?? 0} source{(msg.sources?.length ?? 0) !== 1 ? "s" : ""}</span>
                <span className="pipeline-sep">→</span>
                <span className="pipeline-stage" title="The LLM generated this answer using only the retrieved sources as context — it cannot see code outside these chunks">generated</span>
                {msg.model && (
                  <>
                    <span className="pipeline-sep">·</span>
                    <span
                      className="pipeline-stage"
                      style={{ opacity: 0.45, fontStyle: "italic" }}
                      title={`Model: ${msg.model}`}
                    >
                      {msg.model.split("/").pop()}
                    </span>
                  </>
                )}
                {msg.grade && msg.grade.confidence !== "unknown" && (
                  <>
                    <span className="pipeline-sep">→</span>
                    <span className={`pipeline-stage pipeline-grade-${msg.grade.confidence}`}>
                      {msg.grade.confidence === "high" ? "✓" : msg.grade.confidence === "low" ? "⚠" : "◐"} {msg.grade.confidence}
                    </span>
                  </>
                )}
              </div>
            )}

            {/* Badges + Sources — query type shown as sources header for context */}
            {/* (agent iteration count is shown in the ToolCallTrace header above) */}
            {msg.sources && msg.sources.length > 0 && !msg.streaming && (
              <div className="sources">
                <div className="sources-header">
                  {msg.sources.length} source{msg.sources.length > 1 ? "s" : ""}
                  {msg.queryType && !msg.iterations && (
                    <span className="query-type-badge" style={{ marginLeft: 8 }}>{msg.queryType}</span>
                  )}
                </div>
                {msg.sources.map((s, i) => (
                  <SourceCard key={i} source={s} index={i + 1} showRepo={showRepo} />
                ))}
                {/* "Diagram this →" button — switches to diagram tab with focused-files context */}
                {onDiagramThis && (
                  <button
                    className="diagram-this-btn"
                    onClick={() => onDiagramThis(msg.sources)}
                    title="Switch to Diagram tab and highlight the files cited in this answer"
                  >
                    Diagram this →
                  </button>
                )}
              </div>
            )}
            {/* Query type badge when no sources (e.g. factual answer with no retrieved chunks) */}
            {!msg.streaming && !msg.iterations && msg.queryType && !(msg.sources?.length > 0) && (
              <span className="query-type-badge">{msg.queryType}</span>
            )}

            {/* Standalone confidence badge for medium/low — shows the note text */}
            {msg.grade && msg.grade.confidence !== "unknown" && msg.grade.confidence !== "high" && (
              <ConfidenceBadge grade={msg.grade} />
            )}

          </div>
        </>
      )}
    </div>
  );
});

export default Message;

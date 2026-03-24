import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import SourceCard from "./SourceCard";

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
    if (lang) {
      // Block code with a language tag → syntax-highlighted
      return (
        <SyntaxHighlighter language={lang} style={oneDark} customStyle={{ fontSize: 13 }}>
          {String(children).replace(/\n$/, "")}
        </SyntaxHighlighter>
      );
    }
    // Inline code → plain <code>
    return <code className={className} {...props}>{children}</code>;
  },
};

// ToolCallTrace shows the agent's reasoning steps.
//
// DURING streaming:  shows steps live, expanded, as they accumulate.
// AFTER completion:  collapses to a toggle button to keep the UI clean.
//
// This is the "glass box" view — users can watch the LLM reason in real time,
// see what it searched for, and what it found, step by step.
function ToolCallTrace({ steps, streaming }) {
  const [expanded, setExpanded] = useState(true);
  if (!steps || steps.length === 0) return null;

  // Tool name → emoji for quick visual scanning
  const toolIcon = { search_code: "🔍", get_file_chunk: "📄", find_callers: "🔗" };

  const stepsEl = (
    <div className="agent-trace-steps">
      {steps.map((step, i) => (
        <div key={i} className={`agent-step ${step.output ? "done" : "pending"}`}>
          <div className="agent-step-header">
            <span className="agent-step-icon">{toolIcon[step.tool] || "⚙️"}</span>
            <span className="agent-step-tool">{step.tool}</span>
            <span className="agent-step-query">
              {step.input?.query || step.input?.function_name || JSON.stringify(step.input)}
            </span>
            {/* Spinner on the last step while waiting for result */}
            {!step.output && i === steps.length - 1 && (
              <span className="spinner" style={{ marginLeft: "auto", flexShrink: 0 }} />
            )}
          </div>
          {step.output && (
            <div className="agent-step-output">{step.output}</div>
          )}
        </div>
      ))}
    </div>
  );

  if (streaming) {
    // Live view: always expanded while agent is running
    return (
      <div className="agent-trace live">
        <div className="agent-trace-label">
          ✦ Agent reasoning · {steps.length} step{steps.length !== 1 ? "s" : ""}
        </div>
        {stepsEl}
      </div>
    );
  }

  // Collapsed view after completion
  return (
    <div className="agent-trace">
      <button
        className="agent-trace-toggle"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? "▼" : "▶"} Reasoning trace · {steps.length} step{steps.length !== 1 ? "s" : ""}
      </button>
      {expanded && stepsEl}
    </div>
  );
}

export default function Message({ msg }) {
  const isUser = msg.role === "user";

  return (
    <div className={`message ${msg.role}`}>
      {isUser ? (
        <div className="bubble">{msg.content}</div>
      ) : (
        <>
          {/* Agent reasoning trace — live during streaming, collapsible after */}
          {msg.toolCalls && msg.toolCalls.length > 0 && (
            <ToolCallTrace steps={msg.toolCalls} streaming={msg.streaming} />
          )}

          {/* "Thinking…" shown before the first tool call fires */}
          {msg.streaming && msg.currentTool === null && !msg.content && (!msg.toolCalls || msg.toolCalls.length === 0) && (
            <div className="agent-thinking">
              <span className="spinner" />
              Thinking…
            </div>
          )}

          {/* Answer bubble */}
          <div className="bubble">
            <ReactMarkdown components={mdComponents}>
              {msg.content || " "}
            </ReactMarkdown>
            {msg.streaming && !msg.currentTool && <span className="cursor" />}
          </div>

          {/* Query type badge or agent iterations badge */}
          {!msg.streaming && msg.iterations && (
            <span className="query-type-badge">agent · {msg.iterations} step{msg.iterations !== 1 ? "s" : ""}</span>
          )}
          {!msg.streaming && msg.queryType && !msg.iterations && (
            <span className="query-type-badge">{msg.queryType}</span>
          )}

          {/* Sources (only for non-agent RAG) */}
          {msg.sources && msg.sources.length > 0 && !msg.streaming && (
            <div className="sources">
              <div className="sources-header">
                {msg.sources.length} source{msg.sources.length > 1 ? "s" : ""}
              </div>
              {msg.sources.map((s, i) => (
                <SourceCard key={i} source={s} index={i + 1} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

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

export default function Message({ msg }) {
  const isUser = msg.role === "user";

  return (
    <div className={`message ${msg.role}`}>
      {isUser ? (
        <div className="bubble">{msg.content}</div>
      ) : (
        <>
          {/* Answer bubble */}
          <div className="bubble">
            <ReactMarkdown components={mdComponents}>
              {msg.content || " "}
            </ReactMarkdown>
            {msg.streaming && <span className="cursor" />}
          </div>

          {/* Query type badge */}
          {msg.queryType && !msg.streaming && (
            <span className="query-type-badge">{msg.queryType}</span>
          )}

          {/* Sources */}
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

import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import SourceCard from "./SourceCard";

function CodeBlock({ className, children }) {
  const lang = /language-(\w+)/.exec(className || "")?.[1] || "text";
  return (
    <SyntaxHighlighter language={lang} style={oneDark} customStyle={{ fontSize: 13 }}>
      {String(children).replace(/\n$/, "")}
    </SyntaxHighlighter>
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
          {/* Answer bubble */}
          <div className="bubble">
            <ReactMarkdown components={{ code: CodeBlock }}>
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

import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const LANG_MAP = {
  python: "python", javascript: "javascript", typescript: "typescript",
  go: "go", rust: "rust", java: "java", cpp: "cpp", c: "c",
  markdown: "markdown", yaml: "yaml", json: "json", bash: "bash",
};

export default function SourceCard({ source, index }) {
  const [open, setOpen] = useState(false);
  const lang = LANG_MAP[source.language] || "text";
  const name = source.name ? `${source.name}()` : null;

  return (
    <div className="source-item">
      <div className="source-header" onClick={() => setOpen((o) => !o)}>
        <span className="source-num">{index}</span>
        <span className="source-path">{source.filepath}</span>
        {name && <span className="source-name">{name}</span>}
        <span className="source-lines">
          L{source.start_line}–{source.end_line}
        </span>
        <span className="source-score">{source.score}</span>
        <span className={`source-chevron ${open ? "open" : ""}`}>▶</span>
      </div>

      {open && (
        <div className="source-code">
          <SyntaxHighlighter
            language={lang}
            style={oneDark}
            customStyle={{ fontSize: 12, margin: 0 }}
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

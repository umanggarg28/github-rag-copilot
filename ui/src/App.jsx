import { useState, useEffect, useRef, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import Message from "./components/Message";
import { fetchRepos, streamQuery } from "./api";

export default function App() {
  const [repos, setRepos]           = useState([]);
  const [activeRepo, setActiveRepo] = useState(null);
  const [mode, setMode]             = useState("hybrid");
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [streaming, setStreaming]   = useState(false);

  const bottomRef  = useRef(null);
  const scrollRef  = useRef(null);
  const stopStream = useRef(null); // cleanup fn for active SSE

  // Load repos on mount
  const loadRepos = useCallback(async () => {
    try {
      const data = await fetchRepos();
      setRepos(data.repos || []);
    } catch {
      // backend not ready yet — ignore
    }
  }, []);

  useEffect(() => { loadRepos(); }, [loadRepos]);

  // Auto-scroll: instant during streaming, smooth otherwise
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (streaming) {
      if (distFromBottom < 120) bottomRef.current?.scrollIntoView({ behavior: "instant" });
    } else {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streaming]);

  function handleSubmit(e) {
    e?.preventDefault();
    const question = input.trim();
    if (!question || streaming) return;
    setInput("");

    // Add user message
    const userMsg = { role: "user", content: question };
    // Add placeholder assistant message
    const assistantId = Date.now();
    const assistantMsg = {
      id: assistantId, role: "assistant",
      content: "", sources: [], queryType: null, streaming: true,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    const stop = streamQuery({
      question,
      repo: activeRepo,
      mode,
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) => m.id === assistantId
            ? { ...m, content: m.content + token }
            : m
          )
        );
      },
      onSources: (sources, queryType) => {
        setMessages((prev) =>
          prev.map((m) => m.id === assistantId
            ? { ...m, sources, queryType }
            : m
          )
        );
      },
      onDone: () => {
        setMessages((prev) =>
          prev.map((m) => m.id === assistantId ? { ...m, streaming: false } : m)
        );
        setStreaming(false);
        stopStream.current = null;
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((m) => m.id === assistantId
            ? { ...m, content: `Error: ${err}`, streaming: false }
            : m
          )
        );
        setStreaming(false);
        stopStream.current = null;
      },
    });

    stopStream.current = stop;
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function handleClear() {
    if (stopStream.current) { stopStream.current(); stopStream.current = null; }
    setMessages([]);
    setStreaming(false);
  }

  const placeholder = activeRepo
    ? `Ask about ${activeRepo}…`
    : "Ask about any indexed repo…";

  return (
    <div className="layout">
      <Sidebar
        repos={repos}
        activeRepo={activeRepo}
        onSelectRepo={setActiveRepo}
        onReposChange={loadRepos}
        mode={mode}
        onModeChange={setMode}
      />

      <div className="main">
        {/* Header */}
        <div className="chat-header">
          {activeRepo
            ? <span className="repo-badge">{activeRepo}</span>
            : <span className="no-repo">All indexed repos</span>
          }
          {messages.length > 0 && (
            <button className="clear-btn" onClick={handleClear}>Clear chat</button>
          )}
        </div>

        {/* Messages */}
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="icon">💬</div>
            <h2>Ask about a codebase</h2>
            <p>
              {repos.length === 0
                ? "Index a GitHub repo using the sidebar, then ask questions about it."
                : "Select a repo from the sidebar or ask across all indexed repos."}
            </p>
          </div>
        ) : (
          <div className="messages" ref={scrollRef}>
            {messages.map((msg, i) => <Message key={msg.id ?? i} msg={msg} />)}
            <div ref={bottomRef} />
          </div>
        )}

        {/* Input */}
        <div className="input-bar">
          <textarea
            rows={1}
            placeholder={placeholder}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={streaming}
          />
          <button
            className="btn"
            onClick={handleSubmit}
            disabled={!input.trim() || streaming}
          >
            {streaming ? <span className="spinner" /> : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}

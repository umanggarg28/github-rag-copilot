import { useState, useEffect, useRef, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import Message from "./components/Message";
import CodeGraph from "./components/CodeGraph";
import { fetchRepos, streamQuery, streamAgentQuery } from "./api";

export default function App() {
  const [repos, setRepos]           = useState([]);
  const [activeRepo, setActiveRepo] = useState(null);
  const [mode, setMode]             = useState("hybrid");
  const [agentMode, setAgentMode]   = useState(false);
  const [view, setView]             = useState("chat");  // "chat" | "graph"
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [streaming, setStreaming]   = useState(false);

  const bottomRef   = useRef(null);
  const scrollRef   = useRef(null);
  const textareaRef = useRef(null);
  const stopStream  = useRef(null); // cleanup fn for active SSE

  // Auto-grow textarea as user types
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

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

    // Add user message + placeholder assistant message
    const userMsg = { role: "user", content: question };
    const assistantId = Date.now();
    const assistantMsg = {
      id: assistantId, role: "assistant",
      content: "", sources: [], queryType: null, streaming: true,
      // Agent-mode extras:
      toolCalls: [], currentTool: null, iterations: null,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    // ── Common callbacks ──────────────────────────────────────────────────────
    const onToken = (token) =>
      setMessages((prev) =>
        prev.map((m) => m.id === assistantId ? { ...m, content: m.content + token } : m)
      );

    const onError = (err) => {
      setMessages((prev) =>
        prev.map((m) => m.id === assistantId
          ? { ...m, content: `Error: ${err}`, streaming: false }
          : m
        )
      );
      setStreaming(false);
      stopStream.current = null;
    };

    let stop;

    if (agentMode) {
      // ── Agent mode: ReAct loop with live tool-call trace ──────────────────
      stop = streamAgentQuery({
        question,
        repo: activeRepo,
        onToolCall: (tool, input) => {
          // Show spinner with tool name while agent is calling
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId
              ? { ...m, currentTool: tool }
              : m
            )
          );
          // Append to the tool call trace (output will be filled by onToolResult)
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId
              ? { ...m, toolCalls: [...m.toolCalls, { tool, input, output: "" }] }
              : m
            )
          );
        },
        onToolResult: (tool, output) => {
          // Fill in the output of the last tool call in the trace
          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== assistantId) return m;
              const calls = [...m.toolCalls];
              // Find last call for this tool (most recent) and fill its output
              for (let i = calls.length - 1; i >= 0; i--) {
                if (calls[i].tool === tool && !calls[i].output) {
                  calls[i] = { ...calls[i], output };
                  break;
                }
              }
              return { ...m, toolCalls: calls, currentTool: "thinking" };
            })
          );
        },
        onToken,
        onDone: (iterations) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId
              ? { ...m, streaming: false, currentTool: null, iterations }
              : m
            )
          );
          setStreaming(false);
          stopStream.current = null;
        },
        onError,
      });
    } else {
      // ── Plain RAG mode: single retrieval → stream tokens ──────────────────
      stop = streamQuery({
        question,
        repo: activeRepo,
        mode,
        onToken,
        onSources: (sources, queryType) =>
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, sources, queryType } : m)
          ),
        onDone: () => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, streaming: false } : m)
          );
          setStreaming(false);
          stopStream.current = null;
        },
        onError,
      });
    }

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
        agentMode={agentMode}
        onAgentModeChange={setAgentMode}
      />

      <div className="main">
        {/* Header */}
        <div className="chat-header">
          {activeRepo
            ? <span className="repo-badge">{activeRepo}</span>
            : <span className="no-repo">All indexed repos</span>
          }
          <div className="header-actions">
            {/* View toggle: Chat ↔ Graph */}
            {activeRepo && (
              <div className="view-toggle">
                <button
                  className={`view-btn ${view === "chat" ? "active" : ""}`}
                  onClick={() => setView("chat")}
                >Chat</button>
                <button
                  className={`view-btn ${view === "graph" ? "active" : ""}`}
                  onClick={() => setView("graph")}
                >Graph ✦</button>
              </div>
            )}
            {view === "chat" && messages.length > 0 && (
              <button className="clear-btn" onClick={handleClear}>Clear</button>
            )}
          </div>
        </div>

        {/* ── Graph view ── */}
        {view === "graph" && activeRepo && (
          <CodeGraph
            repo={activeRepo}
            onAskAbout={(question) => {
              setView("chat");
              setInput(question);
              // Small delay so textarea is rendered before focus
              setTimeout(() => textareaRef.current?.focus(), 50);
            }}
          />
        )}

        {/* ── Chat view ── */}
        {view === "chat" && (
          <>
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
                ref={textareaRef}
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
          </>
        )}
      </div>
    </div>
  );
}

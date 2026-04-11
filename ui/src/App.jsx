import { useState, useEffect, useRef, useCallback } from "react";
import posthog from "posthog-js";
import Sidebar from "./components/Sidebar";
import Message from "./components/Message";
import DiagramView from "./components/DiagramView";
import { fetchRepos, streamQuery, streamAgentQuery, fetchMcpStatus, fetchMcpPrompt, fetchAgentModels } from "./api";

export default function App() {
  const [repos, setRepos]           = useState([]);
  const [reposLoading, setReposLoading] = useState(true);
  const [activeRepo, setActiveRepo] = useState(null);
  const [mode, setMode]             = useState("hybrid");
  const [agentMode, setAgentMode]   = useState(() => localStorage.getItem('ghrc_agentMode') === 'true');
  const [view, setView]             = useState("chat");  // "chat" | "graph"
  const [messages, setMessages]     = useState([]);
  const [sessions, setSessions]     = useState([]);      // recent sessions for active repo
  const [lastSources, setLastSources] = useState([]); // sources from last RAG query (kept for future use)
  const [focusFiles, setFocusFiles]   = useState(null); // filepaths from last "Diagram this →" click
  const [input, setInput]           = useState("");
  const [streaming, setStreaming]   = useState(false);
  const [backendOk, setBackendOk]   = useState(null); // null=unknown, true=ok, false=error
  const [currentSessionId, setCurrentSessionId] = useState(null); // highlights active session in sidebar
  const prevRepoRef  = useRef(null); // track previous repo before switching
  const messagesRef  = useRef([]);   // always-fresh messages ref to avoid stale closures
  const sessionIdRef = useRef(null); // ID of the current open session
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem('ghrc_sidebarCollapsed') === 'true'
  );
  // Prompt autocomplete: shown when input starts with "/"
  const [prompts, setPrompts]         = useState([]);      // MCP prompt list
  const [promptMenu, setPromptMenu]   = useState(false);   // dropdown visible
  const [promptFilter, setPromptFilter] = useState("");    // text after "/"

  // Model selector: available models fetched from /agent/models
  const [agentModels, setAgentModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(
    () => localStorage.getItem('ghrc_selectedModel') || null
  );
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const modelMenuRef = useRef(null);

  const bottomRef           = useRef(null);
  const scrollRef           = useRef(null);
  const latestAssistantRef  = useRef(null); // top of the current streaming assistant message
  const textareaRef    = useRef(null);
  const stopStream     = useRef(null);       // cleanup fn for active SSE
  const streamingRef   = useRef(false);      // always-fresh streaming flag for event handlers
  const countdownTimer = useRef(null);       // setInterval handle for rate-limit auto-retry
  const handleSubmitRef = useRef(null);      // stable ref so closures can call handleSubmit
  const msgIdCounter    = useRef(0);         // monotonic counter for message IDs — avoids Date.now() collisions
  const rateLimitRetries = useRef(0);        // consecutive rate-limit count — resets on success

  // ── Multi-session persistence (localStorage, up to 10 sessions per repo) ───
  // Modelled on rag-research-copilot: each session has an id, title (first
  // question truncated to 55 chars), messages array, and ISO timestamp.
  // Sessions are stored as `ghrc_sessions_{repo}` → JSON array, newest first.

  // null = landing page (no repo chosen yet), "all" = All repos explicitly selected
  // Both map to the same storage key; only "all" ever writes sessions.
  function sessionsKey(repo) { return `ghrc_sessions_${repo || "all"}`; }

  function readSessions(repo) {
    try { return JSON.parse(localStorage.getItem(sessionsKey(repo)) || "[]"); } catch { return []; }
  }

  function writeSessions(repo, list) {
    try { localStorage.setItem(sessionsKey(repo), JSON.stringify(list)); } catch {}
  }

  // Strip transient streaming fields before saving so reloaded messages are clean
  function cleanMsgs(msgs) {
    return msgs.map(({ streaming: _s, currentTool: _ct, phase: _p, ...m }) => m);
  }

  function upsertSession(repo, sessionId, msgs, isAgentMode = false) {
    if (!repo || !sessionId || msgs.length === 0) return;
    const title = msgs.find(m => m.role === "user")?.content?.slice(0, 55) ?? "Untitled";
    const session = { id: sessionId, title, messages: cleanMsgs(msgs), timestamp: new Date().toISOString(), agentMode: isAgentMode };
    const prev = readSessions(repo);
    const exists = prev.some(s => s.id === sessionId);
    const next = exists
      ? prev.map(s => s.id === sessionId ? session : s)
      : [session, ...prev].slice(0, 10);
    writeSessions(repo, next);
    return next;
  }

  // Keep refs in sync so event handlers always read the latest values
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  useEffect(() => { streamingRef.current = streaming; }, [streaming]);
  // Persist agent mode preference across page loads
  useEffect(() => { localStorage.setItem('ghrc_agentMode', agentMode); }, [agentMode]);
  // Persist selected model
  useEffect(() => {
    if (selectedModelId) localStorage.setItem('ghrc_selectedModel', selectedModelId);
    else localStorage.removeItem('ghrc_selectedModel');
  }, [selectedModelId]);
  // Fetch available agent models once on mount
  useEffect(() => {
    fetchAgentModels().then(models => {
      setAgentModels(models);
      // If no model selected yet, default to the first available one
      setSelectedModelId(prev => {
        if (prev && models.some(m => m.id === prev)) return prev;
        const first = models.find(m => m.available);
        return first ? first.id : null;
      });
    });
  }, []);
  // Close model menu when clicking outside
  useEffect(() => {
    function onClickOutside(e) {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target)) {
        setModelMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);
  // Keep handleSubmitRef pointing at the latest handleSubmit (avoids stale closures
  // in the rate-limit countdown which captures this ref via closure).
  // We update it on every render so it always has the current state in scope.
  useEffect(() => { handleSubmitRef.current = (q) => handleSubmit(null, q); });

  // Load sessions list whenever active repo changes
  useEffect(() => {
    // Save the current session for the old repo before switching
    if (prevRepoRef.current && prevRepoRef.current !== activeRepo && sessionIdRef.current) {
      upsertSession(prevRepoRef.current, sessionIdRef.current, messagesRef.current, agentMode);
    }
    prevRepoRef.current = activeRepo;
    sessionIdRef.current = null;
    setCurrentSessionId(null);
    setMessages([]);
    setLastSources([]);
    setSessions(readSessions(activeRepo));
    // Auto-navigate to Explore view when a specific repo is selected.
    // Reset to chat for "All repos" or landing page — diagram needs a single repo.
    if (activeRepo && activeRepo !== "all") setView("graph");
    else setView("chat");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRepo]);

  // Auto-save current session after each complete streaming exchange
  const prevStreaming = useRef(false);
  useEffect(() => {
    if (prevStreaming.current && !streaming && activeRepo && sessionIdRef.current) {
      const next = upsertSession(activeRepo, sessionIdRef.current, messagesRef.current, agentMode);
      if (next) setSessions(next);
    }
    prevStreaming.current = streaming;
  }, [streaming, messages, activeRepo]);

  // ── Session actions ─────────────────────────────────────────────────────────

  function handleLoadSession(session) {
    if (streaming) return;
    // Save whatever is currently open before switching
    if (sessionIdRef.current && messagesRef.current.length > 0) {
      const next = upsertSession(activeRepo, sessionIdRef.current, messagesRef.current, agentMode);
      if (next) setSessions(next);
    }
    sessionIdRef.current = session.id;
    setCurrentSessionId(session.id);
    setMessages(session.messages);
    setLastSources([]);
    setView("chat");
    // Scroll to the last message after the messages render
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "instant" }), 50);
  }

  function handleDeleteSession(sessionId) {
    const next = readSessions(activeRepo).filter(s => s.id !== sessionId);
    writeSessions(activeRepo, next);
    setSessions(next);
    // If we deleted the open session, clear the chat
    if (sessionIdRef.current === sessionId) {
      sessionIdRef.current = null;
      setCurrentSessionId(null);
      setMessages([]);
    }
  }

  function handleRenameSession(sessionId, newTitle) {
    const prev = readSessions(activeRepo);
    const next = prev.map(s => s.id === sessionId ? { ...s, title: newTitle } : s);
    writeSessions(activeRepo, next);
    setSessions(next);
  }

  function toggleSidebarCollapse() {
    const next = !sidebarCollapsed;
    setSidebarCollapsed(next);
    localStorage.setItem('ghrc_sidebarCollapsed', String(next));
  }

  function handleDiagramThis(sources) {
    // Extract unique filepaths from the message's source cards, then switch to
    // the Diagram tab showing an architecture view with a focused-files banner.
    const files = [...new Set((sources || []).map(s => s.filepath))];
    setFocusFiles(files.length > 0 ? files : null);
    setView("graph");
  }

  function handleStop() {
    if (stopStream.current) { stopStream.current(); stopStream.current = null; }
    setStreaming(false);
    // Mark the in-progress message as done (no streaming cursor)
    setMessages(prev => prev.map(m =>
      m.streaming ? { ...m, streaming: false, phase: null, currentTool: null } : m
    ));
  }

  // ⌘K / Ctrl+K — focus the input from anywhere in the app.
  // Productivity Tool must_have: keyboard-shortcuts (ui-ux-pro-max-skill #16).
  // navigator.platform is deprecated — prefer userAgentData (Chrome 90+) with fallback.
  const isMac = (navigator.userAgentData?.platform ?? navigator.platform).toUpperCase().includes("MAC");
  useEffect(() => {
    function onGlobalKey(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (view === "graph") setView("chat");
        // Small delay if we just switched views (textarea may not be mounted yet)
        setTimeout(() => textareaRef.current?.focus(), 20);
      }
      // Escape — stop streaming (mirrors Claude/ChatGPT behaviour)
      // Use streamingRef to avoid stale closure (this effect only reruns on view change)
      if (e.key === "Escape" && streamingRef.current) {
        if (stopStream.current) { stopStream.current(); stopStream.current = null; }
        setStreaming(false);
        setMessages(prev => prev.map(m =>
          m.streaming ? { ...m, streaming: false, phase: null, currentTool: null } : m
        ));
      }
    }
    window.addEventListener("keydown", onGlobalKey);
    return () => window.removeEventListener("keydown", onGlobalKey);
  }, [view]);

  // Auto-grow textarea as user types
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  // Load repos on mount — also tracks backend health for the header status dot.
  // Auto-selects the only repo if exactly one is indexed, so new users land in
  // single-repo mode rather than the "All repos" cross-repo view.
  const loadRepos = useCallback(async () => {
    setReposLoading(true);
    try {
      const data = await fetchRepos();
      const list = data.repos || [];
      setRepos(list);
      setBackendOk(true);
      // Auto-select if only one repo is indexed and nothing is selected yet
      if (list.length === 1 && !activeRepo) {
        setActiveRepo(list[0].slug);
      }
    } catch {
      setBackendOk(false);
    } finally {
      setReposLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { loadRepos(); }, [loadRepos]);

  // Load MCP prompts once on mount for the "/" autocomplete
  useEffect(() => {
    fetchMcpStatus()
      .then(info => setPrompts(info.prompts || []))
      .catch(() => {});
  }, []);

  // Scroll to the TOP of the assistant message the moment it first appears.
  // We track the last scrolled-to ID so this only fires once per response.
  const scrolledToId = useRef(null);
  useEffect(() => {
    const streamingMsg = messages.find(m => m.role === "assistant" && m.streaming);
    if (streamingMsg && streamingMsg.id !== scrolledToId.current) {
      scrolledToId.current = streamingMsg.id;
      setTimeout(() => {
        latestAssistantRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    }
  }, [messages]);

  // While streaming, keep scrolling to bottom only if user is already near bottom.
  // After streaming ends, do a final smooth scroll to bottom.
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

  // Accept optional retryQuestion so the rate-limit countdown can re-submit
  // without reading stale `input` state from a closure.
  function handleSubmit(e, retryQuestion = null) {
    e?.preventDefault();
    const question = retryQuestion || input.trim();
    if (!question || streaming) return;
    if (!retryQuestion) setInput(""); // only clear the box on a fresh submit

    // Assign a session ID on the first message of a new conversation
    if (!sessionIdRef.current) {
      const id = Date.now();
      sessionIdRef.current = id;
      setCurrentSessionId(id);
    }

    // Build conversation history from completed exchanges (not the current one).
    // Only include messages with content — skip failed/empty responses.
    // Cap at 10 items (5 back-and-forth exchanges) to stay within LLM token limits.
    const completedMsgs = messagesRef.current.filter(m => !m.streaming && m.content);
    const history = completedMsgs
      .slice(-10)
      .map(m => ({ role: m.role, content: m.content }));

    // Track query event
    posthog.capture("query_submitted", { repo: activeRepo, mode: agentMode ? "agent" : "rag" });

    // Add user message + placeholder assistant message.
    // On auto-retry (retryQuestion set), skip the user message — it's already in the chat
    // from the first attempt. Adding it again causes duplicate question bubbles.
    const userMsg = { role: "user", content: question };
    // Use a unique counter (not Date.now()) so auto-retry can never create a
    // new message with the same ID as the old one — preventing a stale onSources
    // callback from the old RAG stream polluting the new message's state.
    const assistantId = ++msgIdCounter.current;
    const assistantMsg = {
      id: assistantId, role: "assistant",
      // Store mode explicitly so Message.jsx never has to infer it from mutable state.
      // phase, queryType, etc. can all be overwritten by async callbacks; mode cannot.
      mode: agentMode ? "agent" : "rag",
      content: "", sources: [], queryType: null, streaming: true,
      phase: agentMode ? null : "searching",
      sourceCount: null,
      toolCalls: [], currentTool: null, iterations: null,
    };
    if (retryQuestion) {
      setMessages((prev) => [...prev, assistantMsg]);
    } else {
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
    }
    setStreaming(true);

    // ── Common callbacks ──────────────────────────────────────────────────────
    const onToken = (token) =>
      setMessages((prev) =>
        prev.map((m) => m.id === assistantId ? { ...m, content: m.content + token } : m)
      );

    const onError = (err) => {
      // Make errors actionable: distinguish network vs backend vs rate limit vs unknown.
      const errStr = String(err);
      let friendly = `Error: ${err}`;
      let isRateLimit = false;

      if (errStr.includes("fetch") || errStr.includes("network") || errStr.includes("Failed to fetch")) {
        friendly = "Cannot reach the backend (localhost:8000). Is it running?\n\nTry: `uvicorn backend.main:app --reload`";
      } else if (errStr.includes("502") || errStr.includes("503")) {
        friendly = "Backend returned a server error (502/503). Try refreshing in a few seconds.";
      } else if (
        errStr.includes("429") ||
        errStr.includes("rate-limited") ||
        errStr.includes("rate limited") ||
        errStr.includes("daily limit")
      ) {
        isRateLimit = true;
        friendly = "⟳ Rate limited — retrying in 45s";
      } else if (errStr.includes("timeout") || errStr.includes("Timeout")) {
        friendly = "Request timed out. The query may be too complex — try a simpler question.";
      }

      setMessages((prev) =>
        prev.map((m) => m.id === assistantId
          ? { ...m, content: friendly, streaming: false, rateLimited: isRateLimit, retryQuestion: isRateLimit ? question : null }
          : m
        )
      );
      setStreaming(false);
      stopStream.current = null;

      // Rate-limit auto-retry: count down 45 s, then re-submit the same question.
      // Max 2 auto-retries — after that, show a permanent error so it doesn't loop forever.
      // The user can also click "Retry now" to skip the wait (also counted against the limit).
      if (isRateLimit) {
        rateLimitRetries.current += 1;
        const attempt = rateLimitRetries.current;

        if (attempt > 2) {
          // Give up — show a clear message instead of looping endlessly
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: "Rate limit hit too many times. Wait a minute and try again.", streaming: false, rateLimited: false }
              : m
          ));
          return;
        }

        let secsLeft = 45;
        if (countdownTimer.current) clearInterval(countdownTimer.current);
        countdownTimer.current = setInterval(() => {
          secsLeft -= 1;
          if (secsLeft <= 0) {
            clearInterval(countdownTimer.current);
            countdownTimer.current = null;
            // Stop the old stream before retrying — prevents stale onSources/onToken
            // callbacks from the previous attempt firing on the new message.
            stopStream.current?.();
            stopStream.current = null;
            setMessages(prev => prev.filter(m => m.id !== assistantId));
            handleSubmitRef.current?.(question);
          } else {
            setMessages(prev => prev.map(m =>
              m.id === assistantId
                ? { ...m, content: `⟳ Rate limited (attempt ${attempt}/2) — retrying in ${secsLeft}s` }
                : m
            ));
          }
        }, 1000);
      }
    };

    let stop;

    if (agentMode) {
      // ── Agent mode: ReAct loop with live tool-call trace ──────────────────
      stop = streamAgentQuery({
        question,
        repo: activeRepo === "all" ? null : activeRepo,
        model_id: selectedModelId || undefined,
        history,
        onThought: (text) => {
          // Append a thought entry to the trace — rendered as a reasoning bubble
          // before the tool call that follows it.
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId
              ? { ...m, toolCalls: [...m.toolCalls, { type: "thought", text }] }
              : m
            )
          );
        },
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
        onSources: (sources) => {
          // Agent mode: store collected file references for the source cards panel.
          // These arrive just before the "done" event, after all tool calls complete.
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId
              ? { ...m, sources: sources || [] }
              : m
            )
          );
        },
        onDone: (iterations, model) => {
          rateLimitRetries.current = 0; // reset on success
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId
              ? { ...m, streaming: false, currentTool: null, iterations, model }
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
        repo: activeRepo === "all" ? null : activeRepo,
        mode,
        history,
        onToken,
        onSources: (sources, queryType, pipeline, model) => {
          // Transition from "searching" → "generating" so the phase indicator updates.
          // pipeline = {hyde, expanded, reranker} — shows which quality features fired.
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId
              ? { ...m, sources, queryType, pipeline, model, phase: "generating", sourceCount: sources.length }
              : m)
          );
          setLastSources(sources || []);
        },
        onGrade: (grade) => {
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, grade } : m)
          );
        },
        onDone: () => {
          rateLimitRetries.current = 0; // reset on success
          setMessages((prev) =>
            prev.map((m) => m.id === assistantId ? { ...m, streaming: false, phase: null } : m)
          );
          setStreaming(false);
          stopStream.current = null;
        },
        onError,
      });
    }

    stopStream.current = stop;
  }

  function handleInputChange(e) {
    const val = e.target.value;
    setInput(val);
    // Show prompt menu when input is just "/" or "/partial"
    if (val.startsWith("/") && !val.includes(" ")) {
      setPromptFilter(val.slice(1).toLowerCase());
      setPromptMenu(true);
    } else {
      setPromptMenu(false);
    }
  }

  async function handleSelectPrompt(prompt) {
    setPromptMenu(false);
    // Build arguments: pass activeRepo if we have one
    const args = activeRepo ? { repo: activeRepo } : {};
    try {
      const result = await fetchMcpPrompt(prompt.name, args);
      setInput(result.text);
      setTimeout(() => textareaRef.current?.focus(), 0);
    } catch {
      // Fallback: just fill with the prompt name as a question
      setInput(`/${prompt.name}`);
    }
  }

  function handleKeyDown(e) {
    if (promptMenu && e.key === "Escape") {
      setPromptMenu(false);
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function handleClear() {
    if (stopStream.current) { stopStream.current(); stopStream.current = null; }
    if (countdownTimer.current) { clearInterval(countdownTimer.current); countdownTimer.current = null; }
    // Save the current session before starting a new one
    if (sessionIdRef.current && messagesRef.current.length > 0) {
      const next = upsertSession(activeRepo, sessionIdRef.current, messagesRef.current, agentMode);
      if (next) setSessions(next);
    }
    sessionIdRef.current = null;
    setCurrentSessionId(null);
    setMessages([]);
    setStreaming(false);
  }

  // "/" triggers MCP prompt autocomplete — surface this in the placeholder so
  // users discover it without reading docs.
  const placeholder = activeRepo
    ? `Ask about ${activeRepo}… (type / for AI prompts)`
    : "Ask about any indexed repo…";

  return (
    <div className={`layout${sidebarCollapsed ? " layout-collapsed" : ""}`}>
      {/* Sidebar overlay for mobile — closes sidebar when clicking outside */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} aria-hidden="true" />
      )}

      <Sidebar
        repos={repos}
        reposLoading={reposLoading}
        activeRepo={activeRepo}
        onSelectRepo={(repo) => { setActiveRepo(repo); posthog.capture("repo_selected", { repo }); }}
        onReposChange={loadRepos}
        mode={mode}
        onModeChange={setMode}
        agentMode={agentMode}
        onAgentModeChange={setAgentMode}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onLoadSession={handleLoadSession}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        collapsed={sidebarCollapsed}
        onToggleCollapse={toggleSidebarCollapse}
      />

      <div className="main">
        {/* Header */}
        {/* 3-column grid: left (repo badge) | center (toggle) | right (actions)
            Equal 1fr flanks guarantee the center column is always truly centred,
            regardless of asymmetric content on either side — the Linear/Vercel pattern. */}
        <div className="chat-header">
          {/* LEFT — hamburger (mobile) + repo context */}
          <div className="header-left">
            <button
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open navigation"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                <path d="M1 2.75A.75.75 0 011.75 2h12.5a.75.75 0 010 1.5H1.75A.75.75 0 011 2.75zm0 5A.75.75 0 011.75 7h12.5a.75.75 0 010 1.5H1.75A.75.75 0 011 7.75zM1.75 12a.75.75 0 000 1.5h12.5a.75.75 0 000-1.5H1.75z"/>
              </svg>
            </button>
            {backendOk !== null && (
              <span
                className="backend-dot"
                title={backendOk ? "Backend connected" : "Backend unreachable"}
                style={{ background: backendOk ? "var(--green)" : "var(--red)" }}
              />
            )}
            {activeRepo ? (
              <span className="repo-badge">
                {activeRepo === "all" ? "All repos" : (() => {
                  const [owner, name] = activeRepo.split("/");
                  return <><span style={{ opacity: 0.55, fontWeight: 400 }}>{owner}/</span><span style={{ fontWeight: 600 }}>{name}</span></>;
                })()}
              </span>
            ) : (
              <span className="no-repo">All indexed repos</span>
            )}
          </div>

          {/* CENTER — view toggle, only when a specific repo is selected */}
          <div className="header-center">
            {activeRepo && activeRepo !== "all" && (
              <div className="view-toggle">
                <button
                  className={`view-btn ${view === "chat" ? "active" : ""}`}
                  onClick={() => setView("chat")}
                >Chat</button>
                <button
                  className={`view-btn ${view === "graph" ? "active" : ""}`}
                  onClick={() => { setView("graph"); posthog.capture("diagram_view_opened", { repo: activeRepo }); }}
                >Diagram <span style={{ fontSize: 8, verticalAlign: "middle", color: "var(--accent-soft)", marginLeft: 2 }}>●</span></button>
              </div>
            )}
          </div>

          {/* RIGHT — contextual actions */}
          <div className="header-actions">
            {view === "chat" && messages.length > 0 && (
              <button className="clear-btn" onClick={handleClear}>New Chat</button>
            )}
          </div>
        </div>

        {/* ── Diagram view ── */}
        {view === "graph" && activeRepo && (
          <DiagramView
            repo={activeRepo}
            focusFiles={focusFiles}
            onAskAbout={(question) => {
              setView("chat");
              setFocusFiles(null);
              setInput(question);
              setTimeout(() => textareaRef.current?.focus(), 50);
            }}
          />
        )}

        {/* ── Chat view ── */}
        {view === "chat" && (
          <>
            {messages.length === 0 ? (
              <div className="empty-state">
                {activeRepo === "all" && repos.length > 0 ? (
                  // "All repos" explicitly selected with repos indexed — cross-repo query mode
                  <div className="suggest-state">
                    <h2>Ask across all repos</h2>
                    <p>Searching <strong>{repos.length} indexed repos</strong> at once — {repos.map(r => r.slug.split("/")[1]).join(", ")}. Results show which repo each source comes from.</p>
                    <div className="suggestions">
                      {[
                        "Compare the architectures of these repos — what patterns do they share?",
                        "Which repo is most complex and what makes it that way?",
                        "Find all the main entry points across these repos",
                        "How do these repos handle configuration and environment setup?",
                        "What are the common abstractions or design patterns across repos?",
                      ].map(q => (
                        <button key={q} className="suggestion-btn"
                          onClick={() => { setInput(q); textareaRef.current?.focus(); }}>
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : !activeRepo || activeRepo === "all" ? (
                  // No repo selected yet (landing), or All repos selected but nothing indexed
                  <div className="onboarding-steps">
                    <div className="onboarding-header">
                      <svg width="72" height="72" viewBox="0 0 24 24" fill="none"
                        style={{ marginBottom: 24, filter: "drop-shadow(0 0 18px rgba(91,143,249,0.65)) drop-shadow(0 0 6px rgba(91,143,249,0.90))" }}>
                        {/* N — pulses bright */}
                        <path className="compass-north" d="M12 2 L14.5 7 L12 12 L9.5 7 Z" fill="var(--accent)"/>
                        {/* S — delayed fade */}
                        <path className="compass-south" d="M12 22 L13.5 17 L12 12 L10.5 17 Z" fill="var(--accent)"/>
                        {/* E — delayed fade */}
                        <path className="compass-east" d="M22 12 L17 10.5 L12 12 L17 13.5 Z" fill="var(--accent)"/>
                        {/* W — delayed fade */}
                        <path className="compass-west" d="M2 12 L7 10.5 L12 12 L7 13.5 Z" fill="var(--accent)"/>
                        {/* Center */}
                        <circle cx="12" cy="12" r="1.5" fill="white"/>
                      </svg>
                      <div className="onboarding-headline">Map <em>any</em> codebase</div>
                      <div className="onboarding-sub">Index any public repo and ask questions about the code — architecture, data flow, classes, functions, and more.</div>
                    </div>
                    <div className="onboarding-step active">
                      <span className="step-num">1</span>
                      <div>
                        <strong>Paste a GitHub URL in the sidebar</strong>
                        <p>e.g. <code>github.com/karpathy/nanoGPT</code> — indexes every function and class.</p>
                      </div>
                    </div>
                    <div className="onboarding-step">
                      <span className="step-num">2</span>
                      <div>
                        <strong>Ask anything about the code</strong>
                        <p>e.g. <em>"How does the main loop work?"</em> — finds relevant code and explains it with citations.</p>
                      </div>
                    </div>
                    <div className="onboarding-step">
                      <span className="step-num">3</span>
                      <div>
                        <strong>Use Explore to map the structure</strong>
                        <p>Generates a concept map of key components and how they connect.</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  // Repo selected — show mode-aware suggestions + feature discovery cards
                  <div className="suggest-state">
                    <h2>How does {activeRepo.split("/")[1]} work?</h2>


                    {agentMode ? (
                      <>
                        <div className="mode-hint" style={{ marginBottom: 12 }}>
                          <strong>Agent mode</strong> runs the ReAct loop — search → observe → reason → search again. This is the same pattern used in production agents. Watch the tool calls trace as it works.
                        </div>
                        <div className="suggestions">
                          {[
                            `Walk through ${activeRepo.split("/")[1]}'s architecture from entry point to output`,
                            "What are the most important functions and how do they connect?",
                            "Draw a diagram showing how the main components connect",
                            "How is error handling and edge cases managed across the codebase?",
                            "How does data flow from input to the final result?",
                          ].map(q => (
                            <button key={q} className="suggestion-btn"
                              onClick={() => { setInput(q); textareaRef.current?.focus(); }}>
                              {q}
                            </button>
                          ))}
                        </div>
                        <button className="graph-hint-btn" onClick={() => setView("graph")}>
                          Explore Diagrams for {activeRepo.split("/")[1]} →
                        </button>
                      </>
                    ) : (
                      <>
                        <p>Try one of these or ask your own:</p>
                        <div className="suggestions">
                          {[
                            `What is the overall architecture of ${activeRepo.split("/")[1]}?`,
                            "What are the main entry points and how does the code flow?",
                            "What are the key classes and what does each one do?",
                            "How is data processed and transformed through the system?",
                            "What are the external dependencies and how are they used?",
                          ].map(q => (
                            <button key={q} className="suggestion-btn"
                              onClick={() => { setInput(q); textareaRef.current?.focus(); }}>
                              {q}
                            </button>
                          ))}
                        </div>
                        {/* Secondary action row — below suggestions so it doesn't compete */}
                        <div className="suggest-footer">
                          <button className="suggest-footer-btn" onClick={() => setAgentMode(true)}>
                            ✦ Agent mode
                          </button>
                          <span className="suggest-footer-sep">·</span>
                          <button className="suggest-footer-btn" onClick={() => setView("graph")}>
                            ◫ Diagrams
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div
                className="messages"
                ref={scrollRef}
                role="log"
                aria-live="polite"
                aria-label="Chat messages"
              >
                {messages.map((msg, i) => (
                  <Message
                    key={msg.id ?? i}
                    msg={msg}
                    showRepo={!activeRepo}
                    onDiagramThis={activeRepo ? handleDiagramThis : null}
                    ref={msg.role === "assistant" && msg.streaming ? latestAssistantRef : null}
                    onRetry={msg.rateLimited && msg.retryQuestion ? (q) => {
                      // User clicked "Retry now" — cancel countdown and re-submit immediately
                      if (countdownTimer.current) { clearInterval(countdownTimer.current); countdownTimer.current = null; }
                      setMessages(prev => prev.filter(m => m.id !== msg.id));
                      handleSubmit(null, q);
                    } : null}
                  />
                ))}
                <div ref={bottomRef} />
              </div>
            )}

            {/* Input */}
            <div className="input-bar">
              {/* Prompt autocomplete dropdown — shown when input starts with "/" */}
              {promptMenu && prompts.length > 0 && (() => {
                const filtered = prompts.filter(p =>
                  p.name.toLowerCase().includes(promptFilter)
                );
                return filtered.length > 0 ? (
                  <div className="prompt-menu">
                    <div className="prompt-menu-label">MCP Prompts</div>
                    {filtered.map(p => (
                      <button
                        key={p.name}
                        className="prompt-menu-item"
                        onMouseDown={(e) => { e.preventDefault(); handleSelectPrompt(p); }}
                      >
                        <span className="prompt-menu-name">/{p.name}</span>
                        <span className="prompt-menu-desc">{p.description?.slice(0, 60)}…</span>
                      </button>
                    ))}
                  </div>
                ) : null;
              })()}
              <div className="input-row">
                <textarea
                  ref={textareaRef}
                  rows={1}
                  placeholder={agentMode ? "Ask a complex question…" : placeholder}
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  onBlur={() => setTimeout(() => setPromptMenu(false), 150)}
                  disabled={streaming}
                />
                <button
                  className={`btn${streaming ? " btn-stop" : ""}`}
                  onClick={streaming ? handleStop : handleSubmit}
                  disabled={!streaming && !input.trim()}
                  aria-label={streaming ? "Stop generating" : agentMode ? "Run Agent" : "Send"}
                  title={streaming ? "Stop (Esc)" : undefined}
                >
                  {streaming
                    ? <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true"><rect x="1.5" y="1.5" width="9" height="9" rx="2"/></svg>
                    : <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M2 8h10M8 4l4 4-4 4" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  }
                </button>
                {/* ⌘K hint — inside input-row so it positions relative to the textarea, not the whole bar */}
                {!streaming && !input && (
                  <div className="input-hint" aria-hidden="true">{isMac ? "⌘K" : "Ctrl+K"}</div>
                )}
              </div>
              {/* Agent mode footer: badge + model selector */}
              {agentMode && (
                <div className="input-footer-row">
                  <div className="input-mode-badge" title="Agent mode — runs the ReAct loop (Reason + Act): searches the codebase, reads the result, decides if it needs more context, then searches again. The same pattern production agents use.">✦ Agent</div>
                  {agentModels.length > 0 && (() => {
                    const active = agentModels.find(m => m.id === selectedModelId) || agentModels.find(m => m.available) || agentModels[0];
                    return (
                      <div className="model-selector" ref={modelMenuRef}>
                        <button
                          className="model-selector-btn"
                          onClick={() => setModelMenuOpen(o => !o)}
                          title={active?.note}
                        >
                          <span className="model-selector-name">{active?.name ?? "Auto"}</span>
                          {active && <span className={`model-speed-badge model-speed-${active.speed}`}>{active.speed_label}</span>}
                          {/* chevron */}
                          <svg className={`model-chevron${modelMenuOpen ? " open" : ""}`} width="10" height="10" viewBox="0 0 10 10" fill="none">
                            <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </button>
                        {modelMenuOpen && (
                          <div className="model-menu">
                            {agentModels.map(m => (
                              <button
                                key={m.id}
                                className={`model-menu-item${m.id === selectedModelId ? " active" : ""}${!m.available ? " unavailable" : ""}`}
                                onClick={() => { setSelectedModelId(m.id); setModelMenuOpen(false); }}
                                disabled={!m.available}
                                title={!m.available ? `Requires ${m.provider} API key` : undefined}
                              >
                                <div className="model-menu-row">
                                  <span className="model-menu-name">{m.name}</span>
                                  <span className={`model-speed-badge model-speed-${m.speed}`}>{m.speed_label}</span>
                                  {m.id === selectedModelId && (
                                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{marginLeft:"auto",flexShrink:0}}>
                                      <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                                    </svg>
                                  )}
                                </div>
                                <div className="model-menu-note">{m.note}</div>
                                {!m.available && <div className="model-menu-unavail">API key not configured</div>}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

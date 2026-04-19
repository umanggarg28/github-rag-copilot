import { useState, useEffect, useRef, useCallback } from "react";
import posthog from "posthog-js";
import Sidebar from "./components/Sidebar";
import Message from "./components/Message";
import DiagramView from "./components/DiagramView";
import ReadmeView from "./components/ReadmeView";
import CustomCursor from "./components/CustomCursor";
import LandingHero from "./components/LandingHero";
import { fetchRepos, streamQuery, streamAgentQuery, fetchMcpStatus, fetchMcpPrompt, fetchAgentModels } from "./api";

// ── Suggestion card icons ────────────────────────────────────────────────────
// Simple 16×16 line-art SVGs for each suggestion category.
// Kept inline so there's no icon-library dependency.
// Clean Octicons-inspired icons — 16×16 filled/stroked, consistent 1.5px stroke
const ICONS = {
  // Suggestion card icons
  architecture: <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8.75 3.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM3.75 11.25a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM13.75 11.25a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM8 5.5a.5.5 0 0 0-.5.5v1.5H5.06A2.25 2.25 0 1 0 5 9.25h.06l.44.44V11a2.25 2.25 0 1 0 1.5.04V9.69l.44-.44H11a2.25 2.25 0 1 0-.06-1.5H8.5V6a.5.5 0 0 0-.5-.5Z"/></svg>,
  entry:        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1.75 1h6.5c.966 0 1.75.784 1.75 1.75v2.5a.75.75 0 0 1-1.5 0v-2.5a.25.25 0 0 0-.25-.25h-6.5a.25.25 0 0 0-.25.25v11.5c0 .138.112.25.25.25h6.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 8.25 15h-6.5A1.75 1.75 0 0 1 0 13.25V2.75C0 1.784.784 1 1.75 1Zm9.42 7.75-3.22 3.22a.75.75 0 1 1-1.06-1.06l1.97-1.97H3.75a.75.75 0 0 1 0-1.5h5.11L6.89 5.47a.75.75 0 1 1 1.06-1.06l3.22 3.22a.75.75 0 0 1 0 1.06Z"/></svg>,
  classes:      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M14.064 0h.186C15.216 0 16 .784 16 1.75v.186a8.752 8.752 0 0 1-2.564 6.186l-.458.459c-.314.314-.641.616-.979.904v3.207l-2.209 3.322A.75.75 0 0 1 9 15.75v-4.055c-.338-.288-.665-.59-.979-.904l-.458-.459A8.752 8.752 0 0 1 5 4.136V3.75A.75.75 0 0 1 5.75 3H9.5l-1.75 2h3.5l-1 2h2.25l1-4h.064ZM4.751 7.5H1a.75.75 0 0 0 0 1.5h2.37L4.75 7.5Zm.375 2.5L3.75 12.5H7l1.376-2.5H5.126Z"/></svg>,
  flow:         <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M11.28 12.78a.75.75 0 0 1-1.06-1.06l1.72-1.72H6.75a3.25 3.25 0 0 1-3.25-3.25v-3a.75.75 0 0 1 1.5 0v3a1.75 1.75 0 0 0 1.75 1.75h5.19l-1.72-1.72a.75.75 0 1 1 1.06-1.06l3 3a.75.75 0 0 1 0 1.06l-3 3Z"/></svg>,
  functions:    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M4.72.22a.75.75 0 0 1 1.06 0L9.53 4 8.47 5.06 5.25 1.84 3.28 3.81l1.5 1.5a.75.75 0 0 1-1.06 1.06L2.22 4.87a.75.75 0 0 1 0-1.06L4.72.22ZM11.28 11.78a.75.75 0 0 1-1.06 0L6.47 8 7.53 6.94l3.22 3.22 1.97-1.97-1.5-1.5a.75.75 0 1 1 1.06-1.06l1.5 1.5a.75.75 0 0 1 0 1.06l-2.5 2.59ZM1.5 8.75h.69l.5-2H2a.75.75 0 0 1 0-1.5h1.19l.41-1.66a.75.75 0 1 1 1.46.36l-.33 1.3H6l.41-1.66a.75.75 0 1 1 1.46.36L7.5 5.25h.75a.75.75 0 0 1 0 1.5H7.19l-.5 2H8a.75.75 0 0 1 0 1.5H6.31l-.41 1.66a.75.75 0 1 1-1.46-.36l.33-1.3H3.5l-.41 1.66a.75.75 0 0 1-1.46-.36L2 8.75H1.5a.75.75 0 0 1 0-1.5h-.5Zm2.5 0h1.5l.5-2h-1.5l-.5 2Z"/></svg>,
  diagram:      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.75C1 1.784 1.784 1 2.75 1h10.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0 1 13.25 12H9.06l1.22 1.22a.75.75 0 0 1-1.06 1.06L7.75 12.81l-1.47 1.47a.75.75 0 0 1-1.06-1.06L6.44 12H2.75A1.75 1.75 0 0 1 1 10.25v-7.5Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25H2.75Z"/></svg>,
  shield:       <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M7.467.133a1.748 1.748 0 0 1 1.066 0l5.25 1.68A1.75 1.75 0 0 1 15 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.697 1.697 0 0 1-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 0 1 1.217-1.667Zm.61 1.429a.25.25 0 0 0-.153 0l-5.25 1.68a.25.25 0 0 0-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.196.196 0 0 0 .154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.25.25 0 0 0-.174-.237l-5.25-1.68ZM11.28 6.28l-3.5 3.5a.75.75 0 0 1-1.06 0l-1.5-1.5a.75.75 0 0 1 1.06-1.06l.97.97 2.97-2.97a.75.75 0 0 1 1.06 1.06Z"/></svg>,
  package:      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8.878.392a1.75 1.75 0 0 0-1.756 0l-5.25 3.045A1.75 1.75 0 0 0 1 4.951v6.098c0 .624.332 1.2.872 1.514l5.25 3.045a1.75 1.75 0 0 0 1.756 0l5.25-3.045c.54-.313.872-.89.872-1.514V4.951c0-.624-.332-1.2-.872-1.514Zm-.438 1.297a.25.25 0 0 1 .25 0l2.688 1.559-4.003 2.32-2.929-1.71Zm.31 4.171v5.058l-4.25-2.464V5.745Zm1.5 5.058V5.745l4.25-2.464v5.07Z"/></svg>,
  compare:      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M9.78 12.78a.75.75 0 0 1-1.06-1.06l1.97-1.97H5.75a3.25 3.25 0 0 1-3.25-3.25v-3a.75.75 0 0 1 1.5 0v3a1.75 1.75 0 0 0 1.75 1.75h4.94l-1.97-1.97a.75.75 0 1 1 1.06-1.06l3.25 3.25a.75.75 0 0 1 0 1.06l-3.25 3.25Z"/></svg>,
  complexity:   <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.75.75 0 0 1-1.06-1.06l3.25-3.25a.75.75 0 0 1 1.06 0L9.97 7.94l4.72-4.72a.75.75 0 1 1 1.06 1.06Z"/></svg>,
  config:       <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8.2 8.2 0 0 1 .701.031C9.444.095 9.99.645 10.16 1.29l.288 1.107c.018.066.079.158.212.224.231.114.454.243.668.386.123.082.233.09.299.071l1.103-.303c.644-.176 1.392.021 1.82.63.27.385.506.792.704 1.218.315.675.111 1.422-.364 1.891l-.814.806c-.049.048-.098.147-.088.294.016.257.016.515 0 .772-.01.147.038.246.088.294l.814.806c.475.469.679 1.216.364 1.891a7.977 7.977 0 0 1-.704 1.217c-.428.61-1.176.807-1.82.63l-1.103-.303c-.066-.019-.176-.011-.299.071a5.909 5.909 0 0 1-.668.386c-.133.066-.194.158-.211.224l-.29 1.106c-.168.646-.715 1.196-1.458 1.26a8.006 8.006 0 0 1-1.402 0c-.743-.064-1.289-.614-1.458-1.26l-.289-1.106c-.018-.066-.079-.158-.212-.224a5.738 5.738 0 0 1-.668-.386c-.123-.082-.233-.09-.299-.071l-1.103.303c-.644.176-1.392-.021-1.82-.63a8.12 8.12 0 0 1-.704-1.218c-.315-.675-.111-1.422.363-1.891l.815-.806c.05-.048.098-.147.088-.294a6.214 6.214 0 0 1 0-.772c.01-.147-.038-.246-.088-.294l-.815-.806C.635 6.045.431 5.298.746 4.623a7.92 7.92 0 0 1 .704-1.217c.428-.61 1.176-.807 1.82-.63l1.103.303c.066.019.176.011.299-.071.214-.143.437-.272.668-.386.133-.066.194-.158.211-.224l.29-1.106C6.009.645 6.556.095 7.299.03 7.531.01 7.764 0 8 0Zm-.571 1.525c-.036.003-.108.036-.137.146l-.289 1.105c-.147.561-.549.967-.998 1.189-.173.086-.34.183-.5.29-.417.278-.97.423-1.529.27l-1.103-.303c-.109-.03-.175.016-.195.045-.22.312-.412.644-.573.99-.014.031-.021.11.059.19l.815.806c.411.406.562.957.53 1.456a4.709 4.709 0 0 0 0 .582c.032.499-.119 1.05-.53 1.456l-.815.806c-.081.08-.073.159-.059.19.162.346.353.677.573.989.02.03.085.076.195.046l1.102-.303c.56-.153 1.113-.008 1.53.27.161.107.328.204.501.29.447.222.85.629.997 1.189l.289 1.105c.029.109.101.143.137.146a6.6 6.6 0 0 0 1.142 0c.036-.003.108-.036.137-.146l.289-1.105c.147-.561.549-.967.998-1.189.173-.086.34-.183.5-.29.417-.278.97-.423 1.529-.27l1.103.303c.109.029.175-.016.195-.045.22-.313.411-.644.573-.99.014-.031.021-.11-.059-.19l-.815-.806c-.411-.406-.562-.957-.53-1.456a4.709 4.709 0 0 0 0-.582c-.032-.499.119-1.05.53-1.456l.815-.806c.081-.08.073-.159.059-.19a6.464 6.464 0 0 0-.573-.989c-.02-.03-.085-.076-.195-.046l-1.102.303c-.56.153-1.113.008-1.53-.27a4.44 4.44 0 0 0-.501-.29c-.447-.222-.85-.629-.997-1.189l-.289-1.105c-.029-.11-.101-.143-.137-.146a6.6 6.6 0 0 0-1.142 0ZM11 8a3 3 0 1 1-6 0 3 3 0 0 1 6 0ZM9.5 8a1.5 1.5 0 1 0-3.001.001A1.5 1.5 0 0 0 9.5 8Z"/></svg>,
  pattern:      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.75C1 1.784 1.784 1 2.75 1h1.5c.966 0 1.75.784 1.75 1.75v1.5A1.75 1.75 0 0 1 4.25 6h-1.5A1.75 1.75 0 0 1 1 4.25Zm1.75-.25a.25.25 0 0 0-.25.25v1.5c0 .138.112.25.25.25h1.5a.25.25 0 0 0 .25-.25v-1.5a.25.25 0 0 0-.25-.25ZM1 11.75C1 10.784 1.784 10 2.75 10h1.5c.966 0 1.75.784 1.75 1.75v1.5A1.75 1.75 0 0 1 4.25 15h-1.5A1.75 1.75 0 0 1 1 13.25Zm1.75-.25a.25.25 0 0 0-.25.25v1.5c0 .138.112.25.25.25h1.5a.25.25 0 0 0 .25-.25v-1.5a.25.25 0 0 0-.25-.25Zm5.5-9.5C8.25 1 9.034 1.784 9.034 2.75v1.5a1.75 1.75 0 0 1-1.75 1.75h-1.5A1.75 1.75 0 0 1 4.034 4.25v-1.5C4.034 1.784 4.818 1 5.784 1h1.5v.25ZM8.25 1h1.5c.966 0 1.75.784 1.75 1.75v1.5A1.75 1.75 0 0 1 9.75 6h-1.5A1.75 1.75 0 0 1 6.5 4.25v-1.5C6.5 1.784 7.284 1 8.25 1Zm.25 1.5a.25.25 0 0 0-.25.25v1.5c0 .138.112.25.25.25h1.5a.25.25 0 0 0 .25-.25v-1.5a.25.25 0 0 0-.25-.25Zm3.25-.75c0-.966.784-1.75 1.75-1.75h1.5c.966 0 1.75.784 1.75 1.75v1.5A1.75 1.75 0 0 1 15 6h-1.5a1.75 1.75 0 0 1-1.75-1.75Zm1.75-.25a.25.25 0 0 0-.25.25v1.5c0 .138.112.25.25.25H15a.25.25 0 0 0 .25-.25v-1.5A.25.25 0 0 0 15 1.5Zm-3.25 9.5a1.75 1.75 0 0 0-1.75 1.75v1.5c0 .966.784 1.75 1.75 1.75h1.5A1.75 1.75 0 0 0 15 13.25v-1.5A1.75 1.75 0 0 0 13.5 10Zm-.25 1.75a.25.25 0 0 1 .25-.25h1.5a.25.25 0 0 1 .25.25v1.5a.25.25 0 0 1-.25.25h-1.5a.25.25 0 0 1-.25-.25Z"/></svg>,
  // Onboarding step icons
  github:  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>,
  chat:    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.75C1 1.784 1.784 1 2.75 1h10.5c.966 0 1.75.784 1.75 1.75v7.5A1.75 1.75 0 0 1 13.25 12H9.06l-2.573 2.573A1.457 1.457 0 0 1 4 13.543V12H2.75A1.75 1.75 0 0 1 1 10.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h4.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"/></svg>,
  explore: <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm3.94-1.5 2.656.886 1.316-1.316a.75.75 0 0 1 1.09 1.03l-.03.03-1.316 1.316.887 2.657a.75.75 0 0 1-.975.975l-4-2a.75.75 0 0 1-.31-.31l-2-4a.75.75 0 0 1 .975-.975l4 2Zm.16 2.11L4.5 9.5l1.89.63-.69-2.07Zm1.74.5.63 1.89 1.07-1.07-.63-1.89-.07.07-1 1Z"/></svg>,
};

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
  const [showReadme, setShowReadme]   = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem('ghrc_sidebarCollapsed') === 'true'
  );
  // Landing forces the sidebar collapsed so the hero gets the whole viewport,
  // but the user can still click the arrow to expand it. When they do, we
  // record that override here so our landing-default doesn't fight them.
  // null = follow the landing default; boolean = user took control.
  const [landingSidebarOverride, setLandingSidebarOverride] = useState(null);
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
    setFocusFiles(null);
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
    setShowReadme(false);
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
      setFocusFiles(null);
    }
  }

  function handleRenameSession(sessionId, newTitle) {
    const prev = readSessions(activeRepo);
    const next = prev.map(s => s.id === sessionId ? { ...s, title: newTitle } : s);
    writeSessions(activeRepo, next);
    setSessions(next);
  }

  function toggleSidebarCollapse() {
    // On landing, toggle only flips the ephemeral override so the user's
    // persisted preference isn't changed by playing with the landing arrow.
    // Off landing, toggle the real preference (persisted).
    if (isLanding) {
      // Effective collapsed state when the override is null is "true" (landing default);
      // flipping from that perspective means override becomes false (expand).
      const effective = landingSidebarOverride ?? true;
      setLandingSidebarOverride(!effective);
      return;
    }
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
              // Find the first (oldest) unfilled slot for this tool — results arrive
              // in the same order as calls were emitted, so FIFO matching is correct.
              // Scanning backwards was wrong: parallel same-name calls got swapped.
              for (let i = 0; i < calls.length; i++) {
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
    setFocusFiles(null);
    setStreaming(false);
  }

  // "/" triggers MCP prompt autocomplete — surface this in the placeholder so
  // users discover it without reading docs.
  const placeholder = activeRepo
    ? `Ask about ${activeRepo}… (type / for AI prompts)`
    : "Ask about any indexed repo…";

  // Landing mode = a fresh user with nowhere else to be. We dedicate the
  // whole viewport to the hero in this state — sidebar collapses to an icon
  // strip, the chat input hides, and the landing layout takes over. "All
  // repos" selected with zero indexed counts as landing too (nothing to do).
  const isLanding =
    !showReadme &&
    view === "chat" &&
    messages.length === 0 &&
    (!activeRepo || (activeRepo === "all" && repos.length === 0));

  // Effective collapsed state. On landing, the sidebar defaults to collapsed
  // (the hero wants the room) but the user can still pop it open with the
  // expand arrow — that flips `landingSidebarOverride`. Off landing, use the
  // persisted preference.
  const effectiveCollapsed = isLanding
    ? (landingSidebarOverride ?? true)
    : sidebarCollapsed;

  // Reset the landing override the moment we leave landing, so the user's
  // persisted sidebar preference takes over cleanly on the next visit.
  useEffect(() => {
    if (!isLanding && landingSidebarOverride !== null) {
      setLandingSidebarOverride(null);
    }
  }, [isLanding, landingSidebarOverride]);

  // Landing → Sidebar bridge helpers. Tiles and the hero URL input both
  // route through the same "external ingest" event so the Sidebar's existing
  // ingestion flow (progress stream + success handling) owns the UX.
  function handleLandingPick(slug) {
    posthog.capture("landing_tile_clicked", { slug });
    const indexed = repos.find(r => r.slug === slug);
    if (indexed) {
      setActiveRepo(slug);
      setShowReadme(false);
      return;
    }
    window.dispatchEvent(new CustomEvent("cartographer:ingest", {
      detail: { repo: `github.com/${slug}` },
    }));
  }
  function handleLandingUrl(raw) {
    posthog.capture("landing_url_submitted", { input: raw });
    const clean = raw.includes("/") ? raw : null;
    if (!clean) return;
    const withHost = clean.startsWith("github.com/") || clean.startsWith("http")
      ? clean
      : `github.com/${clean}`;
    window.dispatchEvent(new CustomEvent("cartographer:ingest", {
      detail: { repo: withHost },
    }));
  }

  return (
    <div className={`layout${effectiveCollapsed ? " layout-collapsed" : ""}${isLanding ? " layout-landing" : ""}`}>
      <CustomCursor />

      {/* Sidebar overlay for mobile — closes sidebar when clicking outside */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} aria-hidden="true" />
      )}

      <Sidebar
        repos={repos}
        reposLoading={reposLoading}
        activeRepo={activeRepo}
        onSelectRepo={(repo) => { setActiveRepo(repo); setShowReadme(false); posthog.capture("repo_selected", { repo }); }}
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
        collapsed={effectiveCollapsed}
        onToggleCollapse={toggleSidebarCollapse}
        onGenerateReadme={(repo) => { setActiveRepo(repo); setShowReadme(true); posthog.capture("readme_opened", { repo }); }}
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
                  className={`view-btn ${view === "chat" && !showReadme ? "active" : ""}`}
                  onClick={() => { setView("chat"); setShowReadme(false); }}
                >Chat</button>
                <button
                  className={`view-btn ${view === "graph" && !showReadme ? "active" : ""}`}
                  onClick={() => { setView("graph"); setShowReadme(false); posthog.capture("diagram_view_opened", { repo: activeRepo }); }}
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

        {/* ── README view ── */}
        {showReadme && activeRepo && activeRepo !== "all" && (
          <ReadmeView
            repo={activeRepo}
            contextualAt={repos.find(r => r.slug === activeRepo)?.contextual_at ?? null}
            onClose={() => setShowReadme(false)}
          />
        )}

        {/* ── Diagram view ── */}
        {/* Keyed to activeRepo so switching repos (or from chat→diagram) replays
            .view-switch-in. Matches the tab transition inside DiagramView and
            the mode transition inside ExploreView. */}
        {!showReadme && view === "graph" && activeRepo && (
          <div key={`diag-${activeRepo}`} className="view-switch-in app-view-host">
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
          </div>
        )}

        {/* ── Chat view ── */}
        {!showReadme && view === "chat" && (
          <>
            {messages.length === 0 ? (
              isLanding ? (
                // Full-bleed landing — LandingHero owns its own layout
                <div className="landing-root view-switch-in">
                  <LandingHero
                    onPickRepo={handleLandingPick}
                    onPasteUrl={handleLandingUrl}
                  />
                </div>
              ) : (
              <div
                className="empty-state has-cursor-glow"
                // Shared --mx/--my channel: one mousemove feeds both the
                // glow pseudo and the constellation parallax. Percentages
                // are used so the transforms are resolution-independent.
                onMouseMove={(e) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  const mx = ((e.clientX - r.left) / r.width) * 100;
                  const my = ((e.clientY - r.top) / r.height) * 100;
                  e.currentTarget.style.setProperty("--mx", `${mx}%`);
                  e.currentTarget.style.setProperty("--my", `${my}%`);
                }}
                style={{ "--glow-size": "640px", "--glow-intensity": "6%" }}
              >
                {activeRepo === "all" && repos.length > 0 ? (
                  // "All repos" explicitly selected with repos indexed — cross-repo query mode
                  <div className="suggest-state constellation-bg">
                    <h2>Ask across all repos</h2>
                    <p>Searching <strong>{repos.length} indexed repos</strong> at once — {repos.map(r => r.slug.split("/")[1]).join(", ")}. Results show which repo each source comes from.</p>
                    <div className="suggestions">
                      {[
                        { icon: "compare", title: "Compare architectures", body: "What patterns do these repos share?" },
                        { icon: "complexity", title: "Complexity analysis", body: "Which repo is most complex and why?" },
                        { icon: "entry", title: "Entry points",       body: "Find all main entry points across repos" },
                        { icon: "config",  title: "Configuration",    body: "How do these repos handle env & config?" },
                        { icon: "pattern", title: "Design patterns",  body: "Common abstractions across all repos" },
                      ].map(({ icon, title, body }, i) => {
                        const q = `${title}: ${body}`;
                        return (
                          <button key={title} className="suggestion-btn"
                            style={{ animationDelay: `${150 + i * 120}ms` }}
                            onClick={() => { setInput(q); textareaRef.current?.focus(); }}>
                            <span className="suggestion-icon">{ICONS[icon]}</span>
                            <span className="suggestion-content">
                              <span className="suggestion-title">{title}</span>
                              <span className="suggestion-body">{body}</span>
                            </span>
                            <svg className="suggestion-arrow" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h10M9 4l4 4-4 4"/></svg>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  // Repo selected — show mode-aware suggestions + feature discovery cards
                  <div className="suggest-state constellation-bg">
                    <h2>How does {activeRepo.split("/")[1]} work?</h2>


                    {agentMode ? (
                      <>
                        <div className="mode-hint" style={{ marginBottom: 12 }}>
                          <strong>Agent mode</strong> — search → observe → reason → search again. Watch the ReAct loop work in real time.
                        </div>
                        <div className="suggestions">
                          {[
                            { icon: "architecture", title: "Map the architecture",    body: `Walk ${activeRepo.split("/")[1]} from entry point to output`,  q: `How is ${activeRepo.split("/")[1]} structured? Trace the main execution path from the entry point all the way to the output.` },
                            { icon: "functions",    title: "Key functions",           body: "Most important functions and how they connect",                  q: `What are the most important functions in ${activeRepo.split("/")[1]} and how do they call each other?` },
                            { icon: "diagram",      title: "Generate a diagram",      body: "Visual map of the main components",                             q: `Generate a diagram of the main components in ${activeRepo.split("/")[1]} and how they relate to each other.` },
                            { icon: "shield",       title: "Error handling",          body: "How edge cases are managed across the codebase",                q: `How does ${activeRepo.split("/")[1]} handle errors and edge cases?` },
                            { icon: "flow",         title: "Data flow",               body: "How data moves from input to final result",                     q: `How does data flow through ${activeRepo.split("/")[1]} from input to final result?` },
                          ].map(({ icon, title, body, q }, i) => {
                            return (
                              <button key={title} className="suggestion-btn"
                                style={{ animationDelay: `${150 + i * 120}ms` }}
                                onClick={() => { setInput(q); textareaRef.current?.focus(); }}>
                                <span className="suggestion-icon">{ICONS[icon]}</span>
                                <span className="suggestion-content">
                                  <span className="suggestion-title">{title}</span>
                                  <span className="suggestion-body">{body}</span>
                                </span>
                                <svg className="suggestion-arrow" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h10M9 4l4 4-4 4"/></svg>
                              </button>
                            );
                          })}
                        </div>
                        <button className="graph-hint-btn" onClick={() => setView("graph")}>
                          Explore Diagrams for {activeRepo.split("/")[1]} →
                        </button>
                      </>
                    ) : (
                      <>
                        <div className="suggestions">
                          {[
                            { icon: "architecture", title: "Overall architecture",  body: `How is ${activeRepo.split("/")[1]} structured?`,    q: `How is ${activeRepo.split("/")[1]} structured overall? What are the main components and how do they fit together?` },
                            { icon: "entry",        title: "Entry points",          body: "Main entry points and how the code flows",           q: `What are the main entry points of ${activeRepo.split("/")[1]} and how does execution flow through them?` },
                            { icon: "classes",      title: "Key classes",           body: "What each major class does",                         q: `What are the key classes in ${activeRepo.split("/")[1]} and what is each one responsible for?` },
                            { icon: "flow",         title: "Data processing",       body: "How data is transformed through the system",         q: `How is data transformed and processed as it flows through ${activeRepo.split("/")[1]}?` },
                            { icon: "package",      title: "Dependencies",          body: "External libraries and how they're used",            q: `What external libraries does ${activeRepo.split("/")[1]} depend on and how does it use them?` },
                          ].map(({ icon, title, body, q }, i) => {
                            return (
                              <button key={title} className="suggestion-btn"
                                style={{ animationDelay: `${150 + i * 120}ms` }}
                                onClick={() => { setInput(q); textareaRef.current?.focus(); }}>
                                <span className="suggestion-icon">{ICONS[icon]}</span>
                                <span className="suggestion-content">
                                  <span className="suggestion-title">{title}</span>
                                  <span className="suggestion-body">{body}</span>
                                </span>
                                <svg className="suggestion-arrow" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h10M9 4l4 4-4 4"/></svg>
                              </button>
                            );
                          })}
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
              )
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

            {/* Input — hidden on landing (there's no repo to chat about yet) */}
            {!isLanding && (
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
            )}
          </>
        )}
      </div>

    </div>
  );
}

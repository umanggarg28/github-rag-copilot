/**
 * DiagramView.jsx — System diagrams + interactive codebase explorer.
 *
 * Tabs:
 *   Explore        — Interactive concept map (AI-generated, clearly labelled)
 *   Architecture   — Real import edges from AST + AI descriptions     [verified]
 *   Class Hierarchy— Real inheritance from AST + AI descriptions      [verified]
 *   Sequence       — AI-generated call flow                           [speculative]
 *   Data Flow      — AI-generated data pipeline                       [speculative]
 *
 * Interactions:
 *   - Click a diagram node  → inline NodeDetailPanel slides in (no view switch)
 *   - Click a diagram edge  → NodeDetailPanel explains the relationship
 *   - Hover a node          → dims unrelated nodes, highlights connections
 */

import { useEffect, useState } from "react";
import { fetchDiagram } from "../api";
import ExploreView from "./ExploreView";
import GraphDiagram from "./GraphDiagram";
import NodeDetailPanel from "./NodeDetailPanel";

// ── Diagram tab definitions ───────────────────────────────────────────────────
const EXPLORE_TAB = {
  id: "explore", label: "Explore", desc: "Guided concept tour", icon: "◈",
};

// Only AST-verified diagrams — Sequence and Data Flow were removed because
// they were fully LLM-generated with no static analysis backing, making them
// unreliable for a learning tool where accuracy matters.
const DIAGRAM_TABS = [
  {
    id:    "architecture",
    label: "Architecture",
    desc:  "Components & connections",
    icon:  "⬡",
  },
  {
    id:    "class",
    label: "Class Hierarchy",
    desc:  "Classes & relationships",
    icon:  "◫",
  },
];

const ALL_TABS = [EXPLORE_TAB, ...DIAGRAM_TABS];

export default function DiagramView({ repo, onAskAbout, focusFiles }) {
  const [diagramType, setDiagramType] = useState(
    () => localStorage.getItem("ghrc_diagramType") || "explore"
  );
  function setType(t) {
    setDiagramType(t);
    localStorage.setItem("ghrc_diagramType", t);
    // Clear any open detail panel when switching tabs
    setSelected(null);
  }

  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [diagramData, setDiagramData] = useState(null);
  const [cache, setCache]             = useState({});
  const [retryKey, setRetryKey]       = useState(0);
  const [fullscreen, setFullscreen]   = useState(false);

  // ── Inline detail panel state ─────────────────────────────────────────────
  // subject = { kind, label, type, file, description, items, autoQuestion? }
  const [selected, setSelected] = useState(null);

  const isExplore = diagramType === "explore";
  const selectedTypeDef = ALL_TABS.find(t => t.id === diagramType);

  // ── Fetch diagram ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!repo || isExplore) return;
    if (retryKey === 0 && cache[diagramType]) {
      setDiagramData(cache[diagramType]);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setDiagramData(null);
    fetchDiagram(repo, diagramType)
      .then(data => {
        if (cancelled) return;
        setLoading(false);
        if (data.error) { setError(data.error); return; }
        setCache(prev => ({ ...prev, [diagramType]: data.diagram }));
        setDiagramData(data.diagram);
        setRetryKey(0);
      })
      .catch(err => {
        if (cancelled) return;
        setLoading(false);
        setError(err.message);
      });
    return () => { cancelled = true; };
  }, [repo, diagramType, retryKey, isExplore]);

  // Reset on repo change — always land on Explore so the concept map is the
  // first thing seen when switching repos.
  useEffect(() => {
    setCache({});
    setDiagramData(null);
    setError(null);
    setRetryKey(0);
    setSelected(null);
    setDiagramType("explore");
    localStorage.setItem("ghrc_diagramType", "explore");
  }, [repo]);

  // Escape exits fullscreen
  useEffect(() => {
    if (!fullscreen) return;
    function onKey(e) { if (e.key === "Escape") setFullscreen(false); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  function handleRegenerate() {
    setCache(prev => { const n = {...prev}; delete n[diagramType]; return n; });
    setDiagramData(null);
    setError(null);
    setSelected(null);
    setRetryKey(k => k + 1);
  }

  // When user clicks "Open in full chat →" from the detail panel
  function handleOpenInChat(question) {
    setSelected(null);
    onAskAbout?.(question);
  }

  return (
    <div className={`diagram-container${fullscreen ? " diagram-fullscreen" : ""}`}>

      {/* ── Header ── */}
      <div className="diagram-header">
        {!fullscreen && (
          <span className="diagram-title">
            {isExplore ? `Explore — ${repo}` : `System Diagram — ${repo}`}
          </span>
        )}
        <button
          className="diagram-fullscreen-btn"
          onClick={() => setFullscreen(f => !f)}
          title={fullscreen ? "Exit fullscreen" : "Fullscreen"}
          aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
          style={{ marginLeft: "auto" }}
        >
          {fullscreen ? (
            <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor">
              <path d="M5.5 0a.5.5 0 0 1 .5.5v4A1.5 1.5 0 0 1 4.5 6h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5m5 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 10 4.5v-4a.5.5 0 0 1 .5-.5M0 10.5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 6 11.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5m10 1a1.5 1.5 0 0 1 1.5-1.5h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0z"/>
            </svg>
          ) : (
            <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor">
              <path d="M1.5 1h4a.5.5 0 0 1 0 1h-4a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4A1.5 1.5 0 0 1 1.5 1m9 0h4A1.5 1.5 0 0 1 16 2.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1 0-1m-9 9a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 0 13.5v-4a.5.5 0 0 1 .5-.5m15 0a.5.5 0 0 1 .5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5"/>
            </svg>
          )}
        </button>
        {diagramData && !isExplore && (
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button className="diagram-ask-btn" onClick={handleRegenerate} title="Generate a fresh diagram">
              ↺ Regenerate
            </button>
            <button
              className="diagram-ask-btn"
              onClick={() => onAskAbout?.(
                `Explain the ${selectedTypeDef?.label.toLowerCase()} of ${repo} — walk me through each component and how they connect`
              )}
            >
              Ask about this →
            </button>
          </div>
        )}
      </div>

      {/* ── Focus Files banner ── */}
      {!fullscreen && focusFiles?.length > 0 && (
        <div className="diagram-focus-banner">
          <span className="diagram-focus-icon">◎</span>
          <span>Focused on {focusFiles.length} file{focusFiles.length !== 1 ? "s" : ""} from your last answer:</span>
          <span className="diagram-focus-files">
            {focusFiles.slice(0, 3).join(", ")}
            {focusFiles.length > 3 ? ` +${focusFiles.length - 3} more` : ""}
          </span>
        </div>
      )}

      {/* ── Tab selector ── */}
      {!fullscreen && (
        <div className="diagram-type-bar">
          {/* ── Explore — hero tab, visually distinct from technical diagrams ── */}
          <button
            key="explore"
            className={`diagram-type-btn${diagramType === "explore" ? " active" : ""}`}
            onClick={() => setType("explore")}
            style={{
              background: diagramType === "explore"
                ? "rgba(124,58,237,0.18)"
                : "rgba(124,58,237,0.07)",
              borderColor: diagramType === "explore"
                ? "rgba(124,58,237,0.55)"
                : "rgba(124,58,237,0.25)",
            }}
          >
            <span className="diagram-type-icon" style={{ color: "var(--accent-soft)" }}>◈</span>
            <span className="diagram-type-label" style={{ color: diagramType === "explore" ? "var(--accent)" : "var(--accent-soft)" }}>Explore</span>
            <span className="diagram-type-desc">Guided concept tour</span>
          </button>

          {/* Vertical divider between Explore and technical tabs */}
          <div style={{
            width: 1, alignSelf: "stretch",
            background: "var(--border)",
            margin: "0 4px", flexShrink: 0,
          }} />

          {/* ── Technical diagram tabs ── */}
          {DIAGRAM_TABS.map(t => (
            <button
              key={t.id}
              className={`diagram-type-btn${diagramType === t.id ? " active" : ""}`}
              onClick={() => setType(t.id)}
            >
              <span className="diagram-type-icon">{t.icon}</span>
              <span className="diagram-type-label">{t.label}</span>
              <span className="diagram-type-desc">{t.desc}</span>
            </button>
          ))}
        </div>
      )}

      {/* ── Canvas area (diagram on top, detail tray below) ── */}
      <div className="diagram-canvas" style={{ display: "flex", flexDirection: "column", overflow: "hidden", alignItems: "stretch", minHeight: 0 }}>
        <div style={{ flex: 1, position: "relative", overflow: "hidden", minHeight: 0 }}>
          {isExplore ? (
            <ExploreView repo={repo} onAskAbout={onAskAbout} />
          ) : (
            <>
              {loading && (
                <div className="diagram-loading">
                  <span className="spinner" />
                  Analysing codebase and generating {selectedTypeDef?.label.toLowerCase()} diagram…
                </div>
              )}
              {error && (
                <div className="diagram-error">
                  {error}
                  <button className="diagram-retry-btn" onClick={handleRegenerate}>Retry</button>
                </div>
              )}
              {diagramData && (
                <GraphDiagram
                  data={diagramData}
                  repo={repo}
                  diagramType={diagramType}
                  onNodeSelect={setSelected}
                  onEdgeSelect={setSelected}
                  onAskAbout={onAskAbout}
                  panelOpen={!!selected}
                />
              )}
            </>
          )}
        </div>

        {/* Bottom tray detail panel */}
        {selected && (
          <NodeDetailPanel
            subject={selected}
            repo={repo}
            onClose={() => setSelected(null)}
            onOpenInChat={handleOpenInChat}
          />
        )}
      </div>
    </div>
  );
}

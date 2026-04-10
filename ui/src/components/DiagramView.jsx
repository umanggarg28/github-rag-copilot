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

import { useEffect, useRef, useState } from "react";
import { streamDiagram } from "../api";
import ExploreView from "./ExploreView";
import GraphDiagram from "./GraphDiagram";
import NodeDetailPanel from "./NodeDetailPanel";

// ── Diagram tab definitions ───────────────────────────────────────────────────
const EXPLORE_TAB = {
  id: "explore", label: "Explore", desc: "Guided concept tour",
};

// Only AST-verified diagrams — Sequence and Data Flow were removed because
// they were fully LLM-generated with no static analysis backing, making them
// unreliable for a learning tool where accuracy matters.
const DIAGRAM_TABS = [
  {
    id:    "architecture",
    label: "Architecture",
    desc:  "Components & connections",
  },
  {
    id:    "class",
    label: "Class Hierarchy",
    desc:  "Classes & relationships",
  },
];

// ── Tab icons (SVG) ───────────────────────────────────────────────────────────
// Inline SVGs render crisp at every DPI — unicode glyphs (◈ ⬡ ◫) are
// rasterised at screen resolution and look blurry on high-DPI displays.
function TabIcon({ id }) {
  if (id === "explore") return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      {/* Compass rose — N point pulses in the landing animation */}
      <path d="M12 2L14.5 7.5L12 12L9.5 7.5Z" fill="currentColor"/>
      <path d="M12 22L13.5 16.5L12 12L10.5 16.5Z" fill="currentColor" opacity="0.45"/>
      <path d="M22 12L16.5 10.5L12 12L16.5 13.5Z" fill="currentColor" opacity="0.45"/>
      <path d="M2 12L7.5 10.5L12 12L7.5 13.5Z" fill="currentColor" opacity="0.45"/>
      <circle cx="12" cy="12" r="1.5" fill="currentColor"/>
    </svg>
  );
  if (id === "architecture") return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" aria-hidden="true">
      {/* Three nodes connected by edges — represents a dependency/import graph */}
      <circle cx="5" cy="12" r="2.5"/>
      <circle cx="19" cy="6.5" r="2.5"/>
      <circle cx="19" cy="17.5" r="2.5"/>
      <line x1="7.4" y1="11" x2="16.6" y2="7.5"/>
      <line x1="7.4" y1="13" x2="16.6" y2="16.5"/>
    </svg>
  );
  if (id === "class") return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {/* Parent node + two children — represents inheritance / class tree */}
      <rect x="8.5" y="2" width="7" height="5" rx="1.5"/>
      <rect x="1" y="17" width="7" height="5" rx="1.5"/>
      <rect x="16" y="17" width="7" height="5" rx="1.5"/>
      <line x1="12" y1="7" x2="12" y2="12"/>
      <line x1="12" y1="12" x2="4.5" y2="17"/>
      <line x1="12" y1="12" x2="19.5" y2="17"/>
    </svg>
  );
  return null;
}

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
  const [loadStage, setLoadStage]     = useState(null);  // { stage, progress, message }
  const [error, setError]             = useState(null);
  const [diagramData, setDiagramData] = useState(null);
  const [cache, setCache]             = useState({});
  // Per-type retry counter — using a single shared counter caused switching tabs
  // while retryKey > 0 to bypass the cache for the OTHER diagram type too.
  const [retryKeys, setRetryKeys]     = useState({});  // { "architecture": 1, "class": 0, ... }
  const [fullscreen, setFullscreen]   = useState(false);

  // ── Inline detail panel state ─────────────────────────────────────────────
  // subject = { kind, label, type, file, description, items, autoQuestion? }
  const [selected, setSelected] = useState(null);

  // Ref passed into ExploreView so we can call its force-reload from our header
  const exploreRegenRef = useRef(null);

  const isExplore = diagramType === "explore";
  const selectedTypeDef = ALL_TABS.find(t => t.id === diagramType);

  // localStorage key for a diagram — used to persist across page refreshes.
  function diagLsKey(r, type) { return `ghrc_diagram_${r.replace(/\//g, "_")}_${type}`; }

  // ── Fetch diagram ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!repo || isExplore) return;
    const isForced = !!retryKeys[diagramType];
    // 1. In-memory cache: survives tab switches.
    if (!isForced && cache[diagramType]) {
      setDiagramData(cache[diagramType]);
      setError(null);
      return;
    }
    // 2. localStorage cache: survives page refreshes.
    if (!isForced) {
      try {
        const stored = localStorage.getItem(diagLsKey(repo, diagramType));
        if (stored) {
          const parsed = JSON.parse(stored);
          setCache(prev => ({ ...prev, [diagramType]: parsed }));
          setDiagramData(parsed);
          setError(null);
          return;
        }
      } catch { /* corrupt — fall through to fetch */ }
    }
    setLoading(true);
    setLoadStage(null);
    setError(null);
    setDiagramData(null);
    // force=true when retryKeys[diagramType] > 0 (user hit Regenerate) so the
    // backend bypasses its disk cache and actually produces a fresh diagram.
    const cancel = streamDiagram(repo, diagramType, {
      force: isForced,
      onProgress: (ev) => setLoadStage(ev),
      onDone: ({ diagram, type }) => {
        setLoading(false);
        setLoadStage(null);
        setCache(prev => ({ ...prev, [diagramType]: diagram }));
        setDiagramData(diagram);
        setRetryKeys(prev => ({ ...prev, [diagramType]: 0 }));
        try { localStorage.setItem(diagLsKey(repo, diagramType), JSON.stringify(diagram)); } catch {}
      },
      onError: (msg) => {
        setLoading(false);
        setLoadStage(null);
        setError(msg);
      },
    });
    return cancel;
  }, [repo, diagramType, retryKeys[diagramType], isExplore]);

  // Reset on repo change — always land on Explore so the concept map is the
  // first thing seen when switching repos.
  useEffect(() => {
    setCache({});
    setDiagramData(null);
    setError(null);
    setRetryKeys({});
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
    try { localStorage.removeItem(diagLsKey(repo, diagramType)); } catch {}
    setCache(prev => { const n = {...prev}; delete n[diagramType]; return n; });
    setDiagramData(null);
    setError(null);
    setSelected(null);
    setRetryKeys(prev => ({ ...prev, [diagramType]: (prev[diagramType] || 0) + 1 }));
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
        {/* Right-side controls — action buttons then fullscreen on the far right */}
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginLeft: "auto" }}>
          {isExplore ? (
            <button className="diagram-retry-btn" onClick={() => exploreRegenRef.current?.()} title="Generate a fresh tour">
              ↺ Regenerate
            </button>
          ) : diagramData && (
            <>
              <button className="diagram-retry-btn" onClick={handleRegenerate} title="Generate a fresh diagram">
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
            </>
          )}
          <button
            className="diagram-fullscreen-btn"
            onClick={() => setFullscreen(f => !f)}
            title={fullscreen ? "Exit fullscreen" : "Fullscreen"}
            aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
          >
            {fullscreen ? (
              /* Compress — 4 inward corners */
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" style={{ display: "block" }}>
                <path d="M6 1v5H1"/><path d="M10 1v5h5"/>
                <path d="M6 15v-5H1"/><path d="M10 15v-5h5"/>
              </svg>
            ) : (
              /* Expand — 4 outward corners */
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" style={{ display: "block" }}>
                <path d="M1 6V1h5"/><path d="M15 6V1h-5"/>
                <path d="M1 10v5h5"/><path d="M15 10v5h-5"/>
              </svg>
            )}
          </button>
        </div>
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
                ? "rgba(212,132,90,0.18)"
                : "rgba(212,132,90,0.07)",
              borderColor: diagramType === "explore"
                ? "rgba(212,132,90,0.55)"
                : "rgba(212,132,90,0.25)",
            }}
          >
            <span className="diagram-type-icon" style={{ color: "var(--accent-soft)" }}><TabIcon id="explore" /></span>
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
              <span className="diagram-type-icon"><TabIcon id={t.id} /></span>
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
            <ExploreView repo={repo} onAskAbout={onAskAbout} onRegenerateRef={exploreRegenRef} />
          ) : (
            <>
              {loading && (
                <div className="diagram-loading">
                  <span className="spinner" />
                  <div style={{ flex: 1, maxWidth: 300 }}>
                    <div>{loadStage?.message || `Generating ${selectedTypeDef?.label.toLowerCase()} diagram…`}</div>
                    {loadStage && (
                      <div style={{ marginTop: 8 }}>
                        <div style={{
                          height: 3, background: "var(--border)", borderRadius: 2, overflow: "hidden",
                        }}>
                          <div style={{
                            height: "100%",
                            width:  `${Math.round((loadStage.progress || 0) * 100)}%`,
                            background:  "var(--accent)",
                            borderRadius: 2,
                            transition:  "width 0.4s ease",
                          }} />
                        </div>
                        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 3 }}>
                          {Math.round((loadStage.progress || 0) * 100)}%
                        </div>
                      </div>
                    )}
                  </div>
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

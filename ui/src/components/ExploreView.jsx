/**
 * ExploreView.jsx — Interactive Codebase Tour.
 *
 * ═══════════════════════════════════════════════════════════════
 * WHAT THIS SHOWS
 * ═══════════════════════════════════════════════════════════════
 *
 * Instead of a raw scatter plot of files, this view teaches a student
 * HOW to approach a new codebase. The LLM generates 6-8 key concepts —
 * the ideas a student must understand — and their dependencies, then
 * renders them as an interactive node diagram:
 *
 *   [Value class] → [Forward Pass] → [Backward Pass] → [Loss + SGD]
 *                ↘ [MLP layer]   ↗
 *
 * Each node = a card you can click to expand. Arrows mean "you need
 * to understand X before Y". Reading order is encoded as numbered badges.
 *
 * ═══════════════════════════════════════════════════════════════
 * LAYOUT ALGORITHM
 * ═══════════════════════════════════════════════════════════════
 *
 * 1. Topological sort: assign each concept a column depth
 *    (longest dependency chain from a root node).
 * 2. Within each column, sort by reading_order.
 * 3. Center columns vertically relative to the tallest column.
 * 4. Draw bezier arrows between connected cards.
 *
 * ═══════════════════════════════════════════════════════════════
 * INTERACTIONS
 * ═══════════════════════════════════════════════════════════════
 *
 *   Click card  → expand description + key methods
 *   Hover card  → highlight its edges + connected nodes, dim others
 *   Ask button  → pre-fills chat with a targeted question
 *   Scroll      → zoom (non-passive wheel so preventDefault works)
 *   Drag        → pan the canvas
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { streamTour } from "../api";

// Module-level cache — survives tab switches because ExploreView is unmounted
// when the user navigates to Architecture/Class tabs and remounted on return.
// Without this, switching to Explore re-fetches (and re-generates) every time.
// Key: repo slug → tour data object
const tourCache = {};

// localStorage key for a given repo's tour.
// We persist tour data across page refreshes so the backend (and LLM quota)
// is only hit once per repo, not on every refresh.
function tourLsKey(repo) { return `ghrc_tour_${repo.replace(/\//g, "_")}`; }

// ── Type → visual token ───────────────────────────────────────────────────────
// Each concept type maps to a hue from a different part of the spectrum so
// they're immediately legible even at small sizes. Four clearly-distinct hues:
// blue (class), amber (function), violet (module), emerald (algorithm).
const TYPE_STYLE = {
  class:     { border: "#5B8FF9", glow: "rgba(91,143,249,0.38)",  dot: "#7DABFF", tag: "class"  }, // blue   240°
  function:  { border: "#FBBF24", glow: "rgba(251,191,36,0.32)",  dot: "#FCD34D", tag: "fn"     }, // amber   45°
  module:    { border: "#A78BFA", glow: "rgba(167,139,250,0.32)", dot: "#C4B5FD", tag: "module" }, // violet 270°
  algorithm: { border: "#34D399", glow: "rgba(52,211,153,0.32)",  dot: "#6EE7B7", tag: "algo"   }, // emerald 160°
};
const FALLBACK_STYLE = { border: "#4E5E80", glow: "rgba(78,94,128,0.30)", dot: "#8896B8", tag: "?" };

function styleFor(type) {
  return TYPE_STYLE[type] || FALLBACK_STYLE;
}

// ── Card geometry ─────────────────────────────────────────────────────────────
const CARD_W      = 220;  // card width in canvas px
const CARD_H      = 172;  // collapsed card height
const COL_GAP     = 100;  // horizontal gap between cards in the same row
const ROW_GAP     = 72;   // vertical gap between rows
const EXPANSION_H = 195;  // extra px a card grows when expanded (desc + items + padding)

// How many concepts appear in each horizontal row.
// With 12 concepts and PER_ROW=4: 3 rows of 4, reads like a book.
const PER_ROW = 4;

// When a card is expanded, push all cards in rows BELOW it down by EXPANSION_H
// so the expanded content doesn't overlap them. Returns { [id]: yOffset }.
function expansionOffsets(selectedId, concepts, basePositions) {
  if (selectedId === null) return {};
  const sel = basePositions[selectedId];
  if (!sel) return {};
  const offsets = {};
  concepts.forEach(c => {
    if (c.id === selectedId) return;
    const p = basePositions[c.id];
    // Any card in a row strictly below the expanded card's row
    if (p && p.y > sel.y) offsets[c.id] = EXPANSION_H;
  });
  return offsets;
}

// ── Layout: row-major reading order ───────────────────────────────────────────
// Concepts are placed left-to-right by reading_order, wrapping to the next row
// after PER_ROW concepts — exactly like reading text.
//
//   1 → 2 → 3 → 4
//   ↓
//   5 → 6 → 7 → 8
//   ↓
//   9 → 10 → 11 → 12
//
// This avoids the "spreadsheet" feel of column-major layouts where the eye
// must scan down a column then jump back to the top of the next column.
function computeLayout(concepts) {
  if (!concepts.length) return {};

  const sorted = [...concepts].sort((a, b) =>
    (a.reading_order ?? 999) - (b.reading_order ?? 999)
  );

  const positions = {};
  sorted.forEach((c, i) => {
    const row = Math.floor(i / PER_ROW);
    const col = i % PER_ROW;
    positions[c.id] = {
      x: col * (CARD_W + COL_GAP) + 48,
      y: row * (CARD_H + ROW_GAP) + 48,
    };
  });
  return positions;
}

// ── Arrow: cubic bezier between source and target ─────────────────────────────
// Normally left-to-right (right edge → left edge). If the dependency arrow
// goes backwards (prerequisite placed to the right due to reading_order layout),
// flip to exit from the left edge and enter the right edge instead.
function bezierPath(fromPos, toPos) {
  const fromCenterX = fromPos.x + CARD_W / 2;
  const toCenterX   = toPos.x   + CARD_W / 2;
  const leftToRight = toCenterX >= fromCenterX;

  const x1 = leftToRight ? fromPos.x + CARD_W : fromPos.x;
  const y1 = fromPos.y + CARD_H / 2;
  const x2 = leftToRight ? toPos.x             : toPos.x + CARD_W;
  const y2 = toPos.y + CARD_H / 2;

  const tension = Math.max(Math.abs(x2 - x1) * 0.55, 60);
  const dir = leftToRight ? 1 : -1;
  return `M ${x1} ${y1} C ${x1 + dir * tension} ${y1}, ${x2 - dir * tension} ${y2}, ${x2} ${y2}`;
}

// ── ConceptCard ────────────────────────────────────────────────────────────────
function ConceptCard({ concept, visualNum, isEntry, isSelected, isHovered, isDimmed, pos, onSelect, onHover, onAsk, onDragStart, wasDragged }) {
  const s = styleFor(concept.type);

  return (
    <div
      className={`ec-card${isSelected ? " ec-selected" : ""}${isDimmed ? " ec-dimmed" : ""}`}
      style={{
        position: "absolute",
        zIndex: isSelected ? 10 : 1,
        left: pos.x,
        top: pos.y,
        width: CARD_W,
        cursor: "grab",
        borderColor: isSelected ? s.border : isHovered ? s.border : undefined,
        boxShadow: isSelected
          ? `0 0 0 2px ${s.border}, 0 0 20px ${s.glow.replace(/[\d.]+\)$/, '0.60)')}, 0 20px 60px ${s.glow.replace(/[\d.]+\)$/, '0.45)')}`
          : isHovered
          ? `0 0 0 2px ${s.border}, 0 0 20px ${s.glow.replace(/[\d.]+\)$/, '0.60)')}, 0 20px 60px ${s.glow.replace(/[\d.]+\)$/, '0.45)')}`
          : undefined,
      }}
      onMouseDown={(e) => onDragStart?.(e, concept, pos)}
      onClick={() => { if (!wasDragged?.current) onSelect(concept.id); }}
      onMouseEnter={() => onHover(concept.id)}
      onMouseLeave={() => onHover(null)}
    >
      {/* Top row: reading order badge + type tag */}
      <div className="ec-card-top">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="ec-order">{visualNum ?? concept.reading_order}</span>
          {isEntry && <span className="ec-entry-tag">Start here</span>}
        </div>
        <span className="ec-type-tag" style={{ color: s.dot, borderColor: `${s.dot}44` }}>
          {s.tag}
        </span>
      </div>

      {/* Name + subtitle */}
      <div className="ec-name">{concept.name}</div>
      <div className="ec-subtitle">{concept.subtitle}</div>

      {/* File pill */}
      <div className="ec-file">
        <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.5, flexShrink: 0 }}>
          <path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 8.75 4.25V1.5Zm6.75.56v2.19c0 .138.112.25.25.25h2.19Z"/>
        </svg>
        {concept.file}
      </div>

      {/* Expanded: description + key methods */}
      {isSelected && (
        <div className="ec-expanded">
          <p className="ec-desc">{concept.description}</p>
          {concept.key_items?.length > 0 && (
            <div className="ec-items">
              {concept.key_items.map(item => (
                <code key={item} className="ec-item">{item}</code>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Ask button */}
      <button
        className="ec-ask"
        onClick={e => { e.stopPropagation(); onAsk(concept); }}
      >
        Ask about this →
      </button>
    </div>
  );
}

// ── TracePanel — live log of agent investigation steps ─────────────────────────
// Each entry in `log` is the "trace" payload from a TourAgent SSE event:
//   { type: "info"|"thinking"|"found"|"file"|"finding"|"react", text, name?, stages? }
//
// "react" entries come from the agentic Phase 1 ReAct loop — they show the
// THINK → TOOL → RESULT cycle that the agent uses to explore the codebase.
// Showing this live demonstrates how agentic AI works: the model reasons about
// what to read next, calls a tool, reads the result, and decides where to go.
//
// WHY SHOW THIS: transparency builds trust. When users see "Investigating:
// retrieval/hybrid_search.py" they understand WHY that concept appears in
// the tour — it was specifically investigated, not guessed from a keyword scan.
function TracePanel({ log, open, onToggle }) {
  const bodyRef = useRef(null);

  // Auto-scroll to bottom as new lines arrive
  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [log, open]);

  const ICONS = {
    // ReAct loop step — tool icon (wrench) to distinguish from investigation steps
    react: (
      <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
        <path d="M13.371 2.629a3.5 3.5 0 0 0-4.849 4.274L2.78 12.745a1.5 1.5 0 1 0 2.121 2.121l5.842-5.742a3.5 3.5 0 0 0 2.628-6.495zm-1.414 3.536a1.5 1.5 0 1 1-2.121-2.122 1.5 1.5 0 0 1 2.121 2.122z"/>
      </svg>
    ),
    thinking: (
      <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
        <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zm.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533zM8 5.5a1 1 0 1 0 0-2 1 1 0 0 0 0 2z"/>
      </svg>
    ),
    found: (
      <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
        <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/>
      </svg>
    ),
    file: (
      <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
        <path d="M4 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V4.707A1 1 0 0 0 13.707 4L10 .293A1 1 0 0 0 9.293 0H4zm5.5 1.5v2a1 1 0 0 0 1 1h2l-3-3z"/>
      </svg>
    ),
    finding: (
      <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
        <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.099zm-5.242 1.656a5.5 5.5 0 1 1 0-11 5.5 5.5 0 0 1 0 11z"/>
      </svg>
    ),
    info: (
      <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
        <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>
      </svg>
    ),
  };

  return (
    <div className="ec-trace-panel">
      <div className="ec-trace-header" onClick={onToggle}>
        <svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12">
          <path d="M5 3.5h6A1.5 1.5 0 0 1 12.5 5v5.034a.5.5 0 0 1-.276.447l-1.5.75-.448-.894.776-.388V5a.5.5 0 0 0-.5-.5H5a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h3.5v1H5A1.5 1.5 0 0 1 3.5 12V5A1.5 1.5 0 0 1 5 3.5z"/>
          <path d="M11.854 11.146a.5.5 0 0 0-.707.708L12.293 13H9.5a.5.5 0 0 0 0 1h2.793l-1.147 1.146a.5.5 0 0 0 .708.708l2-2a.5.5 0 0 0 0-.708l-2-2z"/>
        </svg>
        Agent trace — {log.length} steps
        <svg viewBox="0 0 16 16" fill="currentColor" width="10" height="10"
          style={{ marginLeft: "auto", transform: open ? "rotate(180deg)" : undefined, transition: "transform 0.2s" }}>
          <path d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
        </svg>
      </div>
      {open && (
        <div className="ec-trace-body" ref={bodyRef}>
          {log.map((entry, i) => (
            <div key={i} className="ec-trace-line">
              <span className={`ec-trace-icon ${entry.type}`}>
                {ICONS[entry.type] || ICONS.info}
              </span>
              <div className="ec-trace-text">
                {entry.type === "react" ? (
                  // ReAct entries: show tool call prominently, THINK text faint + truncated.
                  // entry.tool = "read_file("backend/services/agent.py")"
                  // entry.think = full reasoning sentence (can be 200+ chars)
                  <>
                    {entry.tool && <span className="ec-trace-react-tool">{entry.tool}</span>}
                    {entry.think && (
                      <span className="ec-trace-react-think">
                        {entry.think.length > 90 ? entry.think.slice(0, 90) + "…" : entry.think}
                      </span>
                    )}
                  </>
                ) : (
                  <>
                    {entry.name && <span className="ec-trace-name">{entry.name} </span>}
                    {entry.text && <span className="ec-trace-sub">{entry.text}</span>}
                    {entry.stages && (
                      <div className="ec-trace-stages">
                        {entry.stages.map((s, j) => (
                          <span key={j} className="ec-trace-stage-pill">{s}</span>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── ExploreView ────────────────────────────────────────────────────────────────
export default function ExploreView({ repo, onAskAbout, onRegenerateRef }) {
  const [data, setData]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [loadStage, setStage]   = useState(null);   // { stage, progress, message }
  const [traceLog, setTrace]    = useState([]);      // agent investigation steps
  const [traceOpen, setTrOpen]  = useState(true);   // trace panel expanded?
  const [error, setError]       = useState(null);
  const [selectedId, setSelected] = useState(null);
  const [hoveredId, setHovered]   = useState(null);
  const [xform, setXform]       = useState({ x: 0, y: 0, scale: window.innerWidth < 768 ? 0.5 : 0.85 });
  const dragging   = useRef(false);
  const drag0      = useRef({});
  const wrapRef    = useRef(null);

  // Per-node drag — same pattern as GraphDiagram
  const [nodePos, setNodePos]  = useState({});  // id → {x,y} overrides
  const dragNode   = useRef(null);              // active node drag state
  const wasDragged = useRef(false);            // suppress click-after-drag
  const scaleRef   = useRef(xform.scale);      // current scale for doc-level handlers
  useEffect(() => { scaleRef.current = xform.scale; }, [xform.scale]);

  // ── Fetch ─────────────────────────────────────────────────────────────────
  const load = useCallback((force = false) => {
    if (!repo) return;
    // 1. In-memory cache: survives tab switches within the same page session.
    if (!force && tourCache[repo]) {
      setData(tourCache[repo]);
      setLoading(false);
      setError(null);
      return;
    }
    // 2. localStorage cache: survives page refreshes. Avoids re-generating
    //    expensive LLM calls just because the user hit F5.
    if (!force) {
      try {
        const stored = localStorage.getItem(tourLsKey(repo));
        if (stored) {
          const parsed = JSON.parse(stored);
          tourCache[repo] = parsed;
          setData(parsed);
          setLoading(false);
          setError(null);
          return;
        }
      } catch { /* corrupt entry — fall through to fetch */ }
    }
    setLoading(true);
    setStage(null);
    setTrace([]);
    setTrOpen(true);
    setError(null);
    setData(null);
    setSelected(null);
    setXform({ x: 0, y: 0, scale: window.innerWidth < 768 ? 0.5 : 0.85 });
    const cancel = streamTour(repo, {
      force,
      onProgress: (ev) => {
        setStage(ev);
        // Accumulate trace events for the live-log panel
        if (ev.trace) setTrace(prev => [...prev, ev.trace]);
      },
      onDone:     (d)  => {
        tourCache[repo] = d;
        try { localStorage.setItem(tourLsKey(repo), JSON.stringify(d)); } catch { /* quota full */ }
        setLoading(false);
        setStage(null);
        setData(d);
      },
      onError:    (e)  => { setLoading(false); setStage(null); setError(e); },
    });
    return cancel;
  }, [repo]);

  useEffect(() => { load(); }, [load]);

  // Reset dragged positions whenever a new tour loads
  useEffect(() => { setNodePos({}); }, [data]);

  // Expose a force-reload function to DiagramView via a ref so the header
  // "Regenerate" button can bust the cache without prop-drilling a callback.
  useEffect(() => {
    if (onRegenerateRef) {
      onRegenerateRef.current = () => {
        delete tourCache[repo];
        try { localStorage.removeItem(tourLsKey(repo)); } catch {}
        load(true);  // force=true → api passes ?force=true → backend busts disk cache
      };
    }
  }, [onRegenerateRef, repo, load]);

  // ── Non-passive wheel zoom ─────────────────────────────────────────────────
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    function onWheel(e) {
      e.preventDefault();
      const f = Math.exp(-e.deltaY * 0.001);
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setXform(t => {
        const newScale = Math.min(Math.max(t.scale * f, 0.3), 3);
        const ratio = newScale / t.scale;
        return { x: mx - (mx - t.x) * ratio, y: my - (my - t.y) * ratio, scale: newScale };
      });
    }
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  // !!data as dep: ExploreView has early returns for loading/error/null states,
  // so wrapRef.current is null on first mount. Re-run once data arrives and the
  // canvas wrapper is actually in the DOM. Cleanup removes the old listener
  // before reattaching, so there's no double-registration risk.
  }, [!!data]);

  // Touch pan (1 finger) + pinch-to-zoom (2 fingers) — same logic as GraphDiagram
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    let lastTouch = null;
    let lastPinch = null;
    function pinchDist(t) { return Math.hypot(t[1].clientX - t[0].clientX, t[1].clientY - t[0].clientY); }
    function pinchMid(t, rect) {
      return { x: (t[0].clientX + t[1].clientX) / 2 - rect.left,
               y: (t[0].clientY + t[1].clientY) / 2 - rect.top };
    }
    function onTouchStart(e) {
      if (e.touches.length === 1) {
        lastTouch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        lastPinch = null;
      } else if (e.touches.length === 2) {
        const t = [...e.touches];
        lastPinch = { dist: pinchDist(t), mid: pinchMid(t, el.getBoundingClientRect()) };
        lastTouch = null;
      }
    }
    function onTouchMove(e) {
      e.preventDefault();
      if (e.touches.length === 1 && lastTouch) {
        const dx = e.touches[0].clientX - lastTouch.x;
        const dy = e.touches[0].clientY - lastTouch.y;
        lastTouch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        setXform(t => ({ ...t, x: t.x + dx, y: t.y + dy }));
      } else if (e.touches.length === 2 && lastPinch) {
        const t   = [...e.touches];
        const rect = el.getBoundingClientRect();
        const d   = pinchDist(t);
        const mid = pinchMid(t, rect);
        const f   = d / lastPinch.dist;
        lastPinch = { dist: d, mid };
        setXform(s => {
          const newScale = Math.min(Math.max(s.scale * f, 0.3), 3);
          const ratio = newScale / s.scale;
          return { x: mid.x - (mid.x - s.x) * ratio, y: mid.y - (mid.y - s.y) * ratio, scale: newScale };
        });
      }
    }
    function onTouchEnd() { lastTouch = null; lastPinch = null; }
    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove",  onTouchMove,  { passive: false });
    el.addEventListener("touchend",   onTouchEnd);
    return () => {
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove",  onTouchMove);
      el.removeEventListener("touchend",   onTouchEnd);
    };
  }, [!!data]);

  // ── Pan handlers ──────────────────────────────────────────────────────────
  // We attach mousemove/mouseup to the DOCUMENT rather than the wrapper div.
  //
  // Why: React synthetic events on the wrapper only fire when the pointer is
  // directly over the wrapper element. The moment it moves over a child card
  // the wrapper's onMouseMove stops firing, breaking the drag mid-gesture.
  //
  // Document-level listeners receive every mouse event regardless of which
  // element the cursor is currently over — the standard pattern for drag.
  useEffect(() => {
    function onDocMove(e) {
      // Node drag takes priority over canvas pan
      if (dragNode.current) {
        const dx = (e.clientX - dragNode.current.startMouse.x) / scaleRef.current;
        const dy = (e.clientY - dragNode.current.startMouse.y) / scaleRef.current;
        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) wasDragged.current = true;
        const id   = dragNode.current.id;
        const newX = dragNode.current.startPos.x + dx;
        const newY = dragNode.current.startPos.y + dy;
        setNodePos(prev => ({ ...prev, [id]: { x: newX, y: newY } }));
        return;
      }
      if (!dragging.current) return;
      setXform(t => ({
        ...t,
        x: drag0.current.tx + (e.clientX - drag0.current.mx),
        y: drag0.current.ty + (e.clientY - drag0.current.my),
      }));
    }
    function onDocUp() {
      dragNode.current = null;
      dragging.current = false;
      if (wrapRef.current) wrapRef.current.style.cursor = "grab";
      setTimeout(() => { wasDragged.current = false; }, 0);
    }
    document.addEventListener("mousemove", onDocMove);
    document.addEventListener("mouseup",   onDocUp);
    return () => {
      document.removeEventListener("mousemove", onDocMove);
      document.removeEventListener("mouseup",   onDocUp);
    };
  }, []); // empty deps — only refs + stable setters used inside

  function onMouseDown(e) {
    if (e.button !== 0) return;
    dragging.current = true;
    drag0.current = { mx: e.clientX, my: e.clientY, tx: xform.x, ty: xform.y };
    if (wrapRef.current) wrapRef.current.style.cursor = "grabbing";
  }

  function onNodeDragStart(e, concept, currentPos) {
    if (e.button !== 0) return;
    e.stopPropagation(); // prevent canvas pan from activating
    dragNode.current = {
      id:         concept.id,
      startPos:   currentPos,
      startMouse: { x: e.clientX, y: e.clientY },
    };
  }

  function handleAsk(concept) {
    onAskAbout?.(
      concept.ask ||
      `Explain "${concept.name}" in ${repo} in detail — what does it do, how does it work, and what are the key methods or functions involved?`
    );
  }

  // ── Loading / error states ─────────────────────────────────────────────────
  if (loading) {
    const pct   = loadStage ? Math.round(loadStage.progress * 100) : 0;
    const rawLabel = loadStage?.message || "Building your guided tour…";
    // Cap the progress label — long THINK strings must not overflow this area
    const label = rawLabel.length > 72 ? rawLabel.slice(0, 72) + "…" : rawLabel;
    return (
      <div className="ec-loading" style={{ flexDirection: "column", alignItems: "stretch", gap: 16, maxWidth: 480, margin: "auto" }}>
        {/* Progress row */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span className="spinner" />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, marginBottom: 6, fontSize: 13 }}>{label}</div>
            <div style={{ height: 3, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${pct}%`,
                background: "var(--accent)", borderRadius: 2, transition: "width 0.5s ease",
              }} />
            </div>
            {pct > 0 && (
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{pct}%</div>
            )}
          </div>
        </div>
        {/* Live agent trace log */}
        {traceLog.length > 0 && (
          <TracePanel log={traceLog} open={traceOpen} onToggle={() => setTrOpen(v => !v)} />
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div className="ec-error">
        <div style={{ fontSize: 13, color: "var(--red)" }}>{error}</div>
        <button className="diagram-retry-btn" onClick={() => load(true)}>Retry</button>
      </div>
    );
  }

  if (!data) return null;

  const concepts      = data.concepts || [];
  const basePositions = computeLayout(concepts);

  // Visual sequence numbers: row-first then column — matches the left-to-right,
  // top-to-bottom reading order of the row-major layout.
  const visualNumber = {};
  Object.entries(basePositions)
    .sort(([, a], [, b]) => a.y !== b.y ? a.y - b.y : a.x - b.x)
    .forEach(([id], i) => { visualNumber[Number(id)] = i + 1; });

  // When a card is expanded push the cards below it in the same column down
  // so the expanded content doesn't overlap them.
  const yOffsets  = expansionOffsets(selectedId, concepts, basePositions);
  const positions = Object.fromEntries(
    Object.entries(basePositions).map(([id, pos]) => [
      id,
      { x: pos.x, y: pos.y + (yOffsets[id] ?? 0) },
    ])
  );

  // Dragged position overrides static layout — falls back to positions[id]
  const getPosFor = (id) => nodePos[id] ?? positions[id];

  // Canvas bounding box — account for potential expansion
  const allX = Object.values(positions).map(p => p.x + CARD_W + 80);
  const allY = Object.values(positions).map(p => p.y + CARD_H + (selectedId !== null ? EXPANSION_H : 0) + 80);
  const canvasW = Math.max(...allX, 700);
  const canvasH = Math.max(...allY, 500);

  // Connected set for hover dimming: selected node + its direct neighbors
  const connectedIds = hoveredId !== null
    ? new Set([
        hoveredId,
        ...concepts.filter(c => c.depends_on?.includes(hoveredId)).map(c => c.id),
        ...(concepts.find(c => c.id === hoveredId)?.depends_on ?? []),
      ])
    : null;

  return (
    <div className="ec-container">
      {/* ── Summary header ── */}
      <div className="ec-header">
        <div className="ec-summary">{data.summary}</div>
        {data.entry_point && (
          <div className="ec-entry-hint">
            <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.6 }}>
              <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"/>
            </svg>
            Start reading: <code>{data.entry_point}</code>
          </div>
        )}
      </div>

      {/* ── Canvas ── */}
      <div
        ref={wrapRef}
        className="ec-canvas-wrapper"
        onMouseDown={onMouseDown}
      >
        <div
          className="ec-canvas"
          style={{
            width: canvasW,
            height: canvasH,
            transform: `translate(${xform.x}px, ${xform.y}px) scale(${xform.scale})`,
            transformOrigin: "0 0",
          }}
        >
          {/* ── SVG arrow layer ── */}
          {/* Each arrow has two layers:
              1. Base path — solid thin line + arrowhead (always visible)
              2. Traveling dot — small glowing circle that animates from source to target
                 using <animateMotion> + <mpath>. This communicates DIRECTION: you can
                 instantly see which way concepts depend on each other. The dot fades in
                 after 10% of the journey and fades out before 90% so it never looks
                 abrupt at the endpoints. Highlighted arrows skip the dot — the glow
                 filter communicates selection state instead. */}
          <svg
            width={canvasW}
            height={canvasH}
            style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "visible" }}
            aria-hidden="true"
          >
            <defs>
              <marker id="ec-arrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="rgba(91,143,249,0.4)" />
              </marker>
              <marker id="ec-arrow-hi" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#7DABFF" />
              </marker>
              {/* Amber arrowhead for sequential reading-path arrows */}
              <marker id="ec-arrow-seq" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="rgba(245,158,11,0.75)" />
              </marker>
            </defs>

            {/* ── Sequential reading-path arrows (amber) ──────────────────
                 Connect concept N → N+1 in reading order so the learning
                 path is visually explicit. These are the primary navigation
                 guide; dependency arrows (blue) are supporting context. */}
            {(() => {
              const seq = [...concepts].sort((a, b) =>
                (a.reading_order ?? 999) - (b.reading_order ?? 999)
              );
              return seq.slice(0, -1).map((c, i) => {
                const next = seq[i + 1];
                const from = getPosFor(c.id);
                const to   = getPosFor(next.id);
                if (!from || !to) return null;
                const d = bezierPath(from, to);
                const isDim = connectedIds && !connectedIds.has(c.id) && !connectedIds.has(next.id);
                return (
                  <path
                    key={`seq-${c.id}→${next.id}`}
                    d={d}
                    stroke="rgba(245,158,11,0.50)"
                    strokeWidth="1.5"
                    fill="none"
                    markerEnd="url(#ec-arrow-seq)"
                    strokeDasharray="5 3"
                    style={{ opacity: isDim ? 0.06 : 1, transition: "opacity 0.15s" }}
                  />
                );
              });
            })()}

            {/* ── Dependency arrows (blue) — prerequisite relationships ── */}
            {concepts.map(c =>
              (c.depends_on ?? []).map(depId => {
                const from = getPosFor(depId);
                const to   = getPosFor(c.id);
                if (!from || !to) return null;

                const isHi  = connectedIds?.has(c.id) && connectedIds?.has(depId);
                const isDim = connectedIds && !isHi;
                const pathId = `ec-path-${depId}-${c.id}`;
                // Stagger dot travel per connection so all dots don't move in sync
                const stagger = `${((depId * 3 + c.id * 7) % 40) / 10}s`;
                const d = bezierPath(from, to);

                return (
                  <g key={`${depId}→${c.id}`} style={{ opacity: isDim ? 0.08 : 1, transition: "opacity 0.15s" }}>
                    {/* Base arrow path */}
                    <path
                      id={pathId}
                      d={d}
                      stroke={isHi ? "#7DABFF" : "rgba(91,143,249,0.35)"}
                      strokeWidth={isHi ? 2 : 1.2}
                      fill="none"
                      markerEnd={isHi ? "url(#ec-arrow-hi)" : "url(#ec-arrow)"}
                      style={{
                        filter: isHi ? "drop-shadow(0 0 3px rgba(125,171,255,0.6))" : undefined,
                        transition: "stroke 0.15s, stroke-width 0.15s, filter 0.15s",
                      }}
                    />
                    {/* Traveling dot — communicates flow direction.
                        Hidden on highlighted arrows (glow covers it) and dimmed arrows. */}
                    {!isHi && !isDim && (
                      <circle r="2.5" fill="#7DABFF">
                        {/* Fade in at 10%, full at 20%, full at 80%, fade out at 90% */}
                        <animate attributeName="opacity"
                          values="0;0;1;1;0;0" keyTimes="0;0.1;0.2;0.8;0.9;1"
                          dur="4s" begin={stagger} repeatCount="indefinite" />
                        <animateMotion dur="4s" begin={stagger} repeatCount="indefinite" rotate="auto">
                          <mpath href={`#${pathId}`} />
                        </animateMotion>
                      </circle>
                    )}
                  </g>
                );
              })
            )}
          </svg>

          {/* ── Concept cards ── */}
          {concepts.map(c => {
            const pos = getPosFor(c.id);
            if (!pos) return null;
            // isEntry = the leftmost card (visual number 1) — always the pipeline overview
            const isEntry = visualNumber[c.id] === 1;
            return (
              <ConceptCard
                key={c.id}
                concept={c}
                visualNum={visualNumber[c.id]}
                isEntry={isEntry}
                isSelected={selectedId === c.id}
                isHovered={hoveredId === c.id || (!!connectedIds && connectedIds.has(c.id))}
                isDimmed={!!connectedIds && !connectedIds.has(c.id)}
                pos={pos}
                onSelect={id => setSelected(v => v === id ? null : id)}
                onHover={setHovered}
                onAsk={handleAsk}
                onDragStart={onNodeDragStart}
                wasDragged={wasDragged}
              />
            );
          })}
        </div>
      </div>

      {/* ── Legend + hint ── */}
      <div className="ec-legend">
        {Object.entries(TYPE_STYLE).map(([type, s]) => (
          <span key={type} className="ec-legend-item">
            <span className="ec-legend-dot" style={{ background: s.dot }} />
            {type}
          </span>
        ))}
        <span className="ec-legend-hint">
          {concepts.length} concepts · scroll to zoom · drag canvas or cards · click to expand
        </span>
      </div>
    </div>
  );
}

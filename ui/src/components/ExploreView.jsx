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
// Each concept type gets a distinct accent color so students can visually
// group "data structures" vs "algorithms" vs "entry points" at a glance.
// All colors are warm-toned to stay within the sketchbook palette —
// no cold violet/indigo; teal and amber are the warm analogues.
// Blueprint palette — each type gets a distinct hue in the blue/teal family
const TYPE_STYLE = {
  class:     { border: "#5B8FF9", glow: "rgba(91,143,249,0.35)",  dot: "#7DABFF", tag: "class"  }, // blueprint blue
  function:  { border: "#2DD4BF", glow: "rgba(45,212,191,0.32)",  dot: "#5EEAD4", tag: "fn"     }, // teal
  module:    { border: "#818CF8", glow: "rgba(129,140,248,0.32)", dot: "#A5B4FC", tag: "module" }, // indigo
  algorithm: { border: "#38BDF8", glow: "rgba(56,189,248,0.32)",  dot: "#7DD3FC", tag: "algo"   }, // sky
};
const FALLBACK_STYLE = { border: "#4E5E80", glow: "rgba(78,94,128,0.30)", dot: "#8896B8", tag: "?" };

function styleFor(type) {
  return TYPE_STYLE[type] || FALLBACK_STYLE;
}

// ── Card geometry ─────────────────────────────────────────────────────────────
const CARD_W        = 220;  // card width in canvas px
const CARD_H        = 172;  // collapsed card height
const COL_GAP       = 110;  // horizontal gap between columns
const ROW_GAP       = 36;   // vertical gap between rows in the same column
const EXPANSION_H   = 195;  // extra px a card grows when expanded (desc + items + padding)

// When a card is expanded, push all cards BELOW it in the same column down
// by EXPANSION_H so they don't overlap. Returns { [id]: yOffset }.
function expansionOffsets(selectedId, concepts, basePositions) {
  if (selectedId === null) return {};
  const sel = basePositions[selectedId];
  if (!sel) return {};
  const offsets = {};
  concepts.forEach(c => {
    if (c.id === selectedId) return;
    const p = basePositions[c.id];
    // Same column (same x) and strictly below the expanded card
    if (p && p.x === sel.x && p.y > sel.y) offsets[c.id] = EXPANSION_H;
  });
  return offsets;
}

// ── Layout: topological column assignment ─────────────────────────────────────
// Returns { [conceptId]: { x, y } } in canvas coordinates.
function computeLayout(concepts) {
  if (!concepts.length) return {};

  // Step 1: compute depth = longest path from any root (O(n²) is fine for 8 nodes)
  const depthCache = {};
  function depth(id) {
    if (id in depthCache) return depthCache[id];
    const c = concepts.find(x => x.id === id);
    if (!c || !c.depends_on?.length) return (depthCache[id] = 0);
    depthCache[id] = 1 + Math.max(...c.depends_on.map(depth));
    return depthCache[id];
  }
  concepts.forEach(c => depth(c.id));

  // Step 2: group by column, sort within column by reading_order
  const cols = {};
  concepts.forEach(c => {
    const col = depthCache[c.id] ?? 0;
    if (!cols[col]) cols[col] = [];
    cols[col].push(c);
  });
  Object.values(cols).forEach(arr =>
    arr.sort((a, b) => (a.reading_order ?? 99) - (b.reading_order ?? 99))
  );

  // Step 3: assign pixel positions — center each column vertically
  const maxColH = Math.max(...Object.values(cols).map(a => a.length)) * (CARD_H + ROW_GAP);
  const positions = {};
  Object.entries(cols).forEach(([col, nodes]) => {
    const x = Number(col) * (CARD_W + COL_GAP) + 48;
    const colH = nodes.length * (CARD_H + ROW_GAP) - ROW_GAP;
    const startY = (maxColH - colH) / 2 + 48;
    nodes.forEach((node, row) => {
      positions[node.id] = { x, y: startY + row * (CARD_H + ROW_GAP) };
    });
  });
  return positions;
}

// ── Arrow: cubic bezier from right-edge of source to left-edge of target ──────
function bezierPath(fromPos, toPos) {
  const x1 = fromPos.x + CARD_W;
  const y1 = fromPos.y + CARD_H / 2;
  const x2 = toPos.x;
  const y2 = toPos.y + CARD_H / 2;
  const tension = Math.max((x2 - x1) * 0.55, 60);
  return `M ${x1} ${y1} C ${x1 + tension} ${y1}, ${x2 - tension} ${y2}, ${x2} ${y2}`;
}

// ── ConceptCard ────────────────────────────────────────────────────────────────
function ConceptCard({ concept, isEntry, isSelected, isHovered, isDimmed, chainStyle, pos, onSelect, onHover, onAsk }) {
  const s = styleFor(concept.type);
  // When this card is part of a hover chain, use the source node's color so the
  // entire connected subgraph glows as one unified circuit, not a collision of hues.
  const hs = (isHovered && chainStyle) ? chainStyle : s;

  return (
    <div
      className={`ec-card${isSelected ? " ec-selected" : ""}${isDimmed ? " ec-dimmed" : ""}`}
      style={{
        position: "absolute",
        zIndex: isSelected ? 10 : 1,
        left: pos.x,
        top: pos.y,
        width: CARD_W,
        borderColor: isSelected
          ? s.border
          : isHovered
          ? hs.border
          : undefined,
        boxShadow: isSelected
          ? `0 0 0 2px ${s.border}, 0 0 20px ${s.glow.replace(/[\d.]+\)$/, '0.60)')}, 0 20px 60px ${s.glow.replace(/[\d.]+\)$/, '0.45)')}`
          : isHovered
          ? `0 0 0 2px ${hs.border}, 0 0 20px ${hs.glow.replace(/[\d.]+\)$/, '0.60)')}, 0 20px 60px ${hs.glow.replace(/[\d.]+\)$/, '0.45)')}`
          : undefined,
      }}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={() => onSelect(concept.id)}
      onMouseEnter={() => onHover(concept.id)}
      onMouseLeave={() => onHover(null)}
    >
      {/* Top row: reading order badge + type tag */}
      <div className="ec-card-top">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="ec-order">{concept.reading_order}</span>
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

// ── ExploreView ────────────────────────────────────────────────────────────────
export default function ExploreView({ repo, onAskAbout, onRegenerateRef }) {
  const [data, setData]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [loadStage, setStage]   = useState(null);   // { stage, progress, message }
  const [error, setError]       = useState(null);
  const [selectedId, setSelected] = useState(null);
  const [hoveredId, setHovered]   = useState(null);
  const [xform, setXform]       = useState({ x: 0, y: 0, scale: 0.85 });
  const dragging = useRef(false);
  const drag0    = useRef({});
  const wrapRef  = useRef(null);

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
    setError(null);
    setData(null);
    setSelected(null);
    setXform({ x: 0, y: 0, scale: 0.85 });
    const cancel = streamTour(repo, {
      force,
      onProgress: (ev) => setStage(ev),
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
      const f = e.deltaY > 0 ? 0.9 : 1.11;
      setXform(t => ({ ...t, scale: Math.min(Math.max(t.scale * f, 0.3), 3) }));
    }
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

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
      if (!dragging.current) return;
      setXform(t => ({
        ...t,
        x: drag0.current.tx + (e.clientX - drag0.current.mx),
        y: drag0.current.ty + (e.clientY - drag0.current.my),
      }));
    }
    function onDocUp() {
      if (!dragging.current) return;
      dragging.current = false;
      if (wrapRef.current) wrapRef.current.style.cursor = "grab";
    }
    document.addEventListener("mousemove", onDocMove);
    document.addEventListener("mouseup",   onDocUp);
    return () => {
      document.removeEventListener("mousemove", onDocMove);
      document.removeEventListener("mouseup",   onDocUp);
    };
  }, []); // empty deps — only refs + setXform (stable) are used inside

  function onMouseDown(e) {
    if (e.button !== 0) return;
    dragging.current = true;
    drag0.current = { mx: e.clientX, my: e.clientY, tx: xform.x, ty: xform.y };
    if (wrapRef.current) wrapRef.current.style.cursor = "grabbing";
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
    const label = loadStage?.message || "Building your guided tour…";
    return (
      <div className="ec-loading">
        <span className="spinner" />
        <div style={{ flex: 1, maxWidth: 320 }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>{label}</div>
          {/* Progress bar */}
          <div style={{
            height: 3, background: "var(--border)", borderRadius: 2, overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              width:  `${pct}%`,
              background:  "var(--accent)",
              borderRadius: 2,
              transition:  "width 0.4s ease",
            }} />
          </div>
          {pct > 0 && (
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
              {pct}%
            </div>
          )}
        </div>
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

  // When a card is expanded push the cards below it in the same column down
  // so the expanded content doesn't overlap them.
  const yOffsets  = expansionOffsets(selectedId, concepts, basePositions);
  const positions = Object.fromEntries(
    Object.entries(basePositions).map(([id, pos]) => [
      id,
      { x: pos.x, y: pos.y + (yOffsets[id] ?? 0) },
    ])
  );

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

  // The entry point file tells us which concept is the "Start here" one
  const entryFile = data.entry_point?.split("/").pop() ?? "";

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
          <svg
            width={canvasW}
            height={canvasH}
            style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "visible" }}
            aria-hidden="true"
          >
            <defs>
              {/* Default arrowhead — warm sienna */}
              <marker id="ec-arrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="rgba(91,143,249,0.35)" />
              </marker>
              {/* Highlighted arrowhead */}
              <marker id="ec-arrow-hi" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#7DABFF" />
              </marker>
            </defs>

            {concepts.map(c =>
              (c.depends_on ?? []).map(depId => {
                const from = positions[depId];
                const to   = positions[c.id];
                if (!from || !to) return null;

                const isHi = connectedIds?.has(c.id) && connectedIds?.has(depId);
                const isDim = connectedIds && !isHi;

                return (
                  <path
                    key={`${depId}→${c.id}`}
                    d={bezierPath(from, to)}
                    stroke={isHi ? "#7DABFF" : "rgba(91,143,249,0.22)"}
                    strokeWidth={isHi ? 2 : 1.5}
                    strokeDasharray={isHi ? undefined : "none"}
                    fill="none"
                    markerEnd={isHi ? "url(#ec-arrow-hi)" : "url(#ec-arrow)"}
                    style={{
                      opacity: isDim ? 0.12 : 1,
                      transition: "opacity 0.15s, stroke 0.15s, stroke-width 0.15s",
                    }}
                  />
                );
              })
            )}
          </svg>

          {/* ── Concept cards ── */}
          {(() => {
            // Compute the hovered node's style once so all connected nodes share it,
            // making the entire chain glow as a single unified color.
            const chainStyle = hoveredId
              ? styleFor(concepts.find(c => c.id === hoveredId)?.type)
              : null;
            return concepts.map(c => {
              const pos = positions[c.id];
              if (!pos) return null;
              // Only the lowest reading-order concept from the entry file gets "Start here".
              // Without this, every chunk from engine.py shows the badge.
              const entryMatches = concepts.filter(x => entryFile && x.file?.endsWith(entryFile));
              const minEntryOrder = entryMatches.length ? Math.min(...entryMatches.map(x => x.reading_order)) : null;
              const isEntry = !!(entryFile && c.file?.endsWith(entryFile) && c.reading_order === minEntryOrder);
              return (
                <ConceptCard
                  key={c.id}
                  concept={c}
                  isEntry={isEntry}
                  isSelected={selectedId === c.id}
                  isHovered={hoveredId === c.id || (!!connectedIds && connectedIds.has(c.id))}
                  isDimmed={!!connectedIds && !connectedIds.has(c.id)}
                  chainStyle={chainStyle}
                  pos={pos}
                  onSelect={id => setSelected(v => v === id ? null : id)}
                  onHover={setHovered}
                  onAsk={handleAsk}
                />
              );
            });
          })()}
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
          {concepts.length} concepts · scroll to zoom · drag to pan · click to expand
        </span>
      </div>
    </div>
  );
}

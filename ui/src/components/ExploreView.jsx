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
import { fetchTour } from "../api";

// ── Type → visual token ───────────────────────────────────────────────────────
// Each concept type gets a distinct accent color so students can visually
// group "data structures" vs "algorithms" vs "entry points" at a glance.
const TYPE_STYLE = {
  class:     { border: "#7C3AED", glow: "rgba(124,58,237,0.4)",  dot: "#a78bfa", tag: "class"  },
  function:  { border: "#0D9488", glow: "rgba(13,148,136,0.4)",  dot: "#2dd4bf", tag: "fn"     },
  module:    { border: "#B45309", glow: "rgba(180,83,9,0.4)",    dot: "#fbbf24", tag: "module" },
  algorithm: { border: "#BE185D", glow: "rgba(190,24,93,0.4)",   dot: "#f472b6", tag: "algo"   },
};
const FALLBACK_STYLE = { border: "#4F46E5", glow: "rgba(79,70,229,0.4)", dot: "#818CF8", tag: "?" };

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
function ConceptCard({ concept, isEntry, isSelected, isHovered, isDimmed, pos, onSelect, onHover, onAsk }) {
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
        borderColor: isSelected
          ? s.border
          : isHovered
          ? "rgba(167,139,250,0.5)"
          : undefined,
        boxShadow: isSelected
          ? `0 0 0 1.5px ${s.border}, 0 8px 40px ${s.glow}`
          : isHovered
          ? "0 4px 24px rgba(0,0,0,0.5)"
          : undefined,
      }}
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
export default function ExploreView({ repo, onAskAbout }) {
  const [data, setData]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [selectedId, setSelected] = useState(null);
  const [hoveredId, setHovered]   = useState(null);
  const [xform, setXform]       = useState({ x: 0, y: 0, scale: 0.85 });
  const dragging = useRef(false);
  const drag0    = useRef({});
  const wrapRef  = useRef(null);

  // ── Fetch ─────────────────────────────────────────────────────────────────
  const load = useCallback(() => {
    if (!repo) return;
    setLoading(true);
    setError(null);
    setData(null);
    setSelected(null);
    setXform({ x: 0, y: 0, scale: 0.85 });
    fetchTour(repo)
      .then(d => { setLoading(false); setData(d); })
      .catch(e => { setLoading(false); setError(e.message); });
  }, [repo]);

  useEffect(() => { load(); }, [load]);

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
  function onMouseDown(e) {
    if (e.button !== 0) return;
    dragging.current = true;
    drag0.current = { mx: e.clientX, my: e.clientY, tx: xform.x, ty: xform.y };
    e.currentTarget.style.cursor = "grabbing";
  }
  function onMouseMove(e) {
    if (!dragging.current) return;
    setXform(t => ({
      ...t,
      x: drag0.current.tx + (e.clientX - drag0.current.mx),
      y: drag0.current.ty + (e.clientY - drag0.current.my),
    }));
  }
  function onMouseUp(e) {
    dragging.current = false;
    if (e.currentTarget) e.currentTarget.style.cursor = "grab";
  }

  function handleAsk(concept) {
    onAskAbout?.(
      concept.ask ||
      `Explain "${concept.name}" in ${repo} in detail — what does it do, how does it work, and what are the key methods or functions involved?`
    );
  }

  // ── Loading / error states ─────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="ec-loading">
        <span className="spinner" />
        <div>
          <div style={{ fontWeight: 600 }}>Building your guided tour…</div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
            Analysing {repo.split("/")[1]} to find the key concepts
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ec-error">
        <div style={{ fontSize: 13, color: "var(--red)" }}>{error}</div>
        <button className="diagram-retry-btn" onClick={load}>Retry</button>
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
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
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
              {/* Default arrowhead */}
              <marker id="ec-arrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="rgba(139,92,246,0.35)" />
              </marker>
              {/* Highlighted arrowhead */}
              <marker id="ec-arrow-hi" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#a78bfa" />
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
                    stroke={isHi ? "#a78bfa" : "rgba(139,92,246,0.2)"}
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
          {concepts.map(c => {
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
                isHovered={hoveredId === c.id}
                isDimmed={!!connectedIds && !connectedIds.has(c.id)}
                pos={pos}
                onSelect={id => setSelected(v => v === id ? null : id)}
                onHover={setHovered}
                onAsk={handleAsk}
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
          {concepts.length} concepts · scroll to zoom · drag to pan · click to expand
        </span>
      </div>
    </div>
  );
}

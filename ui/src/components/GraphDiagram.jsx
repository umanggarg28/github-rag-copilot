/**
 * GraphDiagram.jsx — Custom SVG canvas renderer for Architecture and Class diagrams.
 *
 * Replaced React Flow with the same hand-crafted SVG approach as ExploreView:
 *   - No port handles, no selection rings, no third-party CSS artifacts
 *   - Full visual control — same warm sketchbook palette throughout
 *   - Pan/zoom via CSS transform (scroll = zoom, drag = pan)
 *   - Bezier arrows with clickable hit areas for edge detail
 *
 * Layout: BFS topological columns — nodes with no incoming edges go left,
 * their dependents go right. Columns that exceed MAX_PER_COL are split
 * into sub-columns to prevent tall stacks.
 */

import { useEffect, useMemo, useCallback, useState, useRef } from "react";

// ── Type colours — blueprint palette, consistent with ExploreView ────────────
const TYPE_STYLE = {
  module:    { border: "#818CF8", glow: "rgba(129,140,248,0.32)", dot: "#A5B4FC" }, // indigo
  class:     { border: "#5B8FF9", glow: "rgba(91,143,249,0.35)",  dot: "#7DABFF" }, // blueprint blue
  abstract:  { border: "#7DABFF", glow: "rgba(125,171,255,0.30)", dot: "#A8C5FF" }, // lighter blue
  mixin:     { border: "#60A5FA", glow: "rgba(96,165,250,0.30)",  dot: "#93C5FD" }, // sky-blue
  service:   { border: "#2DD4BF", glow: "rgba(45,212,191,0.32)",  dot: "#5EEAD4" }, // teal
  database:  { border: "#38BDF8", glow: "rgba(56,189,248,0.32)",  dot: "#7DD3FC" }, // sky
  external:  { border: "#4E5E80", glow: "rgba(78,94,128,0.28)",   dot: "#8896B8" }, // steel
  input:     { border: "#2DD4BF", glow: "rgba(45,212,191,0.32)",  dot: "#5EEAD4" }, // teal
  transform: { border: "#818CF8", glow: "rgba(129,140,248,0.30)", dot: "#A5B4FC" }, // indigo
  output:    { border: "#5B8FF9", glow: "rgba(91,143,249,0.32)",  dot: "#7DABFF" }, // blue
};
const FALLBACK_STYLE = { border: "#4E5E80", glow: "rgba(78,94,128,0.28)", dot: "#8896B8" };
function styleFor(type) { return TYPE_STYLE[type] || FALLBACK_STYLE; }

// ── Card geometry ─────────────────────────────────────────────────────────────
const CARD_W   = 220;   // canvas pixels — matches ExploreView width
const CARD_H   = 240;   // layout height per card (type+name+file+desc+items+ask ≈ 230px + buffer)
const COL_GAP  = 100;   // horizontal space between columns
const ROW_GAP  = 48;    // vertical gap between cards in the same column

// ── Bezier arrow: right-edge of source → left-edge of target ─────────────────
// Identical formula to ExploreView — consistent arrow shape across all views.
function bezierPath(from, to) {
  const x1 = from.x + CARD_W;
  const y1 = from.y + CARD_H / 2;
  const x2 = to.x;
  const y2 = to.y + CARD_H / 2;
  // Tension scales with horizontal distance so short and long arrows both look
  // natural — short: tight S-curve, long: gentle arc.
  const tension = Math.max((x2 - x1) * 0.55, 60);
  return `M ${x1} ${y1} C ${x1 + tension} ${y1}, ${x2 - tension} ${y2}, ${x2} ${y2}`;
}

// ── Layout: BFS topological columns ──────────────────────────────────────────
// Returns { nodes: [{id, x, y, data}], edges: [{source, target, label}] }
function computeLayout(rawNodes, rawEdges) {
  if (!rawNodes?.length) return { nodes: [], edges: [], posMap: {} };

  const MAX_PER_COL = 4;

  // BFS to assign column depth to each node
  const inDegree = {};
  const outEdges = {};
  rawNodes.forEach(n => { inDegree[n.id] = 0; outEdges[n.id] = []; });
  rawEdges.forEach(e => {
    if (inDegree[e.target] !== undefined) inDegree[e.target]++;
    if (outEdges[e.source])               outEdges[e.source].push(e.target);
  });

  const depth = {};
  rawNodes.forEach(n => { depth[n.id] = 0; });
  const queue   = rawNodes.filter(n => inDegree[n.id] === 0).map(n => n.id);
  const tmpIn   = { ...inDegree };
  const visited = new Set(queue);
  let head = 0;
  while (head < queue.length) {
    const id = queue[head++];
    for (const nb of (outEdges[id] || [])) {
      depth[nb] = Math.max(depth[nb], depth[id] + 1);
      tmpIn[nb]--;
      if (tmpIn[nb] === 0 && !visited.has(nb)) { visited.add(nb); queue.push(nb); }
    }
  }

  // Group nodes by column
  const cols = {};
  rawNodes.forEach(n => {
    const c = depth[n.id];
    if (!cols[c]) cols[c] = [];
    cols[c].push(n);
  });

  // Split over-tall columns into sub-columns
  const colKeys = Object.keys(cols).map(Number).sort((a, b) => a - b);
  const expandedCols = {};
  let ci = 0;
  colKeys.forEach(col => {
    const nodes = cols[col];
    for (let s = 0; s < nodes.length; s += MAX_PER_COL) {
      expandedCols[ci++] = nodes.slice(s, s + MAX_PER_COL);
    }
  });

  // Assign pixel positions — center each column vertically around the tallest column.
  // We must compute maxColH first so every column is centered against the same baseline,
  // which guarantees all y values are non-negative (no card above the canvas top edge).
  // This mirrors ExploreView's layout algorithm exactly.
  const posMap = {};
  const finalColKeys = Object.keys(expandedCols).map(Number).sort((a, b) => a - b);
  const maxColH = Math.max(
    ...finalColKeys.map(col => {
      const n = expandedCols[col].length;
      return n * CARD_H + (n - 1) * ROW_GAP;
    })
  );
  finalColKeys.forEach((col) => {
    const colNodes = expandedCols[col];
    const colH   = colNodes.length * CARD_H + (colNodes.length - 1) * ROW_GAP;
    const startY = (maxColH - colH) / 2 + 48;  // always ≥ 48px — no negative coords
    colNodes.forEach((n, ri) => {
      posMap[n.id] = {
        x: col * (CARD_W + COL_GAP) + 48,
        y: startY + ri * (CARD_H + ROW_GAP),
      };
    });
  });

  const nodes = rawNodes.map(n => ({
    id:   n.id,
    x:    posMap[n.id]?.x ?? 0,
    y:    posMap[n.id]?.y ?? 0,
    data: {
      label:       n.label,
      type:        n.type,
      file:        n.file,
      description: n.description,
      items:       n.items || n.methods || [],
    },
  }));

  const edges = rawEdges.map(e => ({
    source: e.source,
    target: e.target,
    label:  e.label || "",
  }));

  return { nodes, edges, posMap };
}

// ── File icon — same SVG used in ExploreView's ec-file pill ──────────────────
function FileIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"
      style={{ opacity: 0.5, flexShrink: 0 }}>
      <path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 8.75 4.25V1.5Zm6.75.56v2.19c0 .138.112.25.25.25h2.19Z"/>
    </svg>
  );
}

// ── DiagramCard — ec-card positioned absolutely on the SVG canvas ─────────────
function DiagramCard({ node, pos, hoveredId, connectedIds, onSelect, onHover, onAsk, onDragStart, wasDragged }) {
  const dim       = hoveredId && hoveredId !== node.id && !connectedIds.has(node.id);
  const highlight = hoveredId === node.id || (hoveredId && connectedIds.has(node.id));
  const s = styleFor(node.data.type);

  return (
    <div
      className={`ec-card${dim ? " ec-dimmed" : ""}`}
      style={{
        position: "absolute",
        left:  pos.x,
        top:   pos.y,
        width: CARD_W,
        cursor: "grab",
        // Colour-keyed top accent bar using an inset box-shadow on the card —
        // avoids a separate child div while still giving each type its own colour.
        borderColor: highlight ? s.dot : undefined,
        boxShadow:   highlight
          ? `0 0 0 1.5px ${s.dot}, 0 8px 32px ${s.glow}`
          : `inset 0 2.5px 0 0 ${s.border}`,
      }}
      onMouseDown={(e) => onDragStart?.(e, node)}
      onClick={() => { if (!wasDragged?.current) onSelect(node); }}
      onMouseEnter={() => onHover(node.id)}
      onMouseLeave={() => onHover(null)}
    >
      {/* Top row: type tag */}
      <div className="ec-card-top" style={{ marginBottom: 6 }}>
        <span className="ec-type-tag" style={{ color: s.dot, borderColor: `${s.dot}44` }}>
          {node.data.type}
        </span>
      </div>

      {/* Name */}
      <div className="ec-name">{node.data.label}</div>

      {/* File path */}
      {node.data.file && (
        <div className="ec-file" style={{ marginTop: 3 }}>
          <FileIcon />
          {node.data.file}
        </div>
      )}

      {/* Description — truncated; full text shown in NodeDetailPanel */}
      {node.data.description && (
        <div className="ec-subtitle" style={{ marginTop: 4 }}>
          {node.data.description.length > 90
            ? node.data.description.slice(0, 88) + "…"
            : node.data.description}
        </div>
      )}

      {/* Method pills */}
      {node.data.items?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginTop: 6 }}>
          {node.data.items.slice(0, 4).map((item, i) => (
            <code key={i} className="ec-item">{item}</code>
          ))}
        </div>
      )}

      {/* Ask button */}
      {onAsk && (
        <button className="ec-ask" onClick={e => { e.stopPropagation(); onAsk(node.data); }}>
          Ask about this →
        </button>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function GraphDiagram({ data, onNodeSelect, onEdgeSelect, onAskAbout, repo, panelOpen }) {
  const { nodes: layoutNodes, edges: layoutEdges, posMap } = useMemo(
    () => computeLayout(data?.nodes || [], data?.edges || []),
    [data]
  );

  const [hoveredId, setHoveredId] = useState(null);
  const [xform, setXform]         = useState({ x: 0, y: 0, scale: 0.85 });

  // nodePos overrides the static layout position for dragged nodes.
  // Key = node id, value = { x, y } in canvas coordinates.
  const [nodePos, setNodePos] = useState({});

  // canvas pan state
  const dragging  = useRef(false);
  const drag0     = useRef({});
  // Mirror of xform.scale in a ref so document-level handlers (which have
  // empty deps and can't close over xform) can always read the current scale.
  const scaleRef  = useRef(xform.scale);
  useEffect(() => { scaleRef.current = xform.scale; }, [xform.scale]);

  // per-node drag state — set on card mousedown, cleared on mouseup
  // { id, startPos: {x,y}, startMouse: {x,y} }
  const dragNode    = useRef(null);
  // Set to true when the node actually moves during a drag gesture.
  // Read by DiagramCard's onClick to suppress the click-to-select that
  // browsers fire automatically after every mousedown+mouseup pair.
  const wasDragged  = useRef(false);

  const wrapRef  = useRef(null);

  // Reset view and any manual node positions when the diagram data changes
  useEffect(() => {
    setXform({ x: 0, y: 0, scale: 0.85 });
    setNodePos({});
  }, [data]);

  // Non-passive wheel zoom — passive: false required to call preventDefault()
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    function onWheel(e) {
      e.preventDefault();
      const f = e.deltaY > 0 ? 0.9 : 1.11;
      setXform(t => ({ ...t, scale: Math.min(Math.max(t.scale * f, 0.2), 2.5) }));
    }
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // Document-level mousemove/mouseup so drag works even when the cursor
  // glides over child cards or outside the wrapper boundary mid-gesture.
  useEffect(() => {
    function onDocMove(e) {
      if (dragNode.current) {
        const dx = (e.clientX - dragNode.current.startMouse.x) / scaleRef.current;
        const dy = (e.clientY - dragNode.current.startMouse.y) / scaleRef.current;
        // Only count as a drag if the node moved more than 4px — prevents
        // suppressing clicks caused by tiny hand tremor on mousedown.
        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) wasDragged.current = true;
        // Capture id and new position NOW — before passing the callback to
        // setNodePos. React may call the callback asynchronously, by which
        // point onDocUp could have already set dragNode.current = null,
        // causing a "Cannot read property of null" crash.
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
      // wasDragged is reset AFTER the click event fires (click fires synchronously
      // after mouseup in the same task). setTimeout(0) defers the reset past it.
      setTimeout(() => { wasDragged.current = false; }, 0);
    }
    document.addEventListener("mousemove", onDocMove);
    document.addEventListener("mouseup",   onDocUp);
    return () => {
      document.removeEventListener("mousemove", onDocMove);
      document.removeEventListener("mouseup",   onDocUp);
    };
  }, []); // empty deps — only refs + stable setters used inside

  // Returns the live (possibly dragged) position for a node id.
  // Falls back to posMap (static layout) when the node hasn't been moved.
  const getPosFor = useCallback((id) => nodePos[id] ?? posMap[id], [nodePos, posMap]);

  // Canvas pan — fires only when no node drag is in progress
  function onMouseDown(e) {
    if (e.button !== 0) return;
    dragging.current = true;
    drag0.current = { mx: e.clientX, my: e.clientY, tx: xform.x, ty: xform.y };
    e.currentTarget.style.cursor = "grabbing";
  }

  // Called from DiagramCard — stops propagation so the wrapper's onMouseDown
  // (canvas pan) does NOT fire when dragging a node.
  function onNodeDragStart(e, node) {
    if (e.button !== 0) return;
    e.stopPropagation();
    const current = nodePos[node.id] ?? { x: node.x, y: node.y };
    dragNode.current = {
      id:         node.id,
      startPos:   current,
      startMouse: { x: e.clientX, y: e.clientY },
    };
  }


  // Connected IDs for hover dimming
  const connectedIds = useMemo(() => {
    if (!hoveredId) return new Set();
    const ids = new Set();
    layoutEdges.forEach(e => {
      if (e.source === hoveredId) ids.add(e.target);
      if (e.target === hoveredId) ids.add(e.source);
    });
    return ids;
  }, [hoveredId, layoutEdges]);

  const handleNodeSelect = useCallback((node) => {
    onNodeSelect?.({
      kind:        "node",
      id:          node.id,
      label:       node.data.label,
      type:        node.data.type,
      file:        node.data.file,
      description: node.data.description,
      items:       node.data.items || [],
    });
  }, [onNodeSelect]);

  const handleEdgeClick = useCallback((edge) => {
    const srcNode = layoutNodes.find(n => n.id === edge.source);
    const tgtNode = layoutNodes.find(n => n.id === edge.target);
    const srcLabel = srcNode?.data?.label || edge.source;
    const tgtLabel = tgtNode?.data?.label || edge.target;
    onEdgeSelect?.({
      kind:         "edge",
      label:        `${srcLabel} → ${tgtLabel}`,
      type:         "edge",
      file:         null,
      description:  edge.label ? `Relationship: "${edge.label}"` : "",
      items:        [],
      autoQuestion: `In ${repo}, why does "${srcLabel}" ${edge.label || "depend on"} "${tgtLabel}"? What specifically does it use from it, and what would break if this dependency were removed?`,
    });
  }, [onEdgeSelect, layoutNodes, repo]);

  const handleNodeAsk = useCallback((nodeData) => {
    onAskAbout?.(`Explain "${nodeData.label}" in ${repo} in detail — what does it do, what are its key responsibilities, and what other parts of the codebase depend on it?`);
  }, [onAskAbout, repo]);

  // Canvas bounding box
  const allX    = layoutNodes.map(n => n.x + CARD_W + 60);
  const allY    = layoutNodes.map(n => n.y + CARD_H + 80);
  const canvasW = Math.max(...allX, 700);
  const canvasH = Math.max(...allY, 500);

  // Legend: unique types actually present in this diagram
  const presentTypes = useMemo(() => {
    const seen = new Set();
    layoutNodes.forEach(n => { if (n.data?.type) seen.add(n.data.type); });
    return [...seen].sort();
  }, [layoutNodes]);

  return (
    <div className="ec-container">

      {/* ── Canvas ── */}
      <div
        ref={wrapRef}
        className="ec-canvas-wrapper"
        onMouseDown={onMouseDown}
      >
        <div
          className="ec-canvas"
          style={{
            width:           canvasW,
            height:          canvasH,
            transform:       `translate(${xform.x}px, ${xform.y}px) scale(${xform.scale})`,
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
              <marker id="gd-arrow"    markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="rgba(91,143,249,0.35)" />
              </marker>
              <marker id="gd-arrow-hi" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#7DABFF" />
              </marker>
            </defs>

            {layoutEdges.map((edge, i) => {
              const from = getPosFor(edge.source);
              const to   = getPosFor(edge.target);
              if (!from || !to) return null;

              // Highlight an edge when either endpoint is the hovered node OR a direct
              // neighbour. connectedIds only contains neighbours, so we also check
              // hoveredId directly — otherwise edges FROM the hovered node aren't lit.
              const touchesHovered = edge.source === hoveredId || edge.target === hoveredId;
              const isHi  = hoveredId && (touchesHovered || (connectedIds.has(edge.source) && connectedIds.has(edge.target)));
              const isDim = hoveredId && !isHi;
              const d     = bezierPath(from, to);

              // Midpoint for optional edge label
              const mx = (from.x + CARD_W + to.x) / 2;
              const my = (from.y + to.y + CARD_H) / 2;

              return (
                <g key={i}>
                  {/* Wide transparent stroke = clickable hit area.
                      pointerEvents: "stroke" means only the stroke path area fires events,
                      not the bounding box — so cards behind don't get shadowed. */}
                  <path
                    d={d}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={14}
                    style={{ cursor: "pointer", pointerEvents: "stroke" }}
                    onClick={() => handleEdgeClick(edge)}
                  />
                  {/* Visible arrow.
                      "inherits" = solid line (UML: solid with hollow arrow).
                      "uses"     = dashed line (UML: dependency/composition). */}
                  <path
                    d={d}
                    fill="none"
                    stroke={isHi ? "#7DABFF" : "rgba(91,143,249,0.22)"}
                    strokeWidth={isHi ? 2 : 1.5}
                    strokeDasharray={edge.label === "uses" ? "5 3" : undefined}
                    markerEnd={isHi ? "url(#gd-arrow-hi)" : "url(#gd-arrow)"}
                    style={{
                      opacity:    isDim ? 0.1 : 1,
                      transition: "opacity 0.15s, stroke 0.15s, stroke-width 0.15s",
                      pointerEvents: "none",
                    }}
                  />
                  {/* Edge label */}
                  {edge.label && (
                    <text
                      x={mx} y={my}
                      textAnchor="middle"
                      fontSize="10"
                      fontFamily="JetBrains Mono, monospace"
                      fill="#A8C5FF"
                      style={{
                        opacity:    isDim ? 0.08 : 0.75,
                        transition: "opacity 0.15s",
                        pointerEvents: "none",
                        paintOrder: "stroke",
                        stroke: "#09090E",
                        strokeWidth: 3,
                      }}
                    >
                      {edge.label}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {/* ── Diagram node cards ── */}
          {layoutNodes.map(node => (
            <DiagramCard
              key={node.id}
              node={node}
              pos={getPosFor(node.id) ?? { x: node.x, y: node.y }}
              hoveredId={hoveredId}
              connectedIds={connectedIds}
              onSelect={handleNodeSelect}
              onHover={setHoveredId}
              onAsk={onAskAbout ? handleNodeAsk : null}
              onDragStart={onNodeDragStart}
              wasDragged={wasDragged}
            />
          ))}
        </div>
      </div>

      {/* ── Legend bar — same structure as ExploreView's ec-legend ── */}
      {!panelOpen && (
        <div className="ec-legend">
          {presentTypes.map(type => {
            const s = styleFor(type);
            return (
              <span key={type} className="ec-legend-item">
                <span className="ec-legend-dot" style={{ background: s.dot }} />
                {type}
              </span>
            );
          })}
          <span className="ec-legend-hint">
            {layoutNodes.length} nodes · {layoutEdges.length} edges · scroll to zoom · drag to pan · click to explore
          </span>
        </div>
      )}
    </div>
  );
}

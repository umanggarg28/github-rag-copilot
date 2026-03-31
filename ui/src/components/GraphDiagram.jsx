/**
 * GraphDiagram.jsx — React Flow renderer for Architecture, Class, and Data Flow diagrams.
 *
 * Interactions:
 *  - Click a node  → onNodeSelect(node) — opens NodeDetailPanel inline
 *  - Click an edge → onEdgeSelect(edge) — opens NodeDetailPanel with edge context
 *  - Hover a node  → dims unrelated nodes/edges, highlights direct connections
 */

import { useEffect, useMemo, useCallback, useState, createContext, useContext } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  MarkerType,
  Position,
  Handle,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// ── Hover context — lets DiagramNode read hover state without prop drilling ──
// onAsk is included so DiagramNode can trigger the chat without prop drilling
const HoverCtx = createContext({ hoveredId: null, connectedIds: new Set(), onAsk: null });

// ── Type colours ──────────────────────────────────────────────────────────────
const TYPE_STYLE = {
  module:    { border: "#B45309", glow: "rgba(180,83,9,0.35)",    dot: "#fbbf24" },
  class:     { border: "#7C3AED", glow: "rgba(124,58,237,0.35)",  dot: "#a78bfa" },
  abstract:  { border: "#6D28D9", glow: "rgba(109,40,217,0.35)",  dot: "#c4b5fd" },
  mixin:     { border: "#5B21B6", glow: "rgba(91,33,182,0.35)",   dot: "#ddd6fe" },
  service:   { border: "#0D9488", glow: "rgba(13,148,136,0.35)",  dot: "#2dd4bf" },
  database:  { border: "#B45309", glow: "rgba(180,83,9,0.35)",    dot: "#fbbf24" },
  external:  { border: "#374151", glow: "rgba(55,65,81,0.35)",    dot: "#9CA3AF" },
  input:     { border: "#0D9488", glow: "rgba(13,148,136,0.35)",  dot: "#2dd4bf" },
  transform: { border: "#4F46E5", glow: "rgba(79,70,229,0.35)",   dot: "#818CF8" },
  output:    { border: "#BE185D", glow: "rgba(190,24,93,0.35)",   dot: "#f472b6" },
};
const FALLBACK_STYLE = { border: "#4F46E5", glow: "rgba(79,70,229,0.35)", dot: "#818CF8" };
function styleFor(type) { return TYPE_STYLE[type] || FALLBACK_STYLE; }

// ── File icon — same SVG used in ExploreView's ec-file pill ──────────────────
function FileIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.5, flexShrink: 0 }}>
      <path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 8.75 4.25V1.5Zm6.75.56v2.19c0 .138.112.25.25.25h2.19Z"/>
    </svg>
  );
}

// ── Custom node — mirrors ExploreView's ConceptCard styling ──────────────────
function DiagramNode({ id, data }) {
  const { hoveredId, connectedIds, onAsk } = useContext(HoverCtx);

  const dim       = hoveredId && hoveredId !== id && !connectedIds.has(id);
  const highlight = hoveredId && connectedIds.has(id);

  const s = styleFor(data.type);

  return (
    <div
      className={`ec-card${dim ? " ec-dimmed" : ""}`}
      style={{
        width: 210,
        borderColor: highlight ? s.dot : undefined,
        boxShadow: highlight ? `0 0 0 1.5px ${s.dot}, 0 8px 32px ${s.glow}` : undefined,
        position: "relative",
      }}
    >
      {/* Colour-coded top accent bar */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0,
        height: 2.5, borderRadius: "10px 10px 0 0",
        background: highlight ? s.dot : s.border, opacity: highlight ? 1 : 0.7,
      }} />

      <Handle type="target" position={Position.Left}  style={{ background: s.border, width: 7, height: 7, border: "2px solid #0e0e1a" }} />
      <Handle type="source" position={Position.Right} style={{ background: s.border, width: 7, height: 7, border: "2px solid #0e0e1a" }} />

      {/* Top row: type tag */}
      <div className="ec-card-top" style={{ marginBottom: 6 }}>
        <span className="ec-type-tag" style={{ color: s.dot, borderColor: `${s.dot}44` }}>
          {data.type}
        </span>
      </div>

      {/* Label */}
      <div className="ec-name">{data.label}</div>

      {/* File path — with icon, matching ExploreView */}
      {data.file && (
        <div className="ec-file" style={{ marginTop: 3 }}>
          <FileIcon />
          {data.file}
        </div>
      )}

      {/* Description */}
      {data.description && (
        <div className="ec-subtitle" style={{ marginTop: 4 }}>
          {data.description.length > 90 ? data.description.slice(0, 88) + "…" : data.description}
        </div>
      )}

      {/* Method pills — reuse ec-item style */}
      {data.items?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginTop: 6 }}>
          {data.items.slice(0, 4).map((item, i) => (
            <code key={i} className="ec-item">{item}</code>
          ))}
        </div>
      )}

      {/* Ask button — matches ExploreView's ec-ask, stopPropagation avoids double-firing node click */}
      {onAsk && (
        <button
          className="ec-ask"
          onClick={e => { e.stopPropagation(); onAsk(data); }}
        >
          Ask about this →
        </button>
      )}
    </div>
  );
}

const nodeTypes = { diagramNode: DiagramNode };

// ── Layout algorithm ──────────────────────────────────────────────────────────
function computeLayout(rawNodes, rawEdges) {
  const NODE_W = 230;
  const NODE_H = 160;
  const COL_GAP = 90;
  const ROW_GAP = 28;

  if (!rawNodes?.length) return { nodes: [], edges: [] };

  const inDegree = {};
  const outEdges = {};
  rawNodes.forEach(n => { inDegree[n.id] = 0; outEdges[n.id] = []; });
  rawEdges.forEach(e => {
    if (inDegree[e.target] !== undefined) inDegree[e.target]++;
    if (outEdges[e.source]) outEdges[e.source].push(e.target);
  });

  // BFS topological depth (column)
  const depth = {};
  rawNodes.forEach(n => { depth[n.id] = 0; });
  const queue = rawNodes.filter(n => inDegree[n.id] === 0).map(n => n.id);
  const tmpIn = { ...inDegree };
  const visited = new Set(queue);
  let head = 0;
  while (head < queue.length) {
    const id = queue[head++];
    for (const neighbor of (outEdges[id] || [])) {
      depth[neighbor] = Math.max(depth[neighbor], depth[id] + 1);
      tmpIn[neighbor]--;
      if (tmpIn[neighbor] === 0 && !visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }

  // Group by column
  const cols = {};
  rawNodes.forEach(n => {
    const c = depth[n.id];
    if (!cols[c]) cols[c] = [];
    cols[c].push(n);
  });

  // Split any column that exceeds MAX_PER_COL into sub-columns
  const MAX_PER_COL = 4;
  const colKeys = Object.keys(cols).map(Number).sort((a, b) => a - b);
  const expandedCols = {};
  let ci = 0;
  colKeys.forEach(col => {
    const nodes = cols[col];
    if (nodes.length > MAX_PER_COL) {
      for (let start = 0; start < nodes.length; start += MAX_PER_COL) {
        expandedCols[ci++] = nodes.slice(start, start + MAX_PER_COL);
      }
    } else {
      expandedCols[ci++] = nodes;
    }
  });
  Object.keys(cols).forEach(k => delete cols[k]);
  Object.assign(cols, expandedCols);

  // Assign pixel positions
  const posMap = {};
  const finalColKeys = Object.keys(cols).map(Number).sort((a, b) => a - b);
  finalColKeys.forEach((col, ci) => {
    const colNodes = cols[col];
    const totalH = colNodes.length * NODE_H + (colNodes.length - 1) * ROW_GAP;
    const startY = -totalH / 2;
    colNodes.forEach((n, ri) => {
      posMap[n.id] = {
        x: ci * (NODE_W + COL_GAP),
        y: startY + ri * (NODE_H + ROW_GAP),
      };
    });
  });

  const nodes = rawNodes.map(n => ({
    id: n.id,
    type: "diagramNode",
    position: posMap[n.id] || { x: 0, y: 0 },
    data: {
      label: n.label, type: n.type, file: n.file,
      description: n.description, items: n.items || n.methods || [],
    },
  }));

  const edges = rawEdges.map((e, i) => ({
    id: `e-${i}`,
    source: e.source,
    target: e.target,
    // "default" = cubic bezier — matches ExploreView's SVG bezier arrows
    type: "default",
    label: e.label || "",
    animated: false,
    style: { stroke: "rgba(139,92,246,0.35)", strokeWidth: 1.5 },
    labelStyle: { fill: "#a78bfa", fontSize: 10, fontFamily: "JetBrains Mono, monospace" },
    labelBgStyle: { fill: "#1a1a2e", fillOpacity: 0.9 },
    markerEnd: { type: MarkerType.ArrowClosed, color: "rgba(139,92,246,0.5)", width: 12, height: 12 },
    // Store raw edge data for the detail panel
    data: { sourceLabel: e.source, targetLabel: e.target, relationLabel: e.label || "" },
  }));

  return { nodes, edges };
}

// ── Default edge style (used when no hover) ───────────────────────────────────
// Colors match ExploreView's SVG arrow strokes for visual consistency
const DEFAULT_EDGE_STYLE = { stroke: "rgba(139,92,246,0.35)", strokeWidth: 1.5 };
const HOVER_EDGE_STYLE   = { stroke: "#a78bfa",               strokeWidth: 2 };
const DIM_EDGE_STYLE     = { stroke: "rgba(139,92,246,0.08)", strokeWidth: 1 };

// ── Main component ────────────────────────────────────────────────────────────
export default function GraphDiagram({ data, onNodeSelect, onEdgeSelect, onAskAbout, repo, panelOpen }) {
  const { nodes: layoutNodes, edges: layoutEdges } = useMemo(
    () => computeLayout(data?.nodes || [], data?.edges || []),
    [data]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutEdges);
  const [hoveredId, setHoveredId] = useState(null);

  useEffect(() => { setNodes(layoutNodes); setEdges(layoutEdges); }, [layoutNodes, layoutEdges]);

  // Compute which node IDs are directly connected to hoveredId
  const connectedIds = useMemo(() => {
    if (!hoveredId) return new Set();
    const ids = new Set();
    layoutEdges.forEach(e => {
      if (e.source === hoveredId) ids.add(e.target);
      if (e.target === hoveredId) ids.add(e.source);
    });
    return ids;
  }, [hoveredId, layoutEdges]);

  // Update edge styles on hover to highlight/dim
  useEffect(() => {
    setEdges(prev => prev.map(e => {
      if (!hoveredId) return { ...e, style: DEFAULT_EDGE_STYLE, animated: false };
      const connected = e.source === hoveredId || e.target === hoveredId;
      return {
        ...e,
        style: connected ? HOVER_EDGE_STYLE : DIM_EDGE_STYLE,
        animated: connected,
      };
    }));
  }, [hoveredId]);

  const onNodeMouseEnter = useCallback((_, node) => setHoveredId(node.id), []);
  const onNodeMouseLeave = useCallback(() => setHoveredId(null), []);

  const onNodeClick = useCallback((_, node) => {
    if (!onNodeSelect) return;
    onNodeSelect({
      kind: "node",
      id: node.id,
      label: node.data.label,
      type: node.data.type,
      file: node.data.file,
      description: node.data.description,
      items: node.data.items || [],
    });
  }, [onNodeSelect]);

  const onEdgeClick = useCallback((_, edge) => {
    if (!onEdgeSelect) return;
    // Find source/target labels from the layout nodes
    const srcNode = layoutNodes.find(n => n.id === edge.source);
    const tgtNode = layoutNodes.find(n => n.id === edge.target);
    const srcLabel = srcNode?.data?.label || edge.source;
    const tgtLabel = tgtNode?.data?.label || edge.target;
    onEdgeSelect({
      kind: "edge",
      id: edge.id,
      label: `${srcLabel} → ${tgtLabel}`,
      type: "edge",
      file: null,
      description: edge.label ? `Relationship: "${edge.label}"` : "",
      items: [],
      autoQuestion: `In ${repo}, why does "${srcLabel}" ${edge.label || "depend on"} "${tgtLabel}"? What specifically does it use from it, and what would break if this dependency were removed?`,
    });
  }, [onEdgeSelect, layoutNodes, repo]);

  // onAsk: clicking "Ask about this →" on a node card jumps straight to chat
  const handleNodeAsk = useCallback((nodeData) => {
    onAskAbout?.(`Explain "${nodeData.label}" in ${repo} in detail — what does it do, what are its key responsibilities, and what other parts of the codebase depend on it?`);
  }, [onAskAbout, repo]);

  return (
    <HoverCtx.Provider value={{ hoveredId, connectedIds, onAsk: onAskAbout ? handleNodeAsk : null }}>
      <div style={{ width: "100%", height: "100%", background: "#0d0d1a" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onNodeMouseEnter={onNodeMouseEnter}
          onNodeMouseLeave={onNodeMouseLeave}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.25, maxZoom: 0.9 }}
          minZoom={0.2}
          maxZoom={1.5}
          defaultEdgeOptions={{ animated: false }}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="rgba(79,70,229,0.08)" gap={28} size={1} />
          <Controls
            style={{
              background: "#1a1a2e",
              border: "1px solid rgba(79,70,229,0.3)",
              borderRadius: 8,
            }}
            showInteractive={false}
          />
        </ReactFlow>

        {/* Interaction hint — hidden when panel is open to reduce noise */}
        {!panelOpen && (
          <div style={{
            position: "absolute", bottom: 10, left: "50%", transform: "translateX(-50%)",
            fontSize: 9.5, color: "rgba(107,114,128,0.7)",
            fontFamily: "JetBrains Mono, monospace",
            pointerEvents: "none", letterSpacing: "0.04em",
            background: "rgba(13,13,26,0.7)", padding: "3px 10px", borderRadius: 4,
          }}>
            Click node to explore · Click arrow to explain connection · Hover to highlight
          </div>
        )}
      </div>
    </HoverCtx.Provider>
  );
}

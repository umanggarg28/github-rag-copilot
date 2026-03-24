/**
 * CodeGraph.jsx — Interactive D3 force-directed call graph.
 *
 * ═══════════════════════════════════════════════════════════
 * HOW D3 FORCE SIMULATION WORKS
 * ═══════════════════════════════════════════════════════════
 *
 * D3's force simulation is a physics engine for graphs.
 * It runs a loop of "ticks" where each tick applies forces to nodes,
 * moving them closer to equilibrium. The forces we use:
 *
 *   forceLink    — edges act like springs: pull connected nodes together.
 *                  distance=80 means they want to be 80px apart.
 *
 *   forceManyBody — nodes repel each other (like charged particles).
 *                   strength=-200 means moderate repulsion.
 *
 *   forceCenter   — a gentle pull toward the center of the viewport,
 *                   preventing the whole graph from drifting off-screen.
 *
 *   forceCollide  — prevents nodes from overlapping.
 *                   radius = node visual radius + 4px padding.
 *
 * On each tick, d3 updates node.x and node.y. We tell D3 to directly
 * update the SVG elements' positions via selection.attr("cx", d => d.x).
 * This is faster than React re-rendering on every tick (60fps → 60 renders/s
 * would be too slow; D3 DOM manipulation bypasses the React diffing overhead).
 *
 * ═══════════════════════════════════════════════════════════
 * REACT + D3 INTEGRATION PATTERN
 * ═══════════════════════════════════════════════════════════
 *
 * React owns the container <svg> element (via useRef).
 * D3 owns the contents (nodes, edges) after initial render.
 *
 * When graph data changes:
 *   1. useEffect fires with new nodes/edges
 *   2. We clear the SVG contents
 *   3. D3 creates new elements and starts the simulation
 *   4. Simulation runs, updating positions each tick
 *   5. On cleanup, simulation.stop() is called to prevent memory leaks
 *
 * ═══════════════════════════════════════════════════════════
 * NODE ENCODING
 * ═══════════════════════════════════════════════════════════
 *
 *   Color:   class → terracotta, function → blue, module → grey
 *   Size:    r = 5 + caller_count * 2  (more callers = larger node)
 *            This makes "hub" functions visually prominent.
 *   Opacity: 0.85 default, 1.0 on hover
 *
 * ═══════════════════════════════════════════════════════════
 * INTERACTIONS
 * ═══════════════════════════════════════════════════════════
 *
 *   Zoom/pan:     d3.zoom() on the SVG — scroll to zoom, drag to pan
 *   Hover:        shows a tooltip with name + filepath
 *   Click:        onNodeClick(node) — triggers "Ask about this" in App.jsx
 *   Drag node:    d3.drag() on each node — repositions while keeping simulation running
 */

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

// ── Color scheme ──────────────────────────────────────────────────────────────
const COLORS = {
  class:    "#c97a60",   // terracotta
  function: "#5c8fbd",   // muted blue
  module:   "#7a7a7a",   // grey
};

const EDGE_COLOR    = "#3a3a4a";
const EDGE_HOVER    = "#58a6ff";
const BG_COLOR      = "#0d0d14";
const TOOLTIP_BG    = "#1a1a2e";

// Node radius: base + bonus for each caller (hub detection)
function nodeRadius(d) {
  return Math.max(5, 5 + d.caller_count * 1.8);
}

export default function CodeGraph({ repo, onAskAbout }) {
  const svgRef       = useRef(null);
  const [tooltip, setTooltip] = useState(null);   // {x, y, node}
  const [stats, setStats]     = useState(null);   // {node_count, edge_count}
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (!repo || !svgRef.current) return;

    // cancelled flag prevents StrictMode's double-invoke from rendering twice.
    // React StrictMode in dev runs effects twice (mount → cleanup → mount) to
    // detect side effects. Without this flag, both fetches complete and both
    // call _renderGraph, appending duplicate SVG elements.
    let cancelled = false;

    setLoading(true);
    setError(null);
    d3.select(svgRef.current).selectAll("*").remove();

    import("../api").then(({ fetchGraph }) =>
      fetchGraph(repo)
        .then(data => {
          if (cancelled) return;
          setStats(data.stats);
          setLoading(false);
          if (data.nodes.length === 0) {
            setError("No function/class nodes found. Re-ingest the repo to get call data.");
            return;
          }
          _renderGraph(svgRef.current, data, setTooltip, onAskAbout);
        })
        .catch(err => {
          if (cancelled) return;
          setLoading(false);
          setError(err.message);
        })
    );

    return () => {
      cancelled = true;
      d3.select(svgRef.current).selectAll("*").remove();
    };
  }, [repo]);

  return (
    <div className="graph-container">
      {/* ── Header ── */}
      <div className="graph-header">
        <span className="graph-title">Code Call Graph — {repo}</span>
        {stats && (
          <span className="graph-stats">
            {stats.node_count} functions · {stats.edge_count} call edges
            {stats.edge_count === 0 && " · re-ingest repo to see edges"}
          </span>
        )}
      </div>

      {/* ── Legend ── */}
      <div className="graph-legend">
        {Object.entries(COLORS).map(([type, color]) => (
          <span key={type} className="legend-item">
            <span className="legend-dot" style={{ background: color }} />
            {type}
          </span>
        ))}
        <span className="legend-item" style={{ marginLeft: 16, color: "#888", fontSize: 11 }}>
          Larger = called more often · Click node to ask about it
        </span>
      </div>

      {/* ── Canvas ── */}
      <div className="graph-canvas">
        {loading && (
          <div className="graph-loading">
            <span className="spinner" /> Building graph…
          </div>
        )}
        {error && <div className="graph-error">{error}</div>}
        <svg ref={svgRef} width="100%" height="100%" />
      </div>

      {/* ── Tooltip ── */}
      {tooltip && (
        <div
          className="graph-tooltip"
          style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}
        >
          <div className="graph-tooltip-name">{tooltip.node.name}</div>
          <div className="graph-tooltip-path">{tooltip.node.filepath}</div>
          <div className="graph-tooltip-meta">
            lines {tooltip.node.start_line}–{tooltip.node.end_line}
            {tooltip.node.caller_count > 0 && ` · called by ${tooltip.node.caller_count}`}
          </div>
          <button
            className="graph-tooltip-ask"
            onClick={() => onAskAbout?.(`Explain what \`${tooltip.node.name}\` does in \`${tooltip.node.filepath}\``)}
          >
            Ask about this →
          </button>
        </div>
      )}
    </div>
  );
}


// ── D3 rendering (runs outside React's render cycle) ────────────────────────

function _renderGraph(svgEl, data, setTooltip, onAskAbout) {
  const { nodes: rawNodes, edges: rawEdges } = data;

  // D3 mutates node objects (adds x, y, vx, vy) — work on copies
  const nodes = rawNodes.map(n => ({ ...n }));
  const edges = rawEdges.map(e => ({ ...e }));

  const svg    = d3.select(svgEl);
  const width  = svgEl.parentElement?.clientWidth  || 900;
  const height = svgEl.parentElement?.clientHeight || 600;

  svg.attr("viewBox", `0 0 ${width} ${height}`);

  // ── Zoom/pan layer ──────────────────────────────────────────────────────────
  // Everything goes inside a <g> that zoom transforms.
  // This lets us zoom/pan without moving the SVG element itself.
  const zoomGroup = svg.append("g");

  svg.call(
    d3.zoom()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => zoomGroup.attr("transform", event.transform))
  );

  // ── Arrow marker for directed edges ────────────────────────────────────────
  svg.append("defs").append("marker")
    .attr("id",         "arrow")
    .attr("viewBox",    "0 -4 8 8")
    .attr("refX",       14)
    .attr("refY",       0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient",     "auto")
    .append("path")
    .attr("d",    "M0,-4L8,0L0,4")
    .attr("fill", EDGE_COLOR);

  // ── Force simulation ────────────────────────────────────────────────────────
  const simulation = d3.forceSimulation(nodes)
    .force("link",    d3.forceLink(edges).id(d => d.id).distance(90).strength(0.4))
    .force("charge",  d3.forceManyBody().strength(-220))
    .force("center",  d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(d => nodeRadius(d) + 6));

  // ── Edges (lines) ───────────────────────────────────────────────────────────
  const link = zoomGroup.append("g")
    .selectAll("line")
    .data(edges)
    .join("line")
    .attr("stroke",       EDGE_COLOR)
    .attr("stroke-width", 1.2)
    .attr("stroke-opacity", 0.6)
    .attr("marker-end",   "url(#arrow)");

  // ── Nodes (circles + drag) ──────────────────────────────────────────────────
  const node = zoomGroup.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r",    d => nodeRadius(d))
    .attr("fill", d => COLORS[d.chunk_type] || COLORS.function)
    .attr("fill-opacity", 0.85)
    .attr("stroke",       "#ffffff22")
    .attr("stroke-width", 1)
    .style("cursor", "pointer")
    .call(
      // ── Drag handler ─────────────────────────────────────────────────────
      // While dragging, we "fix" the node position (fx/fy) so the simulation
      // keeps it where the user dropped it. Releasing un-fixes it.
      d3.drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end",   (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
    )
    .on("mouseover", (event, d) => {
      d3.select(event.currentTarget).attr("fill-opacity", 1).attr("stroke", "#ffffff88");
      // Get position relative to the graph-container div
      const rect = svgEl.parentElement.getBoundingClientRect();
      setTooltip({ x: event.clientX - rect.left, y: event.clientY - rect.top, node: d });
    })
    .on("mousemove", (event) => {
      const rect = svgEl.parentElement.getBoundingClientRect();
      setTooltip(prev => prev ? { ...prev, x: event.clientX - rect.left, y: event.clientY - rect.top } : null);
    })
    .on("mouseout",  (event) => {
      d3.select(event.currentTarget).attr("fill-opacity", 0.85).attr("stroke", "#ffffff22");
      setTooltip(null);
    })
    .on("click", (event, d) => {
      event.stopPropagation();
      onAskAbout?.(`Explain what \`${d.name}\` does in \`${d.filepath}\``);
    });

  // ── Labels (short names, only for hub nodes to avoid clutter) ──────────────
  const label = zoomGroup.append("g")
    .selectAll("text")
    .data(nodes.filter(n => n.caller_count >= 2))  // only label frequently-called nodes
    .join("text")
    .text(d => d.name.split(".").pop())             // leaf name only
    .attr("font-size",   10)
    .attr("font-family", "JetBrains Mono, monospace")
    .attr("fill",        "#ccccdd")
    .attr("text-anchor", "middle")
    .attr("dy",          d => -nodeRadius(d) - 3)
    .style("pointer-events", "none")               // don't intercept mouse events
    .style("user-select",    "none");

  // ── Tick: update positions on every simulation step ────────────────────────
  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);

    label
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });

  // Stop simulation after it cools (saves CPU)
  simulation.on("end", () => simulation.stop());
}

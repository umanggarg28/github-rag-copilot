/**
 * SemanticMap.jsx — 2D projection of code chunk embeddings.
 *
 * ═══════════════════════════════════════════════════════════════
 * WHAT THIS SHOWS
 * ═══════════════════════════════════════════════════════════════
 *
 * Every function and class in the indexed repo was embedded into a
 * 768-dimensional vector space. The backend uses PCA to project these
 * vectors down to 2D coordinates, preserving as much variance as possible.
 *
 * The result: a spatial map where semantically similar code clusters together.
 * Functions that do similar things (math ops, I/O, testing utilities) will
 * appear near each other — even if they live in different files.
 *
 * This gives an "at a glance" structural understanding that file trees and
 * call graphs can't provide.
 *
 * ═══════════════════════════════════════════════════════════════
 * THE RETRIEVAL OVERLAY (the "wow" feature)
 * ═══════════════════════════════════════════════════════════════
 *
 * After a RAG query, the sources returned by the AI are highlighted
 * on the map with a glowing ring. This makes the invisible retrieval
 * step visible: users can literally see WHERE in the semantic space
 * the AI went to find the answer.
 *
 * "Why did the AI retrieve those three files?"
 * → Because those dots are the closest to where your question lands.
 *
 * ═══════════════════════════════════════════════════════════════
 * INTERACTIONS
 * ═══════════════════════════════════════════════════════════════
 *
 *   Zoom/pan:   d3.zoom() — scroll to zoom, drag to pan
 *   Hover:      tooltip with name, filepath, lines
 *   Click:      onAskAbout() — pre-fills the chat input
 *   Highlighted: glowing rings on chunks used in the last RAG answer
 */

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

// ── Color encoding ────────────────────────────────────────────────────────────
// Nodes are colored by semantic cluster (K-means assignment from backend).
// This groups conceptually similar code visually, regardless of file.
// Using per-file colors would produce 40+ colors for a large repo — illegible.
const BASE_RADIUS  = 5;
const CLASS_RADIUS = 8;

function nodeRadius(d) {
  return d.chunk_type === "class" ? CLASS_RADIUS : BASE_RADIUS;
}

export default function SemanticMap({ repo, highlightedSources, onAskAbout }) {
  const svgRef      = useRef(null);
  const clustersRef = useRef([]);          // color lookup for cluster IDs
  const [tooltip, setTooltip] = useState(null);
  const [stats,   setStats]   = useState(null);
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  // Match sources from RAG to node keys
  const highlightSet = new Set(
    (highlightedSources || []).map(s => `${s.filepath}::${s.name}`)
  );

  useEffect(() => {
    if (!repo || !svgRef.current) return;
    let cancelled = false;

    setLoading(true);
    setError(null);
    d3.select(svgRef.current).selectAll("*").remove();

    import("../api").then(({ fetchSemanticMap }) =>
      fetchSemanticMap(repo)
        .then(data => {
          if (cancelled) return;
          setStats(data.stats);
          setClusters(data.clusters || []);
          clustersRef.current = data.clusters || [];
          setLoading(false);
          if (!data.nodes || data.nodes.length === 0) {
            setError("No chunks found. Re-ingest the repo first.");
            return;
          }
          _render(svgRef.current, data.nodes, data.clusters || [], highlightSet, setTooltip, onAskAbout);
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

  // When highlighted sources change, update node styles without re-fetching
  useEffect(() => {
    if (!svgRef.current) return;
    const clusterColors = Object.fromEntries(
      clustersRef.current.map(c => [c.id, c.color])
    );
    d3.select(svgRef.current).selectAll("circle.map-node")
      .attr("fill", d => {
        const key = `${d.filepath}::${d.name}`;
        return highlightSet.has(key) ? "#FFFFFF" : (clusterColors[d.cluster_id] || "#6366F1");
      })
      .attr("fill-opacity", d => {
        const key = `${d.filepath}::${d.name}`;
        if (highlightSet.size === 0) return 0.75;
        return highlightSet.has(key) ? 1.0 : 0.15;
      })
      .attr("r", d => {
        const key = `${d.filepath}::${d.name}`;
        return highlightSet.has(key) ? nodeRadius(d) + 4 : nodeRadius(d);
      })
      .attr("filter", d => {
        const key = `${d.filepath}::${d.name}`;
        return highlightSet.has(key) ? "url(#glow)" : null;
      });
  }, [highlightedSources]);

  return (
    <div className="graph-container">
      {/* Header */}
      <div className="graph-header">
        <span className="graph-title">Semantic Map — {repo}</span>
        {stats && (
          <span className="graph-stats">
            {stats.node_count} chunks · similar code clusters together
          </span>
        )}
      </div>

      {/* Legend — cluster color chips */}
      <div className="graph-legend">
        {clusters.map(c => (
          <span key={c.id} className="legend-item">
            <span className="legend-dot" style={{ background: c.color }} />
            {c.label}
          </span>
        ))}
        {highlightedSources?.length > 0 && (
          <span className="legend-item" style={{ marginLeft: 8 }}>
            <span className="legend-dot" style={{ background: "#fff", boxShadow: "0 0 6px #fff" }} />
            retrieved ({highlightedSources.length})
          </span>
        )}
        <span className="legend-item" style={{ marginLeft: 8, color: "#888", fontSize: 11 }}>
          Click node to ask about it
        </span>
      </div>

      {/* Canvas */}
      <div className="graph-canvas">
        {loading && (
          <div className="graph-loading">
            <span className="spinner" /> Projecting embeddings…
          </div>
        )}
        {error && <div className="graph-error">{error}</div>}
        <svg ref={svgRef} width="100%" height="100%" />
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="graph-tooltip"
          style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}
        >
          <div className="graph-tooltip-name">{tooltip.node.name}</div>
          <div className="graph-tooltip-path">{tooltip.node.filepath}</div>
          <div className="graph-tooltip-meta">
            {tooltip.node.chunk_type} · lines {tooltip.node.start_line}–{tooltip.node.end_line}
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


// ── D3 rendering ──────────────────────────────────────────────────────────────

function _render(svgEl, nodes, clusters, highlightSet, setTooltip, onAskAbout) {
  // Build cluster_id → color lookup
  const clusterColor = Object.fromEntries(clusters.map(c => [c.id, c.color]));
  const svg    = d3.select(svgEl);
  const width  = svgEl.parentElement?.clientWidth  || 900;
  const height = svgEl.parentElement?.clientHeight || 600;

  // The backend normalises node coords to [50, 950]. We map that into the
  // viewport, centering and scaling to fit.
  const padding = 40;
  const xScale = d3.scaleLinear().domain([0, 1000]).range([padding, width  - padding]);
  const yScale = d3.scaleLinear().domain([0, 1000]).range([padding, height - padding]);

  svg.attr("viewBox", `0 0 ${width} ${height}`);

  // ── Glow filter for highlighted nodes ──────────────────────────────────────
  const defs = svg.append("defs");
  const filter = defs.append("filter").attr("id", "glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
  filter.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "blur");
  const merge = filter.append("feMerge");
  merge.append("feMergeNode").attr("in", "blur");
  merge.append("feMergeNode").attr("in", "SourceGraphic");

  // ── Zoom/pan layer ─────────────────────────────────────────────────────────
  const zoomGroup = svg.append("g");
  svg.call(
    d3.zoom()
      .scaleExtent([0.3, 8])
      .on("zoom", e => zoomGroup.attr("transform", e.transform))
  );

  // ── Draw nodes ─────────────────────────────────────────────────────────────
  zoomGroup.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", "map-node")
    .attr("cx", d => xScale(d.x))
    .attr("cy", d => yScale(d.y))
    .attr("r",  d => {
      const key = `${d.filepath}::${d.name}`;
      return highlightSet.has(key) ? nodeRadius(d) + 4 : nodeRadius(d);
    })
    .attr("fill", d => {
      const key = `${d.filepath}::${d.name}`;
      return highlightSet.has(key) ? "#FFFFFF" : (clusterColor[d.cluster_id] || "#6366F1");
    })
    .attr("fill-opacity", d => {
      const key = `${d.filepath}::${d.name}`;
      if (highlightSet.size === 0) return 0.75;
      return highlightSet.has(key) ? 1.0 : 0.15;
    })
    .attr("filter", d => {
      const key = `${d.filepath}::${d.name}`;
      return highlightSet.has(key) ? "url(#glow)" : null;
    })
    .attr("stroke", d => {
      const key = `${d.filepath}::${d.name}`;
      return highlightSet.has(key) ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.06)";
    })
    .attr("stroke-width", d => {
      const key = `${d.filepath}::${d.name}`;
      return highlightSet.has(key) ? 2 : 0.5;
    })
    .style("cursor", "pointer")
    .on("mouseover", (event, d) => {
      d3.select(event.currentTarget)
        .attr("fill-opacity", 1)
        .attr("r", nodeRadius(d) + 2);
      const rect = svgEl.parentElement.getBoundingClientRect();
      setTooltip({ x: event.clientX - rect.left, y: event.clientY - rect.top, node: d });
    })
    .on("mousemove", event => {
      const rect = svgEl.parentElement.getBoundingClientRect();
      setTooltip(prev => prev ? { ...prev, x: event.clientX - rect.left, y: event.clientY - rect.top } : null);
    })
    .on("mouseout", (event, d) => {
      const key = `${d.filepath}::${d.name}`;
      const isHl = highlightSet.has(key);
      d3.select(event.currentTarget)
        .attr("fill-opacity", isHl ? 1.0 : (highlightSet.size === 0 ? 0.75 : 0.15))
        .attr("r", isHl ? nodeRadius(d) + 4 : nodeRadius(d));
      setTooltip(null);
    })
    .on("click", (event, d) => {
      event.stopPropagation();
      onAskAbout?.(`Explain what \`${d.name}\` does in \`${d.filepath}\``);
    });

  // ── Labels: only for classes and top nodes ─────────────────────────────────
  zoomGroup.append("g")
    .selectAll("text")
    .data(nodes.filter(n => n.chunk_type === "class"))
    .join("text")
    .attr("x", d => xScale(d.x))
    .attr("y", d => yScale(d.y) - CLASS_RADIUS - 4)
    .text(d => d.name.split(".").pop())
    .attr("font-size", 9)
    .attr("font-family", "JetBrains Mono, monospace")
    .attr("fill", d => clusterColor[d.cluster_id] || "#A78BFA")
    .attr("text-anchor", "middle")
    .style("pointer-events", "none")
    .style("user-select", "none");
}

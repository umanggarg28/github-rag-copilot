/**
 * MermaidBlock.jsx — Renders mermaid syntax as a clean SVG diagram inline in chat.
 *
 * Used when the agent outputs a ```diagram``` fenced block (which contains Mermaid syntax).
 * Renders directly via mermaid.js — reliable SVG output, same approach used by ChatGPT/Gemini.
 * Includes an expand-to-modal button with scroll-wheel zoom.
 *
 * Modal uses ReactDOM.createPortal so it attaches to document.body — this prevents
 * position:fixed from being broken by parent CSS transforms (chat message animations).
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { createPortal } from "react-dom";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: false,
  theme: "dark",
  themeVariables: {
    background:         "#09090e",
    primaryColor:       "#151522",   /* --surface-2 */
    primaryBorderColor: "rgba(91,143,249,0.35)",
    primaryTextColor:   "#e2e8f8",   /* --text */
    lineColor:          "rgba(91,143,249,0.50)",
    secondaryColor:     "#0f0f18",   /* --surface */
    tertiaryColor:      "#1c1c2e",   /* --surface-3 */
    edgeLabelBackground: "#09090e",
    clusterBkg:         "#1c1c2e",
    clusterBorder:      "rgba(91,143,249,0.18)",
    titleColor:         "#a8c5ff",   /* --accent-soft */
    nodeTextColor:      "#e2e8f8",
  },
  flowchart: { curve: "basis", padding: 20 },
  sequence:  { useMaxWidth: false },
  class:     { useMaxWidth: false },
});

let _idCounter = 0;

// Auto-quote flowchart node labels that contain characters Mermaid's parser chokes on.
// E.g. D[Call backward()]  →  D["Call backward()"]
//      K[grad += x * y]    →  K["grad += x * y"]
// Only rewrites unquoted labels — already-quoted ones are left alone.
// Targets bracket shapes: [label] only (the most common, and the most error-prone).
function sanitizeMermaid(text) {
  // Special chars that cause parse errors when unquoted inside [ ]
  // Match: opening bracket, content without quotes that contains at least one special char, closing bracket
  return text.replace(
    /\[([^"\]]*[()=+\-*%<>&|#;][^"\]]*)\]/g,
    (_, content) => `["${content.replace(/"/g, "'")}"]`
  );
}

function Diagram({ mermaid: rawText }) {
  const text = sanitizeMermaid(rawText);
  const [svg, setSvg]     = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setSvg(null);
    setError(null);

    const id = `mermaid-${++_idCounter}`;
    // parse() rejects on bad syntax — prevents mermaid from rendering a bomb SVG
    mermaid.parse(text)
      .then(() => mermaid.render(id, text))
      .then(({ svg: rendered }) => {
          if (cancelled) return;
          // Mermaid sets max-width but no explicit width, so browsers shrink the SVG
          // to fit its container. Fix: extract the natural pixel width from max-width
          // and set it as an explicit width attribute so the container scrolls instead.
          const mw = rendered.match(/max-width\s*:\s*([\d.]+)px/i);
          const naturalWidth = mw ? Math.ceil(parseFloat(mw[1])) : null;
          let fixed = rendered.replace(/max-width\s*:\s*[^;'"]+;?\s*/gi, "");
          if (naturalWidth) {
            fixed = fixed.replace(/^<svg /, `<svg width="${naturalWidth}" `);
          }
          setSvg(fixed);
        })
      .catch((err)             => { if (!cancelled) setError(String(err)); });

    return () => { cancelled = true; };
  }, [text]);

  if (error) return (
    <div style={{ padding: "8px 12px" }}>
      <div style={{
        fontSize: 11, color: "rgba(91,143,249,0.80)", marginBottom: 6,
        fontFamily: "var(--mono)",
      }}>
        Diagram syntax error — {error.split("\n")[0].slice(0, 120)}
      </div>
      <pre style={{
        fontSize: 12, opacity: 0.5, margin: 0, padding: "8px 10px",
        background: "#06060F", borderRadius: 6,
        border: "1px solid rgba(255,255,255,0.06)", overflow: "auto",
      }}>
        {text}
      </pre>
    </div>
  );

  if (!svg) return (
    <div style={{ padding: 16, color: "var(--muted)", fontSize: 12, display: "flex", alignItems: "center", gap: 8 }}>
      <span className="spinner" /> Rendering diagram…
    </div>
  );

  return (
    <div
      dangerouslySetInnerHTML={{ __html: svg }}
      style={{ display: "flex", justifyContent: "flex-start", padding: "12px 16px", minWidth: "max-content" }}
    />
  );
}

function DiagramModal({ text, onClose }) {
  const [zoom, setZoom]       = useState(1);
  const [pan, setPan]         = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef(null);
  const containerRef = useRef(null);

  // Scroll-wheel zoom (zoom toward cursor position)
  const onWheel = useCallback((e) => {
    e.preventDefault();
    setZoom(z => Math.min(4, Math.max(0.25, z - e.deltaY * 0.001)));
  }, []);

  // Pan via mouse drag
  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    setDragging(true);
    dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  }, [pan]);

  const onMouseMove = useCallback((e) => {
    if (!dragging || !dragStart.current) return;
    setPan({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
  }, [dragging]);

  const onMouseUp = useCallback(() => { setDragging(false); }, []);

  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  useEffect(() => {
    const el = containerRef.current;
    if (el) el.addEventListener("wheel", onWheel, { passive: false });
    return () => { if (el) el.removeEventListener("wheel", onWheel); };
  }, [onWheel]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 9999,
        background: "rgba(0,0,0,0.85)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        ref={containerRef}
        style={{
          width: "92vw", height: "88vh",
          background: "#09090e",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          position: "relative",
          cursor: dragging ? "grabbing" : "grab",
          userSelect: "none",
        }}
        onClick={e => e.stopPropagation()}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        {/* Controls */}
        <div
          style={{
            position: "absolute", top: 10, right: 10,
            display: "flex", gap: 4, zIndex: 10,
          }}
          onMouseDown={e => e.stopPropagation()}
        >
          <button onClick={() => setZoom(z => Math.min(4, z + 0.2))} title="Zoom in" style={btnStyle}>+</button>
          <button onClick={resetView} title="Reset view" style={{ ...btnStyle, fontSize: 10 }}>{Math.round(zoom * 100)}%</button>
          <button onClick={() => setZoom(z => Math.max(0.25, z - 0.2))} title="Zoom out" style={btnStyle}>−</button>
          <button onClick={onClose} title="Close" style={btnStyle}>
            <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M6 1v5H1"/><path d="M10 1v5h5"/>
              <path d="M6 15v-5H1"/><path d="M10 15v-5h5"/>
            </svg>
          </button>
        </div>

        {/* Diagram — zoom + pan via CSS transform */}
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", alignItems: "center", justifyContent: "center",
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: "center center",
          transition: dragging ? "none" : "transform 0.1s ease",
          padding: 32,
        }}>
          <Diagram mermaid={text} />
        </div>
      </div>
    </div>,
    document.body
  );
}

const btnStyle = {
  background: "rgba(6,6,14,0.92)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)",
  color: "var(--text-2)",
  padding: "4px 8px",
  cursor: "pointer",
  fontSize: 13,
  lineHeight: 1,
};

export default function MermaidBlock({ mermaid: text }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      {/* Outer wrapper: position:relative so the expand button anchors to the corner,
          not to the scrollable inner div */}
      <div style={{ position: "relative", margin: "12px 0" }}>
        <div style={{
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          background: "#09090e",
          overflow: "auto",
          minHeight: 120,
          maxHeight: 480,
        }}>
          <Diagram mermaid={text} />
        </div>
        <button
          onClick={() => setExpanded(true)}
          title="Expand diagram"
          style={{ position: "absolute", top: 8, right: 8, ...btnStyle }}
        >
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            <path d="M1 6V1h5"/><path d="M15 6V1h-5"/>
            <path d="M1 10v5h5"/><path d="M15 10v5h-5"/>
          </svg>
        </button>
      </div>

      {expanded && <DiagramModal text={text} onClose={() => setExpanded(false)} />}
    </>
  );
}

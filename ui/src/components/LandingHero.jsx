/**
 * LandingHero — the awwwards-aimed first impression.
 *
 * Cartographer reveals the hidden process behind AI-driven code understanding,
 * so the hero itself performs that reveal: a concept graph *draws itself* —
 * nodes appear, edges fill in, labels fade on — then loops. What you see is
 * what the product produces, not a screenshot of it.
 *
 * Interactive repo tiles take the place of inert "feature cards". Clicking
 * a tile morphs the hero into that repo's real structure: that's the primary
 * path into the tool.
 */

import { useEffect, useMemo, useRef, useState } from "react";

// Curated demo layouts — each is a tiny node-graph that evokes the shape of
// a real repo (nanoGPT, micrograd, langchain). Coordinates are hand-placed
// on a 600×360 canvas so the composition reads at a glance.
//
// These are *illustrative*, not real structure — they give the landing a
// rotating character so returning visitors see the thing move rather than
// the same static mark.
const DEMOS = [
  {
    slug: "langchain-ai/langchain",
    tagline: "LLM framework",
    accent: "#6EE7B7",
    nodes: [
      { id: "chain",   x: 120, y: 100, label: "Chain",       type: "class"  },
      { id: "llm",     x: 280, y: 60,  label: "LLM",         type: "class"  },
      { id: "prompt",  x: 280, y: 160, label: "PromptTpl",   type: "class"  },
      { id: "memory",  x: 280, y: 250, label: "Memory",      type: "module" },
      { id: "agent",   x: 450, y: 100, label: "Agent",       type: "class"  },
      { id: "tool",    x: 450, y: 200, label: "Tool",        type: "class"  },
      { id: "retrv",   x: 130, y: 240, label: "Retriever",   type: "module" },
      { id: "vstore",  x: 570, y: 60,  label: "VectorStore", type: "module" },
    ],
    edges: [
      ["chain", "llm"],
      ["chain", "prompt"],
      ["chain", "memory"],
      ["agent", "tool"],
      ["agent", "llm"],
      ["llm", "vstore"],
      ["retrv", "vstore"],
      ["chain", "retrv"],
    ],
  },
  {
    slug: "karpathy/nanoGPT",
    tagline: "GPT from scratch — 300 lines",
    accent: "#7DABFF",
    nodes: [
      { id: "train",   x: 110, y: 70,  label: "train.py",     type: "entry" },
      { id: "model",   x: 290, y: 110, label: "model.py",     type: "class" },
      { id: "attn",    x: 440, y: 70,  label: "Attention",    type: "class" },
      { id: "mlp",     x: 460, y: 160, label: "MLP",          type: "class" },
      { id: "block",   x: 310, y: 220, label: "Block",        type: "class" },
      { id: "data",    x: 130, y: 250, label: "prepare.py",   type: "fn"    },
      { id: "sample",  x: 500, y: 280, label: "sample.py",    type: "entry" },
    ],
    edges: [
      ["train", "model"],
      ["model", "attn"],
      ["model", "mlp"],
      ["model", "block"],
      ["block", "attn"],
      ["block", "mlp"],
      ["train", "data"],
      ["model", "sample"],
    ],
  },
  {
    slug: "karpathy/micrograd",
    tagline: "Autograd in 150 lines",
    accent: "#A78BFA",
    nodes: [
      { id: "value",   x: 120, y: 100, label: "Value",     type: "class" },
      { id: "add",     x: 270, y: 60,  label: "__add__",   type: "fn"    },
      { id: "mul",     x: 270, y: 140, label: "__mul__",   type: "fn"    },
      { id: "relu",    x: 270, y: 220, label: "relu",      type: "fn"    },
      { id: "back",    x: 430, y: 140, label: "backward",  type: "algo"  },
      { id: "neuron",  x: 130, y: 230, label: "Neuron",    type: "class" },
      { id: "mlp",     x: 540, y: 90,  label: "MLP",       type: "class" },
      { id: "demo",    x: 560, y: 230, label: "demo.ipynb",type: "entry" },
    ],
    edges: [
      ["value", "add"],
      ["value", "mul"],
      ["value", "relu"],
      ["add", "back"],
      ["mul", "back"],
      ["relu", "back"],
      ["value", "neuron"],
      ["neuron", "mlp"],
      ["mlp", "demo"],
      ["back", "demo"],
    ],
  },
];

// Map node-type → accent colour. Echoes the palette used in ExploreView/
// TourStory so the landing reads as part of the same product, not a brochure.
const TYPE_DOT = {
  class:   "#7DABFF",
  fn:      "#FCD34D",
  module:  "#C4B5FD",
  algo:    "#6EE7B7",
  entry:   "#FFFFFF",
};

function curvePath(a, b) {
  // Slight arc between two points — a straight line would read as a wire
  // diagram; a subtle curve reads as a considered map. Control point is
  // perpendicular-offset 18% along the line.
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const off = 0.18 * len;
  const cx = mx + nx * off;
  const cy = my + ny * off;
  return `M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`;
}

export default function LandingHero({ onPickRepo, onPasteUrl }) {
  const [demoIdx, setDemoIdx] = useState(0);
  const [url, setUrl] = useState("");

  // Auto-cycle demos every ~7s so the page feels alive without demanding
  // the user's attention. Pause on hover — the hover handler on the stage
  // sets data-paused, which a simple state mirror follows.
  const [paused, setPaused] = useState(false);
  useEffect(() => {
    if (paused) return;
    const t = setInterval(() => setDemoIdx(i => (i + 1) % DEMOS.length), 7000);
    return () => clearInterval(t);
  }, [paused]);

  const demo = DEMOS[demoIdx];

  // Cursor-reactive node displacement. We track the cursor position on the
  // stage in SVG-user-space coordinates (not CSS px) so the math matches
  // the node coordinates directly — no scale conversion per frame.
  // Why this matters: static graphs read as screenshots; the smallest hint
  // of nodes "knowing" where the cursor is tips the hero from diagram → tool.
  const stageRef = useRef(null);
  const svgRef   = useRef(null);
  const [cursor, setCursor] = useState({ x: -9999, y: -9999, active: false });
  function onStageMove(e) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    // Map from CSS px → SVG viewBox units (680×340).
    const x = ((e.clientX - rect.left) / rect.width)  * 680;
    const y = ((e.clientY - rect.top)  / rect.height) * 340;
    setCursor({ x, y, active: true });
    setPaused(true);
  }
  function onStageLeave() {
    setCursor({ x: -9999, y: -9999, active: false });
    setPaused(false);
  }

  // Per-node displacement — inverse-distance push away from cursor.
  // Capped at 14 SVG units so labels don't drift into other nodes.
  function nodeOffset(n) {
    if (!cursor.active) return { dx: 0, dy: 0 };
    const dx = n.x - cursor.x;
    const dy = n.y - cursor.y;
    const d  = Math.hypot(dx, dy) || 1;
    const R  = 140;
    if (d > R) return { dx: 0, dy: 0 };
    const push = (1 - d / R) * 14;
    return { dx: (dx / d) * push, dy: (dy / d) * push };
  }

  // Bump animKey on demo change so the SVG remounts → the drawing animation
  // replays. React rebuilds the subtree rather than us having to toggle classes.
  const animKey = useMemo(() => `${demoIdx}-${demo.slug}`, [demoIdx, demo.slug]);

  function submit(e) {
    e?.preventDefault();
    const clean = url.trim();
    if (!clean) return;
    // Reveal origin = center of the "Map it" button, if we can find it.
    const btn = e?.currentTarget?.querySelector(".lh-url-btn");
    const rect = btn?.getBoundingClientRect();
    const origin = rect
      ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
      : undefined;
    onPasteUrl?.(clean, origin);
  }

  return (
    <div className="lh-root">
      {/* Headline — serif display, single focal line */}
      <div className="lh-title-wrap">
        <h1 className="lh-title display-serif">
          Map <em>any</em> codebase.
        </h1>
        <p className="lh-sub">
          Paste a GitHub repo — Cartographer reads every file, extracts the key concepts,
          and turns the codebase into a guided tour. Watch the hidden structure surface.
        </p>
      </div>

      {/* Live concept-map demo — the thing the product produces, performing itself */}
      <div
        key={animKey}
        ref={stageRef}
        className="lh-stage"
        onMouseMove={onStageMove}
        onMouseLeave={onStageLeave}
        style={{ "--demo-accent": demo.accent }}
      >
        <svg
          ref={svgRef}
          viewBox="0 0 680 340"
          className="lh-svg"
          role="img"
          aria-label={`Animated concept graph for ${demo.slug}`}
        >
          <defs>
            <radialGradient id={`lh-glow-${demoIdx}`} cx="50%" cy="50%" r="50%">
              <stop offset="0%"  stopColor={demo.accent} stopOpacity="0.35" />
              <stop offset="70%" stopColor={demo.accent} stopOpacity="0"    />
            </radialGradient>
          </defs>

          {/* Background glow */}
          <rect x="0" y="0" width="680" height="340" fill={`url(#lh-glow-${demoIdx})`} opacity="0.7" />

          {/* Edges — drawn first so nodes paint on top.
              Two layers per edge: the static curve (reveals via dashoffset)
              plus a "pulse" overlay that travels the same path in a loop,
              so the graph reads as data actively flowing, not a still diagram. */}
          {demo.edges.map(([a, b], i) => {
            const na = demo.nodes.find(n => n.id === a);
            const nb = demo.nodes.find(n => n.id === b);
            if (!na || !nb) return null;
            const oa = nodeOffset(na);
            const ob = nodeOffset(nb);
            const d  = curvePath(
              { x: na.x + oa.dx, y: na.y + oa.dy },
              { x: nb.x + ob.dx, y: nb.y + ob.dy },
            );
            const pulseId = `pulse-${demoIdx}-${i}`;
            return (
              <g key={`e-${a}-${b}`}>
                <path
                  id={pulseId}
                  d={d}
                  fill="none"
                  stroke={demo.accent}
                  strokeWidth="1.2"
                  strokeLinecap="round"
                  opacity="0.55"
                  className="lh-edge"
                  style={{ animationDelay: `${400 + i * 90}ms` }}
                />
                {/* Flowing pulse — a bright dot that slides along the edge */}
                <circle r="2.2" fill={demo.accent} className="lh-pulse" style={{ filter: `drop-shadow(0 0 4px ${demo.accent})` }}>
                  <animateMotion
                    dur={`${3.6 + (i % 3) * 0.5}s`}
                    begin={`${1.2 + i * 0.18}s`}
                    repeatCount="indefinite"
                    rotate="auto"
                  >
                    <mpath href={`#${pulseId}`} />
                  </animateMotion>
                </circle>
              </g>
            );
          })}

          {/* Nodes */}
          {demo.nodes.map((n, i) => {
            const off = nodeOffset(n);
            const cx = n.x + off.dx;
            const cy = n.y + off.dy;
            return (
              <g
                key={n.id}
                className="lh-node"
                style={{ animationDelay: `${i * 120}ms`, transformOrigin: `${n.x}px ${n.y}px` }}
              >
                {/* Soft breathing halo — only shows up close to cursor */}
                {cursor.active && Math.hypot(n.x - cursor.x, n.y - cursor.y) < 140 && (
                  <circle cx={cx} cy={cy} r="26" fill={TYPE_DOT[n.type] || demo.accent} opacity="0.12" />
                )}
                <circle cx={cx} cy={cy} r="18" fill="rgba(20,22,38,0.85)" stroke={TYPE_DOT[n.type] || demo.accent} strokeWidth="1.5" />
                <circle cx={cx} cy={cy} r="3" fill={TYPE_DOT[n.type] || demo.accent} />
                <text
                  x={cx}
                  y={cy + 34}
                  textAnchor="middle"
                  fill="rgba(230,235,250,0.85)"
                  fontSize="10"
                  fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                  className="lh-label"
                  style={{ animationDelay: `${200 + i * 120}ms` }}
                >
                  {n.label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Stage caption — tells the viewer what they're looking at */}
        <div className="lh-stage-meta">
          <span className="lh-stage-dot" style={{ background: demo.accent }} />
          <span className="lh-stage-slug">{demo.slug}</span>
          <span className="lh-stage-dash">·</span>
          <span className="lh-stage-tagline">{demo.tagline}</span>
        </div>
      </div>

      {/* Primary action — pick a demo repo (click = ingest) */}
      <div className="lh-tiles">
        {DEMOS.map((d, i) => (
          <button
            key={d.slug}
            className={`lh-tile hover-lift${i === demoIdx ? " is-active" : ""}`}
            onClick={(e) => onPickRepo?.(d.slug, d.accent, { x: e.clientX, y: e.clientY })}
            onMouseEnter={() => setDemoIdx(i)}
            style={{ "--tile-accent": d.accent }}
          >
            <span className="lh-tile-kicker">Try</span>
            <span className="lh-tile-name">{d.slug.split("/")[1]}</span>
            <span className="lh-tile-tagline">{d.tagline}</span>
            <svg className="lh-tile-arrow" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 8h10M9 4l4 4-4 4"/>
            </svg>
          </button>
        ))}
      </div>

      {/* Secondary action — paste a URL */}
      <form className="lh-url" onSubmit={submit}>
        <span className="lh-url-prefix">github.com/</span>
        <input
          className="lh-url-input"
          type="text"
          placeholder="owner/repo"
          value={url}
          onChange={e => setUrl(e.target.value)}
          spellCheck="false"
          autoComplete="off"
        />
        <button type="submit" className="lh-url-btn" disabled={!url.trim()} title="Index this repo">
          Map it
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8h10M9 4l4 4-4 4"/></svg>
        </button>
      </form>
    </div>
  );
}

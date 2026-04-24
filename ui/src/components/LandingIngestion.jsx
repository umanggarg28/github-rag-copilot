/**
 * LandingIngestion — the signature moment.
 *
 * When the user clicks a tile or submits a URL on the landing, we don't
 * yank them away into a sidebar progress bar. The hero stage stays, and
 * the living map of the repo forms inside it — driven by real SSE events
 * from /ingest/stream.
 *
 * POV of the user: "show me this repo" → they WATCH it happen. The
 * illustrative graph dissolves, file-dots stream in, clusters form,
 * concepts flash as the LLM understands chunks, and when ingestion is
 * done the whole thing stabilises into an invitation: "Begin the tour."
 *
 * No backend changes. We use only the existing event shape:
 *   { step: "fetching|filtering|chunking|embedding|contextualizing|storing|done|error",
 *     detail: "human-readable message" }
 * The `filtering`/`chunking` details carry a file count we parse; the
 * `contextualizing` detail carries "X / Y" we parse. Everything else is
 * theatre tuned to the rhythm of the real pipeline.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { BASE } from "../api";

// Narrative copy per phase — the lines that rotate below the stage as the
// pipeline progresses. Tuned to be evocative, not technical. "Reading every
// file" beats "Downloading repo" — we're selling a journey, not an API call.
const PHASE_COPY = {
  embark:         "Opening the map…",
  fetching:       "Pulling the repository from GitHub…",
  filtering:      "Reading every file…",
  chunking:       "Breaking down the structure…",
  embedding:      "Learning how it all fits together…",
  contextualizing:"Finding the key concepts…",
  storing:        "Sealing the map…",
  done:           "The map is ready.",
  error:          "Something went wrong on the journey.",
};

// Concept flashes — short labels that briefly highlight over a random file
// dot during the long contextualizing phase. We don't know the real concept
// names yet (they're extracted after ingestion), so these are generic
// structural terms that could plausibly describe any node. Variety reads as
// "many things being discovered" rather than "one fake label on repeat."
const CONCEPT_FLASHES = [
  "class", "module", "function", "loop", "pipeline",
  "handler", "state", "cache", "schema", "route",
  "builder", "parser", "worker", "store", "transform",
];

// Central stage coordinates — match LandingHero's 680x340 viewBox so the
// visual hand-off is seamless (same stage dimensions, same glow container).
const CX = 340;
const CY = 170;

/**
 * Lay out up to N file-dots in concentric rings around the centre.
 * Each ring has a fixed capacity; we keep filling until we run out of
 * count or reach ring 4. Dots are slightly jittered on angle and radius
 * so the arrangement reads as organic rather than gridded.
 */
function layoutDots(count, seed = 0) {
  const rings = [
    { r: 78,  n: 8,  ry: 0.82 },
    { r: 120, n: 14, ry: 0.82 },
    { r: 160, n: 20, ry: 0.78 },
    { r: 200, n: 28, ry: 0.72 },
  ];
  const dots = [];
  let idx = 0;
  // Deterministic pseudo-random via the seed — so re-renders don't scramble
  // the layout mid-animation. (Math.random would reroll every render.)
  let s = seed || 1;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  for (const ring of rings) {
    for (let i = 0; i < ring.n && idx < count; i++) {
      const angle = (i / ring.n) * Math.PI * 2 + (rand() - 0.5) * 0.35;
      const r = ring.r + (rand() - 0.5) * 14;
      dots.push({
        id:    idx,
        x:     CX + Math.cos(angle) * r,
        y:     CY + Math.sin(angle) * r * ring.ry,
        delay: idx * 28 + Math.floor(rand() * 60),
      });
      idx++;
    }
  }
  return dots;
}

/** Parse "247 files, filtering..." or similar → 247. */
function parseFileCount(detail = "") {
  const m = detail.match(/(\d+)\s+files?/i);
  return m ? parseInt(m[1], 10) : null;
}
/** Parse any "N / M" fragment (used for contextualizing and embedding events).
 *  Same shape is emitted by both phases; consumers pick the right progress
 *  state based on the event's `step`. */
function parseProgressRatio(detail = "") {
  const m = detail.match(/(\d+)\s*\/\s*(\d+)/);
  return m ? { done: parseInt(m[1], 10), total: parseInt(m[2], 10) } : null;
}

export default function LandingIngestion({
  repoUrl,
  repoSlug: slugProp,
  accent = "#5B8FF9",
  onComplete,
  onError,
  onAbort,
}) {
  // Extract slug for display ("owner/repo") — prefer the explicit prop (set
  // from tile clicks where we already know the slug) but fall back to URL
  // parsing for the URL-input case.
  const displaySlug = useMemo(() => {
    if (slugProp) return slugProp;
    const m = repoUrl?.match(/github\.com\/([^/]+\/[^/]+)/);
    return m ? m[1] : repoUrl;
  }, [slugProp, repoUrl]);

  const [phase,         setPhase]         = useState("embark");
  const [fileCount,     setFileCount]     = useState(null);
  const [ctxProgress,   setCtxProgress]   = useState(null); // {done,total} | null
  const [embedProgress, setEmbedProgress] = useState(null); // {done,total} | null
  const [errorMsg,      setErrorMsg]      = useState(null);

  // Concept flashes: each is { id, x, y, label }. They're transient — added
  // on a timer during contextualizing, removed after ~1.4s. Using an array
  // instead of a single flash lets multiple overlap for a "busy" feel.
  const [flashes, setFlashes] = useState([]);
  const flashIdRef = useRef(0);

  // Open the SSE connection exactly once per repoUrl. EventSource auto-
  // reconnects on network blips, so we close it explicitly on done/error
  // to prevent re-running the pipeline.
  useEffect(() => {
    if (!repoUrl) return;
    const url = `${BASE}/ingest/stream?repo=${encodeURIComponent(repoUrl)}`;
    const es  = new EventSource(url);

    es.onmessage = (ev) => {
      const event = JSON.parse(ev.data);
      setPhase(event.step);

      if (event.step === "filtering" || event.step === "chunking") {
        const n = parseFileCount(event.detail);
        if (n != null) setFileCount(n);
      }
      if (event.step === "contextualizing") {
        const p = parseProgressRatio(event.detail);
        if (p) setCtxProgress(p);
      }
      if (event.step === "embedding") {
        // The backend now emits "Embedded X/Y chunks..." during the
        // checkpointed embed loop. Drive the progress bar dynamically
        // off those — without this, the bar would sit at the embedding
        // base weight (0.40) for the entire 2–3 min phase and read as stuck.
        const p = parseProgressRatio(event.detail);
        if (p) setEmbedProgress(p);
      }

      if (event.step === "done") {
        es.close();
        // Deliberately do NOT auto-advance. The "map is ready" moment is a
        // beat of arrival — the CTA blooms, the user chooses to enter.
        // Auto-advancing would rob them of that agency.
      } else if (event.step === "error") {
        es.close();
        setErrorMsg(event.detail || "Unknown error");
        onError?.(event.detail);
      }
    };

    es.onerror = () => {
      es.close();
      setErrorMsg("Connection dropped — check the backend is running.");
      onError?.("connection");
    };

    return () => es.close();
  }, [repoUrl, displaySlug, onComplete, onError]);

  // Layout dots based on the real file count. Minimum 32 so the stage feels
  // populated even on tiny repos; cap at 70 so the SVG doesn't choke. We
  // scale the visible count linearly with phase — dots stream in gradually.
  const totalDots = Math.min(70, Math.max(32, fileCount || 48));
  const allDots   = useMemo(() => layoutDots(totalDots, 42), [totalDots]);

  // Visible dot count ramps up through phases so the viewer perceives the
  // map forming, not appearing all at once.
  const visibleCount = useMemo(() => {
    switch (phase) {
      case "embark":          return 0;
      case "fetching":        return Math.round(totalDots * 0.15);
      case "filtering":       return Math.round(totalDots * 0.55);
      case "chunking":        return Math.round(totalDots * 0.80);
      case "embedding":       return totalDots;
      case "contextualizing": return totalDots;
      case "storing":
      case "done":            return totalDots;
      default:                return totalDots;
    }
  }, [phase, totalDots]);

  const visibleDots = allDots.slice(0, visibleCount);

  // During contextualizing, spawn a concept flash every ~1.1s over a random
  // visible dot. Flashes self-expire after 1.4s via their own setTimeout.
  // This loop runs ONLY in the contextualizing phase to keep the stage busy.
  useEffect(() => {
    if (phase !== "contextualizing" || visibleDots.length === 0) return;
    const tick = setInterval(() => {
      const dot   = visibleDots[Math.floor(Math.random() * visibleDots.length)];
      const label = CONCEPT_FLASHES[Math.floor(Math.random() * CONCEPT_FLASHES.length)];
      const id    = ++flashIdRef.current;
      setFlashes((prev) => [...prev, { id, x: dot.x, y: dot.y, label }]);
      setTimeout(() => {
        setFlashes((prev) => prev.filter((f) => f.id !== id));
      }, 1400);
    }, 1100);
    return () => clearInterval(tick);
  }, [phase, visibleDots]);

  // Edges — draw a modest number of short connections between nearby dots.
  // Computed once we have enough visible dots. "Nearby" is a fixed distance
  // threshold so the graph looks organic without being a hairball.
  const edges = useMemo(() => {
    if (visibleDots.length < 4) return [];
    const out = [];
    for (let i = 0; i < visibleDots.length; i++) {
      for (let j = i + 1; j < visibleDots.length; j++) {
        const a = visibleDots[i], b = visibleDots[j];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 60 && d > 22) {
          out.push({ a, b, key: `e-${a.id}-${b.id}` });
          if (out.length > 46) return out;
        }
      }
    }
    return out;
  }, [visibleDots]);

  // Progress 0..1 — used for both the progress bar fill and the stage's
  // accent-glow intensity. We weight phases so the bar feels truthful:
  // fetching & filtering are quick; embedding & contextualizing are long,
  // so they each claim a wide range and fill dynamically from real
  // {done,total} events.
  //
  // Weight layout:
  //   0.00 → 0.28  fetching · filtering · chunking (quick phases)
  //   0.28 → 0.55  contextualizing (force re-index only)
  //   0.40 → 0.92  embedding  (overlaps contextualizing intentionally —
  //                 the paths are mutually exclusive in time)
  //   0.92 → 1.00  storing · done
  const progress = useMemo(() => {
    if (phase === "done")   return 1;
    if (phase === "error")  return 0;
    const base = {
      embark:          0.02,
      fetching:        0.08,
      filtering:       0.18,
      chunking:        0.28,
      contextualizing: 0.28,
      embedding:       0.40,
      storing:         0.92,
    }[phase] ?? 0;
    // Contextualizing fills 0.28 → 0.55 off real X/Y events.
    if (phase === "contextualizing" && ctxProgress && ctxProgress.total) {
      return base + (0.27 * (ctxProgress.done / ctxProgress.total));
    }
    // Embedding fills 0.40 → 0.92 off real X/Y events emitted per batch.
    if (phase === "embedding" && embedProgress && embedProgress.total) {
      return base + (0.52 * (embedProgress.done / embedProgress.total));
    }
    return base;
  }, [phase, ctxProgress, embedProgress]);

  const isDone  = phase === "done";
  const isError = phase === "error";

  // Rotate the copy every ~2.8s within the same phase so the long phases
  // don't feel stuck on one line. The variants are gentle rewordings —
  // same meaning, different rhythm.
  const [copyVariant, setCopyVariant] = useState(0);
  useEffect(() => {
    setCopyVariant(0);
    const t = setInterval(() => setCopyVariant((v) => v + 1), 2800);
    return () => clearInterval(t);
  }, [phase]);
  const copy = (() => {
    if (isError) return errorMsg || PHASE_COPY.error;
    const variants = COPY_VARIANTS[phase] || [PHASE_COPY[phase] || ""];
    return variants[copyVariant % variants.length];
  })();

  return (
    <div className="li-root">
      {/* Title — "Mapping owner/repo" with the slug in accent colour */}
      <div className="lh-title-wrap">
        <h1 className="lh-title display-serif li-title">
          {isDone ? "Ready to explore" : isError ? "A detour" : "Mapping"}{" "}
          <em className="li-slug" style={{ "--demo-accent": accent }}>
            {displaySlug}
          </em>
        </h1>
        <p className="lh-sub li-sub" key={`${phase}-${copyVariant}`}>{copy}</p>
      </div>

      {/* Stage — same box as LandingHero so the transition is visually seamless */}
      <div
        className={`lh-stage li-stage ${isDone ? "is-done" : ""} ${isError ? "is-error" : ""}`}
        style={{ "--demo-accent": accent, "--li-progress": progress }}
      >
        <svg viewBox="0 0 680 340" className="lh-svg" role="img" aria-label={`Building map of ${displaySlug}`}>
          <defs>
            <radialGradient id="li-core-glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%"  stopColor={accent} stopOpacity="0.55" />
              <stop offset="60%" stopColor={accent} stopOpacity="0"    />
            </radialGradient>
            <radialGradient id="li-bg-glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%"  stopColor={accent} stopOpacity="0.22" />
              <stop offset="70%" stopColor={accent} stopOpacity="0"    />
            </radialGradient>
          </defs>

          {/* Background wash — darkens slightly, glow tracks progress */}
          <rect x="0" y="0" width="680" height="340" fill="url(#li-bg-glow)" opacity="0.9" />

          {/* Core node at centre — the repository itself */}
          <g className="li-core">
            <circle cx={CX} cy={CY} r="64" fill="url(#li-core-glow)" />
            <circle cx={CX} cy={CY} r="22" fill="rgba(14,18,34,0.95)" stroke={accent} strokeWidth="1.6" />
            <circle cx={CX} cy={CY} r="3"  fill={accent} />
          </g>

          {/* Edges between nearby dots — thin, dashed so they read as relationships not wires */}
          {edges.map(({ a, b, key }) => (
            <line
              key={key}
              x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              stroke={accent} strokeOpacity="0.55"
              strokeWidth="1.25" strokeLinecap="round"
              className="li-edge"
            />
          ))}

          {/* File dots — fade in in order, scale with a little overshoot */}
          {visibleDots.map((d) => (
            <g
              key={d.id}
              className="li-dot"
              style={{ animationDelay: `${d.delay}ms`, transformOrigin: `${d.x}px ${d.y}px` }}
            >
              <circle cx={d.x} cy={d.y} r="4"   fill="rgba(14,18,34,0.9)" stroke={accent} strokeWidth="1" />
              <circle cx={d.x} cy={d.y} r="1.5" fill={accent} opacity="0.85" />
            </g>
          ))}

          {/* Concept flashes — transient labels over random dots */}
          {flashes.map((f) => (
            <g key={f.id} className="li-flash" style={{ transformOrigin: `${f.x}px ${f.y}px` }}>
              <circle cx={f.x} cy={f.y} r="8" fill="none" stroke={accent} strokeWidth="1.2" opacity="0.9" />
              <text
                x={f.x}
                y={f.y - 14}
                textAnchor="middle"
                fill="rgba(240,246,255,0.95)"
                fontSize="10"
                fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              >
                {f.label}
              </text>
            </g>
          ))}
        </svg>

        {/* Progress strip — same visual language as the stage-meta pill */}
        <div className="li-meta">
          <span className="li-meta-dot" style={{ background: accent }} />
          <span className="li-meta-label">
            {isDone
              ? (fileCount ? `${fileCount} files indexed` : "Indexing complete")
              : isError
                ? "Something interrupted the map"
                : (fileCount ? `${fileCount} files` : "Reading repository…")}
          </span>
          <div className="li-bar" aria-hidden="true">
            <div className="li-bar-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </div>
      </div>

      {/* Footer CTA — on done, the invitation to begin. On error, a retry. */}
      <div className="li-cta-row">
        {isDone && (
          <button
            className="li-cta hover-lift"
            style={{ "--demo-accent": accent }}
            onClick={() => onComplete?.(displaySlug)}
            autoFocus
          >
            Begin the tour
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 8h10M9 4l4 4-4 4"/>
            </svg>
          </button>
        )}
        {isError && (
          <>
            <button
              className="li-cta hover-lift"
              style={{ "--demo-accent": accent }}
              onClick={() => onRetry?.()}
              autoFocus
            >
              Try again
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M3.5 8a4.5 4.5 0 1 1 1.32 3.18M3.5 4.5v3h3"/>
              </svg>
            </button>
            <button className="li-cta-ghost" onClick={onAbort}>
              Back
            </button>
          </>
        )}
        {!isDone && !isError && (
          <button className="li-cta-ghost li-cancel" onClick={onAbort}>
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}

// Variants are grouped here at the bottom so the component body reads top
// to bottom without the copy tables interrupting the flow. Each phase can
// have 1–3 variants that rotate every ~2.8s so long phases don't feel stuck.
const COPY_VARIANTS = {
  embark: [
    "Opening the map…",
  ],
  fetching: [
    "Pulling the repository from GitHub…",
    "Downloading every file in the tree…",
  ],
  filtering: [
    "Reading every file…",
    "Seeing what's here…",
  ],
  chunking: [
    "Breaking down the structure…",
    "Splitting code into meaningful pieces…",
  ],
  embedding: [
    "Learning how it all fits together…",
    "Turning code into meaning…",
  ],
  contextualizing: [
    "Finding the key concepts…",
    "Understanding what each piece does…",
    "Surfacing the shape of the project…",
  ],
  storing: [
    "Sealing the map…",
  ],
  done: [
    "The map is ready.",
  ],
};

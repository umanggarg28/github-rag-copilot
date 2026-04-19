/**
 * TourStory — focused, single-concept reading mode for the tour.
 *
 * Complements the canvas grid: same data, one concept at a time, with
 * keyboard scrub, a progress rail, a flow strip, and an animated transition
 * between steps. The goal is to *pace* the learning — readers can't take in
 * 7 cards at once, but they can take in one well.
 */

import { useEffect, useState } from "react";

// Mirror ExploreView's palette so Story mode reads as the same product —
// kept local (not imported) to avoid a cyclic import when ExploreView later
// mounts TourStory as a child.
const TYPE_STYLE = {
  class:     { border: "#5B8FF9", dot: "#7DABFF", tag: "class"  },
  function:  { border: "#FBBF24", dot: "#FCD34D", tag: "fn"     },
  module:    { border: "#A78BFA", dot: "#C4B5FD", tag: "module" },
  algorithm: { border: "#34D399", dot: "#6EE7B7", tag: "algo"   },
};
const FALLBACK = { border: "#4E5E80", dot: "#8896B8", tag: "?" };

function ChevronLeft() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M9.78 12.78a.75.75 0 0 1-1.06 0L4.47 8.53a.75.75 0 0 1 0-1.06l4.25-4.25a.75.75 0 0 1 1.06 1.06L6.06 8l3.72 3.72a.75.75 0 0 1 0 1.06Z"/>
    </svg>
  );
}
function ChevronRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 1 1-1.06-1.06L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"/>
    </svg>
  );
}
function ExternalIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" style={{ opacity: 0.7 }}>
      <path d="M3.75 2a.75.75 0 0 0 0 1.5h5.19L2.22 10.22a.75.75 0 1 0 1.06 1.06L10 4.56v5.19a.75.75 0 0 0 1.5 0v-7a.75.75 0 0 0-.75-.75h-7Z"/>
      <path d="M12.5 8.75a.75.75 0 0 0-1.5 0v3.5a.25.25 0 0 1-.25.25h-7.5a.25.25 0 0 1-.25-.25v-7.5a.25.25 0 0 1 .25-.25h3.5a.75.75 0 0 0 0-1.5h-3.5A1.75 1.75 0 0 0 1.5 4.75v7.5A1.75 1.75 0 0 0 3.25 14h7.5a1.75 1.75 0 0 0 1.75-1.75v-3.5Z"/>
    </svg>
  );
}

export default function TourStory({ data, repo, onAskAbout }) {
  // Concepts are indexed by reading_order so ← / → match the author's intended sequence
  const concepts = [...(data.concepts || [])].sort(
    (a, b) => (a.reading_order ?? 999) - (b.reading_order ?? 999)
  );

  const [idx, setIdx] = useState(0);
  // Bump a key on index change so the card remounts — CSS animation replays
  // without needing to toggle className off then on.
  const [animKey, setAnimKey] = useState(0);

  useEffect(() => { setAnimKey(k => k + 1); }, [idx]);

  // Clamp idx if concepts shrink (e.g. regenerate produced fewer)
  useEffect(() => {
    if (idx > concepts.length - 1) setIdx(Math.max(0, concepts.length - 1));
  }, [concepts.length, idx]);

  // Keyboard nav. Guarded against input/textarea focus so the chat box still works.
  useEffect(() => {
    function onKey(e) {
      const tag = (e.target?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || e.target?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        setIdx(i => Math.min(i + 1, concepts.length - 1));
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        setIdx(i => Math.max(i - 1, 0));
      } else if (e.key === "Home") {
        e.preventDefault(); setIdx(0);
      } else if (e.key === "End") {
        e.preventDefault(); setIdx(concepts.length - 1);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [concepts.length]);

  if (concepts.length === 0) return null;

  const c      = concepts[idx];
  const prev   = idx > 0 ? concepts[idx - 1] : null;
  const next   = idx < concepts.length - 1 ? concepts[idx + 1] : null;
  const style  = TYPE_STYLE[c.type] || FALLBACK;
  const ghUrl  = `https://github.com/${repo}/blob/HEAD/${c.file}`;
  // id → visual reading position, so "depends on" pills show the user-visible number
  const idToPos = Object.fromEntries(concepts.map((cc, i) => [cc.id, i + 1]));

  function handleAsk() {
    onAskAbout?.(
      c.ask ||
      `Explain "${c.name}" in ${repo} in detail — what does it do, how does it work, and what are the key methods or functions involved?`
    );
  }

  return (
    <div
      className="ts-root"
      // Drive the ambient tint from the active concept's type colour.
      // Consumed by .ts-root background gradients — transitions via @property.
      style={{ "--ts-tint": style.border }}
    >
      {/* Top flow strip — every step visible, dependency arrows drawn as lines */}
      <div className="ts-flow" role="tablist" aria-label="Tour steps">
        {concepts.map((cc, i) => {
          const s = TYPE_STYLE[cc.type] || FALLBACK;
          const isActive = i === idx;
          const isDone   = i < idx;
          return (
            <button
              key={cc.id}
              role="tab"
              aria-selected={isActive}
              className={`ts-flow-step${isActive ? " is-active" : ""}${isDone ? " is-done" : ""}`}
              onClick={() => setIdx(i)}
              title={cc.name}
              style={isActive ? { borderColor: s.border, color: s.dot } : undefined}
            >
              <span className="ts-flow-num">{String(i + 1).padStart(2, "0")}</span>
              <span className="ts-flow-label">{cc.name}</span>
            </button>
          );
        })}
      </div>

      {/* Stage — peripheral prev/next hints flanking the focus card */}
      <div className="ts-stage">
        <button
          className="ts-nav ts-nav-prev"
          onClick={() => setIdx(i => Math.max(i - 1, 0))}
          disabled={!prev}
          aria-label={prev ? `Previous: ${prev.name}` : "At start"}
        >
          <ChevronLeft />
          {prev && <span className="ts-nav-peek">{prev.name}</span>}
        </button>

        <article
          key={animKey}
          className="ts-card"
          onMouseMove={(e) => {
            // Set CSS vars for the cursor-tracking glow. Measured against the
            // card box — not viewport — so it works regardless of scroll.
            const r = e.currentTarget.getBoundingClientRect();
            e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
            e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
          }}
          style={{
            // Accent tint per concept type — subtle left border rail + glow
            borderColor: "rgba(255,255,255,0.08)",
            boxShadow: `0 0 0 1px ${style.border}22, 0 30px 80px rgba(0,0,16,0.6), 0 0 120px ${style.border}22`,
            // Consumed by the cursor-glow pseudo-element
            "--glow-color": `${style.border}`,
          }}
        >
          {/* Dedicated clip layer — absolute overlay sitting on the non-scrolling
              card box. Holds the accent rail and any future fixed decorations
              so they respect the card's rounded corners at all times. */}
          <div className="ts-card-clip">
            <div className="ts-rail-accent" style={{ background: `linear-gradient(180deg, ${style.dot} 0%, ${style.border} 100%)` }} />
          </div>

          {/* Inner body owns the scroll context — keeps .ts-card itself
              non-scrolling so border-radius clipping and the ::after glow
              layer both stay bound to the visible card box. */}
          <div className="ts-card-body">
          <header className="ts-card-head">
            <div className="ts-head-left">
              <span className="ts-num">
                {String(idx + 1).padStart(2, "0")}<span className="ts-num-sep">/</span>{String(concepts.length).padStart(2, "0")}
              </span>
              {idx === 0 && <span className="ts-entry-badge">Start here</span>}
            </div>
            <span className="ts-type" style={{ color: style.dot, borderColor: `${style.dot}44` }}>{style.tag}</span>
          </header>

          <h2 className="ts-title">{c.name}</h2>
          {c.subtitle && <p className="ts-subtitle">{c.subtitle}</p>}
          {c.description && <p className="ts-desc">{c.description}</p>}

          {c.key_items?.length > 0 && (
            <div className="ts-items">
              {c.key_items.map(item => (
                <code key={item} className="ts-item">{item}</code>
              ))}
            </div>
          )}

          {c.depends_on?.length > 0 && (
            <div className="ts-deps">
              <span className="ts-deps-label">Builds on</span>
              {c.depends_on.map(depId => {
                const pos = idToPos[depId];
                const dep = concepts.find(cc => cc.id === depId);
                if (!pos || !dep) return null;
                return (
                  <button
                    key={depId}
                    className="ts-dep-pill"
                    onClick={() => setIdx(pos - 1)}
                    title={dep.name}
                  >
                    <span className="ts-dep-num">{String(pos).padStart(2, "0")}</span>
                    {dep.name}
                  </button>
                );
              })}
            </div>
          )}

          <footer className="ts-card-foot">
            <a href={ghUrl} target="_blank" rel="noreferrer" className="ts-file" title="Open file on GitHub">
              <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" style={{ opacity: 0.55 }} aria-hidden="true">
                <path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 8.75 4.25V1.5Zm6.75.56v2.19c0 .138.112.25.25.25h2.19Z"/>
              </svg>
              <span className="ts-file-path">{c.file}</span>
              <ExternalIcon />
            </a>
            <button className="ts-ask" onClick={handleAsk}>Ask about this →</button>
          </footer>
          </div>
        </article>

        <button
          className="ts-nav ts-nav-next"
          onClick={() => setIdx(i => Math.min(i + 1, concepts.length - 1))}
          disabled={!next}
          aria-label={next ? `Next: ${next.name}` : "At end"}
        >
          {next && <span className="ts-nav-peek">{next.name}</span>}
          <ChevronRight />
        </button>
      </div>

      {/* Bottom rail: progress line + clickable dots + keyboard hint */}
      <div className="ts-rail">
        <div className="ts-rail-track">
          <div
            className="ts-rail-fill"
            style={{ width: concepts.length > 1 ? `${(idx / (concepts.length - 1)) * 100}%` : "100%" }}
          />
          <div className="ts-rail-dots">
            {concepts.map((cc, i) => (
              <button
                key={cc.id}
                className={`ts-rail-dot${i === idx ? " is-active" : ""}${i < idx ? " is-done" : ""}`}
                onClick={() => setIdx(i)}
                aria-label={`Go to step ${i + 1}: ${cc.name}`}
              />
            ))}
          </div>
        </div>
        <div className="ts-rail-hint">
          <kbd>←</kbd><kbd>→</kbd> navigate · <kbd>Home</kbd>/<kbd>End</kbd> jump · click a step
        </div>
      </div>
    </div>
  );
}

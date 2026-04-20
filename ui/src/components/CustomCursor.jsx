/**
 * CustomCursor — a small companion dot that follows the mouse and
 * subtly scales when over interactive elements.
 *
 * Intentionally *additive*: the native OS cursor is preserved (text
 * selection still feels right), and this dot just rides alongside. It's
 * a craft detail that communicates "this page notices you" — the same
 * signal as a cursor-glow spotlight but at a smaller, precise scale.
 *
 * Motion is driven via CSS custom properties (--x, --y) set on a fixed
 * element rather than React state — so movement never triggers rerenders.
 *
 * Disabled on coarse pointers (touch) and when reduced motion is preferred.
 */

import { useEffect, useRef, useState } from "react";

const INTERACTIVE_SELECTOR = [
  "button",
  "a",
  '[role="button"]',
  '[role="tab"]',
  ".ts-flow-step",
  ".ts-rail-dot",
  ".ts-dep-pill",
  ".suggestion-btn",
  ".try-repo-chip",
  ".ec-ask",
  ".ec-node",
  ".onboarding-step",
].join(", ");

// Surfaces where the dot becomes noise instead of signal:
//   - inputs / textareas: the I-beam is the actual pointer, the dot fights it
//   - pre / code: monospaced text where a 10px dot reads like a glyph
//   - contenteditable: same as inputs
// We hide the dot entirely over these. Pure-prose elements (p, span) are
// NOT in this list — the dot is useful ambient signal over long reading.
const HIDE_OVER_SELECTOR = [
  'input:not([type="button"]):not([type="submit"]):not([type="checkbox"]):not([type="radio"])',
  'textarea',
  '[contenteditable="true"]',
  'pre',
  'code',
  '.mcp-detail-preview',
].join(', ');

export default function CustomCursor() {
  const ref = useRef(null);
  // Only enabled if hover is supported + user hasn't requested reduced motion.
  // Decided once on mount — both media queries are user-level, not volatile.
  const [enabled] = useState(() => {
    if (typeof window === "undefined") return false;
    const hasHover = window.matchMedia?.("(hover: hover)").matches;
    const reduced  = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    return hasHover && !reduced;
  });

  useEffect(() => {
    if (!enabled) return;
    const el = ref.current;
    if (!el) return;

    function onMove(e) {
      el.style.setProperty("--x", `${e.clientX}px`);
      el.style.setProperty("--y", `${e.clientY}px`);
      if (el.dataset.visible !== "1") {
        el.dataset.visible = "1";
      }
    }
    function onOver(e) {
      // Hide entirely over text-entry surfaces; otherwise upgrade to the
      // active ring when over any interactive element in the bubble path.
      const overInput = !!e.target.closest?.(TEXT_INPUT_SELECTOR);
      el.dataset.overInput = overInput ? "1" : "0";
      const interactive = !overInput && e.target.closest?.(INTERACTIVE_SELECTOR);
      el.dataset.active = interactive ? "1" : "0";
    }
    function onLeave() { el.dataset.visible = "0"; }
    function onEnter() { el.dataset.visible = "1"; }

    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("mouseover", onOver, { passive: true });
    document.addEventListener("mouseleave", onLeave);
    document.addEventListener("mouseenter", onEnter);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseover", onOver);
      document.removeEventListener("mouseleave", onLeave);
      document.removeEventListener("mouseenter", onEnter);
    };
  }, [enabled]);

  if (!enabled) return null;
  return <div ref={ref} className="custom-cursor" aria-hidden="true" data-visible="0" data-active="0" data-over-input="0" />;
}

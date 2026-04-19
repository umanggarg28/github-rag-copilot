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
  "input",
  "textarea",
  ".ts-flow-step",
  ".ts-rail-dot",
  ".ts-dep-pill",
  ".suggestion-btn",
  ".try-repo-chip",
  ".ec-ask",
  ".ec-node",
  ".onboarding-step",
].join(", ");

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
      // Upgrade when hovering anything interactive, anywhere in the bubble path.
      const interactive = e.target.closest?.(INTERACTIVE_SELECTOR);
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
  return <div ref={ref} className="custom-cursor" aria-hidden="true" data-visible="0" data-active="0" />;
}

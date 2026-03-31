/**
 * SequenceDiagram.jsx — SVG sequence diagram renderer.
 *
 * Based on UML 2.x conventions used by PlantUML / Mermaid:
 *  - Solid filled-arrowhead  = synchronous call
 *  - Dashed open-arrowhead   = return
 *  - 3-segment right-angle   = self-call
 *  - Thin activation rect    = "object is busy"
 *  - Actors mirrored at bottom so tall diagrams stay readable
 *  - Labels above the arrow, centred, with background pill
 */

// ── Layout constants (derived from Mermaid/PlantUML conventions) ──────────────
const ACTOR_W    = 150;   // actor box width
const ACTOR_H    = 52;    // actor box height
const ACTOR_GAP  = 100;   // gap between actor box edges (not centres)
const MSG_H      = 72;    // vertical space per message row
const PAD_X      = 60;    // outer left/right padding
const PAD_TOP    = 16;    // space above actor boxes
const PAD_BOTTOM = 40;    // space below the last message
const ACT_W      = 12;    // activation bar width (PlantUML default: ~10-14px)

// Arrow taxonomy (UML 2.x):
//   call   → solid line + filled arrowhead   (synchronous invocation)
//   return → dashed line + open arrowhead    (return value)
//   create → solid line + open arrowhead     (object creation)
const TYPE = {
  call:   { dash: null,  filled: true,  color: "#a78bfa", dim: "rgba(167,139,250,0.6)", bg: "rgba(88,28,235,0.25)"  },
  return: { dash: "6 4", filled: false, color: "#2dd4bf", dim: "rgba(45,212,191,0.45)", bg: "rgba(13,148,136,0.2)"  },
  create: { dash: null,  filled: false, color: "#fbbf24", dim: "rgba(251,191,36,0.55)", bg: "rgba(180,83,9,0.25)"   },
};
const FALLBACK = TYPE.call;

export default function SequenceDiagram({ data }) {
  if (!data?.actors?.length) return null;

  const actors   = data.actors;
  const messages = data.messages || [];
  const n        = actors.length;

  // Centre X of each actor column
  const cx = (i) => PAD_X + i * (ACTOR_W + ACTOR_GAP) + ACTOR_W / 2;

  // Total canvas dimensions
  const totalW = PAD_X + n * (ACTOR_W + ACTOR_GAP) - ACTOR_GAP + ACTOR_W + PAD_X;
  const msgAreaH = messages.length * MSG_H;
  // Reserve room for actor boxes top + bottom (mirrored) + messages
  const totalH = PAD_TOP + ACTOR_H + msgAreaH + ACTOR_H + PAD_BOTTOM;

  // Y where message i's arrow sits
  const msgY = (i) => PAD_TOP + ACTOR_H + (i + 0.5) * MSG_H;

  // ── Activation bar calculation ────────────────────────────────────────────
  // An actor is "active" from when it first receives a message to when it
  // sends its corresponding return. We use a simple open/close stack.
  const actBars = {}; // actorIdx → [{y1, y2}]
  actors.forEach((_, i) => { actBars[i] = []; });
  const openStack = {}; // actorIdx → [openMsgIdx, ...]
  actors.forEach((_, i) => { openStack[i] = []; });

  messages.forEach((msg, idx) => {
    const toI   = actors.indexOf(msg.to);
    const fromI = actors.indexOf(msg.from);
    if (msg.type !== "return" && toI !== -1) {
      openStack[toI].push(idx);
    }
    if (msg.type === "return" && fromI !== -1 && openStack[fromI].length > 0) {
      const startIdx = openStack[fromI].pop();
      actBars[fromI].push({ y1: msgY(startIdx), y2: msgY(idx) });
    }
  });
  // Close still-open bars at the last message
  actors.forEach((_, i) => {
    openStack[i].forEach(startIdx => {
      actBars[i].push({ y1: msgY(startIdx), y2: msgY(messages.length - 1) + MSG_H * 0.4 });
    });
  });

  return (
    <div style={{
      width: "100%", height: "100%",
      overflow: "auto", background: "#0d0d1a",
      display: "flex", alignItems: "flex-start", justifyContent: "center",
      padding: "24px",
    }}>
      <svg
        width={totalW} height={totalH}
        style={{ fontFamily: "JetBrains Mono, monospace", minWidth: totalW, display: "block" }}
      >
        <defs>
          {/* Filled arrowhead — synchronous call */}
          {Object.entries(TYPE).map(([type, t]) => (
            t.filled
              ? <marker key={`f-${type}`} id={`arr-f-${type}`} markerWidth="10" markerHeight="8" refX="10" refY="4" orient="auto">
                  <polygon points="0 0, 10 4, 0 8" fill={t.color} />
                </marker>
              : <marker key={`o-${type}`} id={`arr-o-${type}`} markerWidth="10" markerHeight="8" refX="10" refY="4" orient="auto">
                  <polyline points="0 0, 10 4, 0 8" fill="none" stroke={t.color} strokeWidth="1.5" />
                </marker>
          ))}
        </defs>

        {/* ── Actor boxes + lifelines (top + mirrored bottom) ── */}
        {actors.map((actor, i) => {
          const x = cx(i);
          const lifeY1 = PAD_TOP + ACTOR_H;
          const lifeY2 = PAD_TOP + ACTOR_H + msgAreaH;

          return (
            <g key={actor}>
              {/* Dashed lifeline */}
              <line
                x1={x} y1={lifeY1} x2={x} y2={lifeY2}
                stroke="rgba(124,58,237,0.22)" strokeWidth={1.5} strokeDasharray="5 5"
              />

              {/* Activation bars */}
              {actBars[i].map((bar, bi) => (
                <rect
                  key={bi}
                  x={x - ACT_W / 2} y={bar.y1 - 4}
                  width={ACT_W} height={Math.max(bar.y2 - bar.y1 + 8, 16)}
                  rx={3}
                  fill="rgba(139,92,246,0.18)"
                  stroke="rgba(139,92,246,0.55)"
                  strokeWidth={1}
                />
              ))}

              {/* Top actor box */}
              <ActorBox x={x} y={PAD_TOP} label={actor} />

              {/* Bottom actor box (mirrored) — keeps tall diagrams readable */}
              <ActorBox x={x} y={PAD_TOP + ACTOR_H + msgAreaH} label={actor} />
            </g>
          );
        })}

        {/* ── Messages ── */}
        {messages.map((msg, idx) => {
          const fromI = actors.indexOf(msg.from);
          const toI   = actors.indexOf(msg.to);
          if (fromI === -1 || toI === -1) return null;

          const y    = msgY(idx);
          const t    = TYPE[msg.type] || FALLBACK;
          const markerId = t.filled ? `arr-f-${msg.type || "call"}` : `arr-o-${msg.type || "call"}`;

          if (fromI === toI) {
            // ── Self-call: 3-segment right-angle path ──────────────────────
            // Segment: right → down → back left (offset 40px right of lifeline)
            const lx = cx(fromI);
            const offset = 44;
            const topY   = y - MSG_H * 0.18;
            const botY   = y + MSG_H * 0.28;
            const labelX = lx + offset + 8;
            const labelText = msg.label || "";
            return (
              <g key={idx}>
                <path
                  d={`M ${lx} ${topY} H ${lx + offset} V ${botY} H ${lx}`}
                  fill="none"
                  stroke={t.dim}
                  strokeWidth={2}
                  strokeDasharray={t.dash || undefined}
                  markerEnd={`url(#${markerId})`}
                />
                {/* Label above the top segment */}
                <LabelPill x={labelX} y={topY - 10} text={labelText} color={t.color} bg={t.bg} align="left" />
              </g>
            );
          }

          // ── Normal arrow ────────────────────────────────────────────────
          const goRight = toI > fromI;
          // Start/end at activation bar edge (not lifeline centre)
          const x1 = cx(fromI) + (goRight ?  ACT_W / 2 : -ACT_W / 2);
          const x2 = cx(toI)   + (goRight ? -ACT_W / 2 :  ACT_W / 2);
          const midX = (x1 + x2) / 2;

          return (
            <g key={idx}>
              <line
                x1={x1} y1={y} x2={x2} y2={y}
                stroke={t.dim}
                strokeWidth={2}
                strokeDasharray={t.dash || undefined}
                markerEnd={`url(#${markerId})`}
              />
              {/* Label centred above arrow */}
              <LabelPill x={midX} y={y - 10} text={msg.label || ""} color={t.color} bg={t.bg} align="center" />
              {/* Step number — small, at source side below arrow */}
              <text
                x={goRight ? x1 + 4 : x1 - 4}
                y={y + 16}
                textAnchor={goRight ? "start" : "end"}
                fill="rgba(167,139,250,0.35)"
                fontSize={10}
              >{idx + 1}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function ActorBox({ x, y, label }) {
  return (
    <g>
      <rect
        x={x - ACTOR_W / 2} y={y}
        width={ACTOR_W} height={ACTOR_H}
        rx={8}
        fill="#151528"
        stroke="rgba(124,58,237,0.7)"
        strokeWidth={1.5}
      />
      {/* Subtle top accent line */}
      <rect
        x={x - ACTOR_W / 2} y={y}
        width={ACTOR_W} height={3}
        rx={8}
        fill="rgba(124,58,237,0.5)"
      />
      <text
        x={x} y={y + ACTOR_H / 2 + 6}
        textAnchor="middle"
        fill="#e2d9ff"
        fontSize={13}
        fontWeight={700}
      >{label}</text>
    </g>
  );
}

function LabelPill({ x, y, text, color, bg, align }) {
  if (!text) return null;
  const chars  = text.length;
  const pillW  = Math.min(Math.max(chars * 7.8 + 22, 60), 280);
  const pillH  = 22;
  const pillX  = align === "center" ? x - pillW / 2
               : align === "left"   ? x
               : x - pillW;

  return (
    <g>
      <rect x={pillX} y={y - pillH / 2} width={pillW} height={pillH} rx={5} fill={bg} />
      <text
        x={pillX + pillW / 2}
        y={y + 5}
        textAnchor="middle"
        fill={color}
        fontSize={12}
        fontWeight={600}
      >{text.length > 34 ? text.slice(0, 32) + "…" : text}</text>
    </g>
  );
}

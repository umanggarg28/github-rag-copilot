# Cartographer — Claude Code Instructions

This file is read by Claude Code at the start of every session.
It tells Claude how to work in this project.

---

## Project Purpose

A RAG system that indexes GitHub repositories and answers questions about code.
This is a **learning project** — prioritise clarity and explanation over brevity.

---

## Architecture at a Glance

```
ingestion/          ← repo fetching, file filtering, AST chunking, embedding
retrieval/          ← Qdrant hybrid search, BM25 sparse vectors
backend/            ← FastAPI: /ingest, /query, /search endpoints
  services/         ← ingestion_service.py, retrieval_service.py, generation.py
  routers/          ← ingest.py, query.py
  models/           ← schemas.py (Pydantic models)
ui/                 ← React + Vite frontend
notes/              ← Updated after every PR (NNN-title.md)
PLAN.md             ← Build plan and phase tracking
LEARN.md            ← Learning guide, updated as features are built
```

---

## Design System (UI components — enforce strictly)

All UI components must use the token system defined in `ui/src/index.css`.
Never hardcode colours that duplicate a CSS variable.

| Token              | Use for                                         |
|--------------------|--------------------------------------------------|
| `var(--surface-2/3/4)` | Panel backgrounds, cards, elevated surfaces  |
| `var(--border)`    | All dividers and card borders                   |
| `var(--text)` / `var(--text-2)` / `var(--muted)` / `var(--faint)` | Text hierarchy |
| `var(--accent)` / `var(--accent-soft)` / `var(--accent-dim)` / `var(--accent-border)` | Violet accent and interactive states |
| `var(--mono)`      | All monospace / code text                       |
| `var(--transition)` / `var(--transition-slow)` | All CSS transitions       |
| `var(--radius-sm)` / `var(--radius)` / `var(--radius-lg)` | Border radii           |

**Reuse existing button classes — do not invent new ones:**
- `.diagram-ask-btn` — primary action button (violet accent fill)
- `.diagram-retry-btn` — ghost/secondary action button
- `.session-delete` — icon/close button inside `.session-item` only (has `opacity: 0` by default, revealed on parent hover). If using outside `.session-item`, add `style={{ opacity: 1 }}` to override.
- `.ec-ask` — inline "Ask about this →" button on concept cards

**Collapse/expand controls:** use SVG chevrons (same viewport as DiagramView fullscreen icons), never Unicode arrows like ▶/◀.

**Panel layout convention:**
- Detail panels that appear alongside content go at the **bottom** (vertical collapse), not the right side (horizontal collapse). This preserves horizontal canvas space.
- Bottom panels: `height` transition between `44px` (header strip) and content height. Use `overflow: hidden` on the container for clip.

---

## Coding Rules

- Write comments explaining **why**, not what — this is a learning project
- Each new concept gets a docstring explaining it from first principles
- Prefer explicit over implicit — avoid magic
- No LangChain, no LlamaIndex — build from scratch so concepts are visible
- Write comments explaining **why**, not what

## File Hygiene (enforce strictly)

- **Delete superseded files immediately** when a module is replaced or renamed — don't leave orphans
- **No dead routers/modules** — if `backend/routers/` has no active router files, remove it
- **One canonical location per concern** — if two files do the same thing, one must go
- **Before creating a file**, check whether an existing file can be extended instead
- After any refactor, run `find . -name "*.py" | grep -v __pycache__ | grep -v .venv` and confirm every file is still imported or is a valid entry point

---

## Notes Convention

After every significant feature (PR-worthy), add an entry to `notes/`:
- Filename: `NNN-short-title.md` (zero-padded, e.g. `001-ingestion.md`)
- Contents: what was built, key decisions, concepts learned, what's next

---

## Running the Project

```bash
# Backend
cd github-rag-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload

# Frontend
cd ui && npm install && npm run dev
```

---

## Environment Variables

```
# Embeddings (one required)
NOMIC_API_KEY=       # Default. Free at https://atlas.nomic.ai — nomic-embed-text-v1.5 (768-dim)
VOYAGE_API_KEY=      # Optional upgrade. Free at https://voyageai.com — voyage-code-3 (1024-dim)
                     # ⚠️  Switching requires EMBEDDING_MODEL=voyage-code-3, EMBEDDING_DIM=1024,
                     #    and a NEW QDRANT_COLLECTION (dims are incompatible)

# Vector DB (required)
QDRANT_URL=          # Qdrant Cloud cluster URL
QDRANT_API_KEY=      # Qdrant Cloud API key
QDRANT_COLLECTION=   # Default: github_repos_nomic (create a new one if you change dims)

# LLM (at least one required)
CEREBRAS_API_KEY=    # Free at https://cloud.cerebras.ai — llama-3.3-70b, 1M tok/day, 2600 tok/s (FASTEST)
GROQ_API_KEY=        # Free at https://console.groq.com
GEMINI_API_KEY=      # Free at https://aistudio.google.com
OPENROUTER_API_KEY=  # Free tier at https://openrouter.ai
ANTHROPIC_API_KEY=   # Optional paid fallback

# Reranking (optional)
COHERE_API_KEY=      # Free at https://cohere.com (1000 calls/month) — rerank-v3.5
                     # Falls back to local ms-marco cross-encoder if not set

# GitHub (optional but recommended)
GITHUB_TOKEN=        # Increases API rate limit from 60 → 5000 req/hr

# Quality features (enabled by default)
USE_HYDE=true        # Hypothetical Document Embeddings — generate code snippet before searching
EXPAND_QUERIES=true  # Generate 2-3 query variants, search all, merge with RRF
CONTEXTUAL_TOP_N=0   # 0 = contextualise all chunks on force re-index (best quality)

# Deployment (set in HF Space settings)
FRONTEND_URL=        # Your Vercel frontend URL — needed for CORS
```

## Deployment

Backend → **HuggingFace Spaces** (Docker): `Dockerfile` in repo root, port 7860.
Frontend → **Vercel**: set `VITE_API_URL` env var to the HF Space URL.

Set all env vars in HF Space → Settings → Variables (not in the Dockerfile — never commit secrets).

---

## Slash Commands Available

- `/ingest-repo` — ingest a GitHub repository by URL
- `/search-code` — search the index without generating an answer
- `/add-to-notes` — add a note entry for the current work

---

## Key Design Decisions (don't change without good reason)

- **Qdrant Cloud** for vector storage (not ChromaDB) — enables free deployment
- **AST chunking** at function/class boundaries — not character windows
- **nomic-embed-code via Nomic API** — same model as local, but zero RAM cost (enables free hosting)
- **Qdrant native hybrid search** — replaces manual BM25 index + RRF fusion
- **No auth required** for public repo ingestion — GitHub API unauthenticated allows 60 req/hr, with token 5000 req/hr

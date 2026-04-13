"""
diagram_service.py — AST-grounded JSON system design diagrams.

═══════════════════════════════════════════════════════════════
WHAT THIS GENERATES
═══════════════════════════════════════════════════════════════

Four diagram types, each answering a different question:

  architecture  — "What are the main pieces and how do they connect?"
                  JSON: {nodes, edges} node graph

  class         — "What classes exist and how do they relate?"
                  JSON: {nodes, edges} node graph

  sequence      — "What happens step-by-step when the main operation runs?"
                  JSON: {actors, messages} sequence

  dataflow      — "How does data move from input to output?"
                  JSON: {nodes, edges} node graph

Each is generated once and cached. Call invalidate(repo) after re-ingestion.

═══════════════════════════════════════════════════════════════
ARCHITECTURE: STATIC ANALYSIS + LLM DESCRIPTIONS
═══════════════════════════════════════════════════════════════

For architecture, class, and dataflow diagrams the graph structure
(nodes + edges) is built entirely from AST data extracted at ingest time:

  - edges in architecture come from real "import" statements
  - edges in class come from real "class Foo(Bar):" declarations
  - edges in dataflow come from real function call relationships

The LLM's only job is to write a one-sentence description per node.
It cannot invent nodes or connections that don't exist in the code.

For sequence diagrams, execution order genuinely requires understanding the
intent of the code, not just its structure — so the LLM still generates those.
We ground the sequence prompt with real function names from the AST, but the
ordering and message flow are LLM-generated.

═══════════════════════════════════════════════════════════════
WHY JSON INSTEAD OF MERMAID?
═══════════════════════════════════════════════════════════════

React Flow renders interactive node graphs from JSON data directly.
JSON gives us full control over layout, styling, and interactivity
(click-to-chat, zoom, pan). The LLM produces cleaner JSON than Mermaid
syntax, and we can validate + sanitise the output programmatically.
"""

from pathlib import Path

from ingestion.qdrant_store import QdrantStore
from backend.services.generation import GenerationService


def _parse_json(raw: str) -> dict:
    """
    Parse JSON from an LLM response, handling both clean and fence-wrapped output.

    LLMs sometimes emit trailing text after the closing brace (explanatory commentary,
    extra whitespace, a second JSON block). json.loads() rejects this as "Extra data".
    raw_decode() stops at the first complete JSON value and ignores everything after it,
    which is exactly what we want.
    """
    import json as _json, re as _re
    cleaned = raw.strip()
    # Strip markdown fences — ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    # Try raw_decode first — tolerates trailing commentary after the JSON object
    try:
        obj, _ = _json.JSONDecoder().raw_decode(cleaned)
        return obj
    except _json.JSONDecodeError:
        pass

    # Fallback: extract the first {...} block via regex (handles deeply nested fences)
    match = _re.search(r'\{.*\}', cleaned, _re.DOTALL)
    if match:
        try:
            obj, _ = _json.JSONDecoder().raw_decode(match.group(0))
            return obj
        except _json.JSONDecodeError:
            pass

    raise ValueError(f"No JSON found in LLM response: {raw[:200]}")


_DIAGRAM_SYSTEM = (
    "You are an expert software architect analysing codebases. "
    "Return ONLY valid JSON — no markdown fences, no explanation, just the JSON object."
)

_ENRICH_SYSTEM = (
    "You are an expert software architect. Given a list of real code components "
    "extracted by static analysis, write a short description for each one. "
    "Return ONLY valid JSON — no markdown fences, no explanation."
)

_TOUR_SYSTEM = (
    "You are an expert software architect explaining codebases to curious developers. "
    "Your job is to identify the 6-8 core concepts that unlock understanding of any codebase — "
    "the foundational data structures, key algorithms, and architectural patterns that everything else builds on. "
    "You are skilled at reading real source code and extracting the 'aha moments' that make a design click. "
    "Return ONLY valid JSON — no markdown fences, no explanation, just the JSON object."
)

_TOUR_PROMPT = """\
Explain the repository "{repo}" — how it works and what patterns it demonstrates.

Below are the most important code elements from this repo, with their actual source code.
Read the code carefully — your descriptions must reflect what the code literally does, not
what you might guess from the function name alone.

{chunk_summary}

Generate a JSON "concept map" — 6-8 key concepts that explain how this codebase is structured and why.

Surface the concepts that unlock understanding of this specific codebase. Use domain knowledge to pick the right framing:
- ML training codebase → forward pass, loss function, backward pass, optimizer step, data pipeline
- RAG / AI agent codebase → retrieval pipeline, context assembly, agent loop, tool dispatch, LLM call
- Web framework or API → routing, middleware, request lifecycle, data models, auth, persistence
- Compiler / interpreter → lexer, parser, AST, IR, code generation, symbol table
- Game engine → game loop, entity-component system, rendering pipeline, physics, input handling
- Database / storage engine → query planner, storage layout, index structure, transaction log, cache
- CLI tool or DevOps → config parsing, command dispatch, execution engine, output formatting
- Any other codebase → the core data structure it operates on, the main algorithm, and the key abstraction layers

Always pick concepts that a developer must understand in order — from foundation to full picture.

WHAT MAKES A GOOD CONCEPT (prefer these):
  ✓ A core data structure or algorithm the whole codebase is built around
  ✓ A key design pattern that explains WHY the code is structured this way
  ✓ A step in the main processing pipeline that transforms data in an interesting way
  ✓ An abstraction layer that unlocks reuse or extensibility

WHAT TO SKIP (avoid these):
  ✗ Integration glue — thin wrappers that just wire two libraries together (e.g. a client wrapper, a protocol adapter, a config loader)
  ✗ Config / schema files that only define data shapes but contain no interesting logic
  ✗ Utility modules with no algorithmic content
  ✗ Anything whose removal wouldn't change how the core algorithm works

If a concept is just "we use library X to do Y", it's glue — skip it and pick the component that uses the result.

Return ONLY this JSON structure (no markdown, no extra text):
{{
  "summary": "2-sentence plain-English overview: what this repo does and what pattern it demonstrates",
  "entry_point": "the most FOUNDATIONAL file for a new reader — the file that defines the core data structure or central abstraction that all other concepts build on top of. Choose the simplest, most primitive piece, NOT the most complex or feature-rich file. A student with zero context must be able to start here and understand everything else afterward. (just filename, e.g. 'engine.py')",
  "concepts": [
    {{
      "id": 0,
      "name": "Short name (2-4 words)",
      "subtitle": "One-line description of this component's role",
      "file": "filename where this concept lives",
      "type": "class|function|module|algorithm",
      "description": "2-3 sentences: WHAT it does, WHY it was designed this way, and the key insight — the 'aha moment' that makes the design click. Focus on non-obvious decisions.",
      "key_items": ["method_or_function_name_1", "method_or_function_name_2", "method_or_function_name_3"],
      "depends_on": [],
      "reading_order": 1,
      "ask": "A natural question a curious developer would ask about this component (start with 'How', 'Why', 'What', or 'Explain')"
    }}
  ]
}}

Rules:
- Exactly 6-8 concepts — pick the most essential, skip utilities and tests
- PREFER concepts with interesting logic over integration glue — a concept is only worth including if it teaches a pattern
- depends_on: list of concept ids this concept builds on (empty [] for foundational concepts with no prerequisites)
- reading_order: 1-indexed integer — order a student should read to build mental model from scratch
- CRITICAL: reading_order=1 MUST be the concept whose depends_on is [] AND whose file matches entry_point — the simplest foundation, not the most complex file
- CRITICAL: entry_point and reading_order=1 MUST agree — the concept with reading_order=1 must have its file match entry_point
- The concepts must form a DAG — no circular dependencies
- Descriptions must explain WHY this component exists and the key design insight, not just what it does
- key_items: 2-4 actual method or function names found in the indexed code above
- type must be exactly one of: class, function, module, algorithm
"""

_JSON_PROMPTS = {
    "architecture": """\
Analyse the architecture of repository "{repo}" and return a JSON node-graph.

Code elements indexed from this repo:
{chunk_summary}

Return ONLY this JSON (no markdown, no extra text):
{{
  "nodes": [
    {{
      "id": "short_unique_id",
      "label": "Component Name",
      "type": "module|class|service|database|external",
      "file": "filename.py",
      "description": "One sentence: what this component does",
      "items": ["key_method_1", "key_method_2"]
    }}
  ],
  "edges": [
    {{"source": "id1", "target": "id2", "label": "calls|uses|produces|inherits"}}
  ]
}}

Rules:
- 6-10 nodes — the most important architectural components only
- Every node id must be unique, short (snake_case), and referenced correctly in edges
- Edges must only reference node ids that exist
- items: 1-3 actual method/function names from the indexed code
- type must be exactly one of: module, class, service, database, external
""",

    "class": """\
Analyse the class hierarchy of repository "{repo}" and return a JSON node-graph.

Code elements indexed from this repo:
{chunk_summary}

Return ONLY this JSON (no markdown, no extra text):
{{
  "nodes": [
    {{
      "id": "ClassName",
      "label": "ClassName",
      "type": "class|abstract|mixin",
      "file": "filename.py",
      "description": "One sentence: responsibility of this class",
      "items": ["method_1", "method_2", "method_3"]
    }}
  ],
  "edges": [
    {{"source": "Child", "target": "Parent", "label": "inherits"}},
    {{"source": "Container", "target": "Contained", "label": "composes"}},
    {{"source": "User", "target": "Used", "label": "uses"}}
  ]
}}

Rules:
- 5-10 classes — skip utility classes and tests
- Every node id must be unique and match an actual class name from the code
- Edges must only reference node ids that exist
- items: 2-4 actual method names from that class
- type must be exactly one of: class, abstract, mixin
- For inheritance: source=child, target=parent
""",

    "sequence": """\
Analyse the main operation flow of repository "{repo}" and return a JSON sequence diagram.

Code elements indexed from this repo:
{chunk_summary}

Return ONLY this JSON (no markdown, no extra text):
{{
  "actors": ["ActorName1", "ActorName2", "ActorName3"],
  "messages": [
    {{
      "from": "ActorName1",
      "to": "ActorName2",
      "label": "method_name(args)",
      "type": "call|return|create"
    }}
  ]
}}

Rules:
- 3-5 actors maximum (components or classes)
- 8-14 messages total showing the complete main operation (e.g. forward+backward pass, or request→response)
- Every actor in messages must be listed in actors array
- Use real method names from the indexed code
- type: "call" for invocations, "return" for responses, "create" for instantiation
- Order messages chronologically — first message = first thing that happens
""",

    "dataflow": """\
Analyse the data transformation pipeline of repository "{repo}" and return a JSON node-graph.

Code elements indexed from this repo:
{chunk_summary}

Return ONLY this JSON (no markdown, no extra text):
{{
  "nodes": [
    {{
      "id": "short_unique_id",
      "label": "Data State Label",
      "type": "input|transform|output",
      "description": "What the data looks like here: shape, type, or value (e.g. 'float x', 'tensor [B,T,C]', 'loss scalar')",
      "items": ["operation_name"]
    }}
  ],
  "edges": [
    {{"source": "id1", "target": "id2", "label": "operation_name()"}}
  ]
}}

Rules:
- 6-9 nodes showing the complete input→output pipeline
- Flow must go from raw input at top to final output at bottom
- items: the specific method/function that produces this state (1 item max)
- Edge labels: actual method names like "__add__", "relu()", "backward()"
- type must be exactly one of: input, transform, output
""",
}


class DiagramService:
    """
    Generates and caches JSON system design diagrams and codebase tours for indexed repos.

    Each (repo, diagram_type) pair is generated once and cached.
    Call invalidate(repo) after re-ingestion to force a rebuild.

    For architecture/class/dataflow diagrams, edges come from real AST analysis
    (imports, base_classes, calls) extracted at ingest time — not LLM guesses.
    The LLM is called only to write one-sentence descriptions per node.

    For sequence diagrams, the LLM still generates the full diagram (execution
    order requires understanding intent, not just structure), but the prompt is
    grounded with real function names from the AST.
    """

    # Disk cache directory — diagrams are expensive LLM calls, persist them
    # across server restarts so users never wait twice for the same repo.
    _CACHE_DIR = Path(__file__).parent.parent / "diagrams"

    def __init__(self, store: QdrantStore, generation_svc: GenerationService):
        self._store = store
        self._gen   = generation_svc
        self._cache: dict[tuple[str, str], dict] = {}  # (repo, type) → parsed JSON dict
        self._tour_cache: dict[str, dict] = {}          # repo → tour JSON
        self._CACHE_DIR.mkdir(exist_ok=True)

    # ── Disk helpers ──────────────────────────────────────────────────────────

    def _diagram_path(self, repo: str, diagram_type: str) -> Path:
        slug = repo.replace("/", "_")
        return self._CACHE_DIR / f"{slug}_{diagram_type}.json"

    def _tour_path(self, repo: str) -> Path:
        slug = repo.replace("/", "_")
        return self._CACHE_DIR / f"{slug}_tour.json"

    def _load_diagram(self, repo: str, diagram_type: str) -> dict | None:
        """Load a cached diagram from disk into memory. Returns None if not found."""
        import json as _json
        path = self._diagram_path(repo, diagram_type)
        if path.exists():
            try:
                data = _json.loads(path.read_text())
                self._cache[(repo, diagram_type)] = data
                return data
            except Exception:
                path.unlink(missing_ok=True)  # delete corrupt file
        return None

    def _save_diagram(self, repo: str, diagram_type: str, data: dict) -> None:
        import json as _json
        try:
            self._diagram_path(repo, diagram_type).write_text(_json.dumps(data))
        except Exception:
            pass  # disk write failure is non-fatal — memory cache still works

    def _load_tour(self, repo: str) -> dict | None:
        import json as _json
        path = self._tour_path(repo)
        if path.exists():
            try:
                data = _json.loads(path.read_text())
                self._tour_cache[repo] = data
                return data
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def _save_tour(self, repo: str, data: dict) -> None:
        import json as _json
        try:
            self._tour_path(repo).write_text(_json.dumps(data))
        except Exception:
            pass

    def build_diagram(self, repo: str, diagram_type: str) -> dict:
        """
        Return (or generate + cache) a JSON diagram for the given repo and type.

        Returns:
            { "diagram": <parsed_json_dict>, "type": "<diagram_type>" }
            or
            { "error": "<reason>" }
        """
        import json as _json

        if diagram_type not in _JSON_PROMPTS:
            return {"error": f"Unknown diagram type '{diagram_type}'. Valid types: {list(_JSON_PROMPTS.keys())}"}

        cache_key = (repo, diagram_type)
        if cache_key in self._cache:
            return {"diagram": self._cache[cache_key], "type": diagram_type}

        # Check disk cache before hitting the LLM
        disk = self._load_diagram(repo, diagram_type)
        if disk is not None:
            return {"diagram": disk, "type": diagram_type}

        chunks = self._list_chunks(repo)
        if not chunks:
            return {"error": "No indexed chunks found for this repo. Try re-ingesting."}

        if diagram_type == "sequence":
            # Sequence diagrams (execution order) still use LLM — static analysis
            # cannot determine the temporal order of calls reliably.
            # But we ground the prompt with real function names from the call graph.
            data = self._build_sequence_from_llm(repo, chunks)
        else:
            # Architecture, Class, Data Flow: ground-truth structure from AST,
            # LLM only writes node descriptions.
            data = self._build_static_graph(repo, diagram_type)
            if not data or not data.get("nodes"):
                # Fallback to LLM if static analysis yields nothing
                # (e.g. non-Python repo with no AST data)
                data = self._build_diagram_from_llm(repo, diagram_type, chunks)
            else:
                data = self._enrich_nodes(repo, diagram_type, data, chunks)

        if not data:
            return {"error": "Could not generate diagram. Try regenerating."}

        self._cache[cache_key] = data
        self._save_diagram(repo, diagram_type, data)
        return {"diagram": data, "type": diagram_type}

    def build_tour(self, repo: str) -> dict:
        """
        Return (or generate + cache) a codebase tour for the given repo.

        The tour is a structured learning guide: 6-8 key concepts a student
        must understand, with descriptions, dependencies, and reading order.
        Rendered by the frontend as an interactive concept map.

        Returns:
            { "summary": "...", "entry_point": "...", "concepts": [...] }
            or
            { "error": "<reason>" }
        """
        import json as _json

        if repo in self._tour_cache:
            return self._tour_cache[repo]

        disk = self._load_tour(repo)
        if disk is not None:
            return disk

        chunks = self._list_chunks(repo)
        if not chunks:
            return {"error": "No indexed chunks found for this repo. Try re-ingesting."}

        # ── Rank chunks by importance so the LLM reads real code, not just names ──
        # Accuracy problem: if we only send names like "- forward (function) in engine.py"
        # the LLM must guess what the code does from the identifier alone.
        # Fix: score each chunk, pick the top ~35 most important, include their actual
        # source text (truncated). This is how tools like Claude Code / Cursor do it.
        _ENTRY_FILES   = {"main.py", "app.py", "server.py", "index.py", "__init__.py",
                          "agent.py", "run.py", "train.py", "model.py", "pipeline.py"}
        _HIGH_PRIORITY = {"class", "module"}

        def _chunk_score(c: dict) -> int:
            score = 0
            if c["type"] in _HIGH_PRIORITY:
                score += 10
            fname = c["file"].split("/")[-1] if c["file"] else ""
            if fname in _ENTRY_FILES:
                score += 8
            if c["text"]:
                score += 3  # prefer chunks that actually have source text
            if c["base_classes"]:
                score += 2  # class hierarchies are informationally rich
            return score

        ranked = sorted(
            [c for c in chunks if c["name"]],
            key=_chunk_score, reverse=True,
        )

        # Top 50 with real code, rest as name-only stubs (gives context breadth)
        TOP_N        = 50
        STUB_N       = 80
        SNIPPET_LEN  = 700   # chars per snippet — signature + full docstring + key logic

        code_sections = []
        for c in ranked[:TOP_N]:
            snippet = (c["text"] or "").strip()[:SNIPPET_LEN]
            if snippet:
                code_sections.append(
                    f"### {c['name']} ({c['type']}) in {c['file']}\n{snippet}"
                )
            else:
                # No text stored — fall back to name-only line
                code_sections.append(f"- {c['name']} ({c['type']}) in {c['file']}")

        # Include a broader name-only list for the remaining chunks so the LLM
        # can see the full shape of the codebase even without reading every file.
        name_only = "\n".join(
            f"- {c['name']} ({c['type']}) in {c['file']}"
            for c in ranked[TOP_N:TOP_N + STUB_N]
        )

        chunk_summary = "\n\n".join(code_sections)
        if name_only:
            chunk_summary += "\n\n--- Additional elements (names only) ---\n" + name_only

        prompt = _TOUR_PROMPT.format(repo=repo, chunk_summary=chunk_summary)
        # temperature=0.0 — tour must be factual and consistent across regenerations.
        # json_mode=True — forces JSON output on OpenAI-compatible providers so we
        # never have to strip markdown fences or rescue half-parsed responses.
        raw = self._gen.generate(_TOUR_SYSTEM, prompt, temperature=0.0, json_mode=True)

        try:
            tour = _parse_json(raw)
        except ValueError:
            print(f"DiagramService: failed to parse tour JSON:\n{raw[:400]}")
            return {"error": "Could not parse tour from LLM response. Try regenerating."}

        # Validate and sanitise: ensure depends_on only references valid ids
        valid_ids = {c["id"] for c in tour.get("concepts", [])}
        for c in tour.get("concepts", []):
            c["depends_on"] = [d for d in c.get("depends_on", []) if d in valid_ids and d != c["id"]]

        self._tour_cache[repo] = tour
        self._save_tour(repo, tour)
        return tour

    def build_tour_stream(self, repo: str, force: bool = False):
        """
        Generator version of build_tour that yields progress events.

        Each event is a dict: { "stage": str, "progress": float, ...payload }
        The frontend consumes these via SSE to show a live progress bar instead
        of a blank spinner during the 5-10s LLM call.

        Stages:
          loading    (0.10) — fetching chunks from Qdrant
          analysing  (0.35) — ranking chunks by importance
          generating (0.55) — LLM call in progress (the slow part)
          parsing    (0.90) — validating and caching result
          done       (1.00) — full tour data in payload
          error      (1.00) — error message in payload

        force=True skips both memory and disk cache — used by the Regenerate button.
        """
        if force:
            self._tour_cache.pop(repo, None)
            disk_path = self._tour_path(repo)
            if disk_path.exists():
                disk_path.unlink()
        else:
            if repo in self._tour_cache:
                yield {"stage": "done", "progress": 1.0, **self._tour_cache[repo]}
                return

            disk = self._load_tour(repo)
            if disk is not None:
                yield {"stage": "done", "progress": 1.0, **disk}
                return

        yield {"stage": "loading", "progress": 0.10, "message": "Loading repository chunks…"}
        chunks = self._list_chunks(repo)
        if not chunks:
            yield {"stage": "error", "progress": 1.0, "error": "No indexed chunks found for this repo. Try re-ingesting."}
            return

        yield {"stage": "analysing", "progress": 0.35, "message": f"Analysing {len(chunks)} code chunks…"}

        _ENTRY_FILES   = {"main.py", "app.py", "server.py", "index.py", "__init__.py",
                          "agent.py", "run.py", "train.py", "model.py", "pipeline.py"}
        _HIGH_PRIORITY = {"class", "module"}

        def _chunk_score(c: dict) -> int:
            score = 0
            if c["type"] in _HIGH_PRIORITY:
                score += 10
            fname = c["file"].split("/")[-1] if c["file"] else ""
            if fname in _ENTRY_FILES:
                score += 8
            if c["text"]:
                score += 3
            if c["base_classes"]:
                score += 2
            return score

        ranked      = sorted([c for c in chunks if c["name"]], key=_chunk_score, reverse=True)
        TOP_N       = 50
        STUB_N      = 80
        SNIPPET_LEN = 700

        code_sections = []
        for c in ranked[:TOP_N]:
            snippet = (c["text"] or "").strip()[:SNIPPET_LEN]
            if snippet:
                code_sections.append(f"### {c['name']} ({c['type']}) in {c['file']}\n{snippet}")
            else:
                code_sections.append(f"- {c['name']} ({c['type']}) in {c['file']}")

        name_only = "\n".join(
            f"- {c['name']} ({c['type']}) in {c['file']}"
            for c in ranked[TOP_N:TOP_N + STUB_N]
        )

        chunk_summary = "\n\n".join(code_sections)
        if name_only:
            chunk_summary += "\n\n--- Additional elements (names only) ---\n" + name_only

        prompt = _TOUR_PROMPT.format(repo=repo, chunk_summary=chunk_summary)

        yield {"stage": "generating", "progress": 0.55, "message": "Generating concept tour with AI…"}
        raw = self._gen.generate(_TOUR_SYSTEM, prompt, temperature=0.0, json_mode=True)

        yield {"stage": "parsing", "progress": 0.90, "message": "Finalizing…"}
        try:
            tour = _parse_json(raw)
        except ValueError:
            yield {"stage": "error", "progress": 1.0, "error": "Could not parse tour from LLM response. Try regenerating."}
            return

        valid_ids = {c["id"] for c in tour.get("concepts", [])}
        for c in tour.get("concepts", []):
            c["depends_on"] = [d for d in c.get("depends_on", []) if d in valid_ids and d != c["id"]]

        self._tour_cache[repo] = tour
        self._save_tour(repo, tour)
        yield {"stage": "done", "progress": 1.0, **tour}

    def build_diagram_stream(self, repo: str, diagram_type: str, force: bool = False):
        """
        Generator version of build_diagram that yields progress events.

        Stages:
          loading    (0.10) — fetching chunks
          building   (0.40) — static AST graph construction
          enriching  (0.70) — LLM call for node descriptions
          done       (1.00) — full diagram data in payload
          error      (1.00) — error message in payload

        force=True skips both memory and disk cache — used by the Regenerate button.
        """
        import json as _json

        if diagram_type not in _JSON_PROMPTS:
            yield {"stage": "error", "progress": 1.0,
                   "error": f"Unknown diagram type '{diagram_type}'."}
            return

        cache_key = (repo, diagram_type)

        if force:
            # Bust both caches so a fresh diagram is generated
            self._cache.pop(cache_key, None)
            disk_path = self._diagram_path(repo, diagram_type)
            if disk_path.exists():
                disk_path.unlink()
        else:
            if cache_key in self._cache:
                yield {"stage": "done", "progress": 1.0,
                       "diagram": self._cache[cache_key], "type": diagram_type}
                return

            disk = self._load_diagram(repo, diagram_type)
            if disk is not None:
                yield {"stage": "done", "progress": 1.0, "diagram": disk, "type": diagram_type}
                return

        yield {"stage": "loading", "progress": 0.10, "message": "Loading repository chunks…"}
        chunks = self._list_chunks(repo)
        if not chunks:
            yield {"stage": "error", "progress": 1.0,
                   "error": "No indexed chunks found for this repo. Try re-ingesting."}
            return

        yield {"stage": "building", "progress": 0.40, "message": "Building graph from AST…"}
        if diagram_type == "sequence":
            data = self._build_sequence_from_llm(repo, chunks)
        else:
            data = self._build_static_graph(repo, diagram_type)
            if not data or not data.get("nodes"):
                yield {"stage": "enriching", "progress": 0.70,
                       "message": "Generating diagram with AI…"}
                data = self._build_diagram_from_llm(repo, diagram_type, chunks)
            else:
                yield {"stage": "enriching", "progress": 0.70,
                       "message": "Enriching node descriptions with AI…"}
                data = self._enrich_nodes(repo, diagram_type, data, chunks)

        if not data:
            yield {"stage": "error", "progress": 1.0,
                   "error": "Could not generate diagram. Try regenerating."}
            return

        self._cache[cache_key] = data
        self._save_diagram(repo, diagram_type, data)
        yield {"stage": "done", "progress": 1.0, "diagram": data, "type": diagram_type}

    def invalidate(self, repo: str):
        """Remove all cached diagrams and tours for a repo — call after re-ingestion."""
        keys = [k for k in self._cache if k[0] == repo]
        for k in keys:
            del self._cache[k]
        self._tour_cache.pop(repo, None)
        # Delete disk files so the next request regenerates from fresh AST data
        for diagram_type in ("architecture", "class"):
            self._diagram_path(repo, diagram_type).unlink(missing_ok=True)
        self._tour_path(repo).unlink(missing_ok=True)

    # ── Static graph builders (AST-grounded) ──────────────────────────────────

    def _build_static_graph(self, repo: str, diagram_type: str) -> dict:
        """
        Build a ground-truth node+edge graph from statically-extracted AST data.

        This is the core accuracy improvement: edges come from real code analysis,
        not LLM guessing. The LLM is only called afterwards to write descriptions.

        Returns {"nodes": [...], "edges": [...]} with real structure but
        placeholder descriptions (filled by _enrich_nodes).
        """
        chunks = self._list_chunks(repo)
        if not chunks:
            return {}

        if diagram_type == "class":
            return self._build_class_graph(chunks)
        elif diagram_type == "architecture":
            return self._build_arch_graph(chunks)
        elif diagram_type == "dataflow":
            return self._build_dataflow_graph(chunks)
        return {}

    def _build_class_graph(self, chunks: list[dict]) -> dict:
        """
        Build class hierarchy from real base_classes extracted at ingest time.
        Edges are 100% accurate — from "class Foo(Bar):" in the actual code.
        """
        nodes = {}
        raw_edges = []

        for chunk in chunks:
            if chunk["type"] != "class":
                continue
            name = chunk["name"]
            if not name or "." in name:  # skip split method chunks like "MLP.forward"
                continue
            # Accumulate method names from sibling function chunks in same file
            methods = [
                c["name"].split(".")[-1]
                for c in chunks
                if c["type"] == "function"
                and c["file"] == chunk["file"]
                and "." in c["name"]
                and c["name"].split(".")[0] == name
            ]
            nodes[name] = {
                "id":          name,
                "label":       name,
                "type":        "class",
                "file":        chunk["file"].split("/")[-1],   # just filename, not full path
                "description": "",    # filled by _enrich_nodes
                "items":       methods[:4],
            }
            for base in chunk.get("base_classes", []):
                raw_edges.append({"source": name, "target": base, "label": "inherits"})

        # Only keep inheritance edges where both endpoints are in the repo.
        # External bases like nn.Module are dropped here — they aren't repo nodes.
        valid = set(nodes.keys())
        edges = [e for e in raw_edges if e["source"] in valid and e["target"] in valid]

        # Add composition edges: when class A's code calls class B's constructor
        # (i.e. B appears in A's calls list and B is a known repo class), A "uses" B.
        #
        # Why this matters: repos like nanogpt have zero intra-repo inheritance
        # (all classes inherit nn.Module which is external), but rich composition —
        # Block instantiates CausalSelfAttention, MLP, and LayerNorm. Without these
        # edges the class diagram shows floating disconnected nodes.
        #
        # The `calls` field on class chunks comes from walking the *entire* class
        # body with ast.walk, so it captures constructor calls inside __init__.
        seen_comp = set()
        for chunk in chunks:
            if chunk["type"] != "class":
                continue
            src = chunk["name"]
            if "." in src or src not in valid:
                continue
            for called in chunk.get("calls", []):
                if called in valid and called != src:
                    key = (src, called)
                    if key not in seen_comp:
                        seen_comp.add(key)
                        edges.append({"source": src, "target": called, "label": "uses"})

        return {"nodes": list(nodes.values()), "edges": edges}

    def _resolve_imports(self, file_chunks: list[dict], all_files: set[str]) -> list[tuple[str, str]]:
        """
        Resolve import strings to actual file paths within the repo.

        Example:
          file: "micrograd/nn.py"
          import: "micrograd.engine"
          → resolves to "micrograd/engine.py" (exists in all_files)
          → edge: ("micrograd/nn.py", "micrograd/engine.py")

        Returns list of (source_filepath, target_filepath) tuples.
        Only returns edges where BOTH files are in all_files (repo-internal only).
        External deps (numpy, torch, etc.) are silently dropped.
        """
        edges = []
        for chunk in file_chunks:
            if chunk["type"] != "module":
                continue
            src_file = chunk["file"]
            for imp in chunk.get("imports", []):
                # Handle relative imports: ".engine" from "micrograd/nn.py" → "micrograd/engine.py"
                if imp.startswith("."):
                    # Get the package directory of the source file
                    src_dir = "/".join(src_file.split("/")[:-1])
                    # Strip leading dots — one dot = same package, two = parent, etc.
                    dots = len(imp) - len(imp.lstrip("."))
                    rel   = imp.lstrip(".")
                    parts = src_dir.split("/")
                    # Go up (dots-1) levels for relative imports
                    if dots > 1:
                        parts = parts[:-(dots - 1)] if dots - 1 <= len(parts) else []
                    candidate_dir  = "/".join(parts) if parts else ""
                    candidate_path = f"{candidate_dir}/{rel.replace('.', '/')}.py".lstrip("/")
                    if candidate_path in all_files:
                        edges.append((src_file, candidate_path))
                else:
                    # Absolute import — try matching against known files
                    # "micrograd.engine" → "micrograd/engine.py"
                    # "micrograd.engine" → "engine.py" (flat layout)
                    candidate1 = imp.replace(".", "/") + ".py"
                    candidate2 = imp.split(".")[-1] + ".py"
                    # Also try as package __init__: "micrograd" → "micrograd/__init__.py"
                    candidate3 = imp.replace(".", "/") + "/__init__.py"
                    for candidate in (candidate1, candidate2, candidate3):
                        # Match against all_files (partial suffix match)
                        for f in all_files:
                            if f == candidate or f.endswith("/" + candidate):
                                edges.append((src_file, f))
                                break
        # Deduplicate and remove self-loops
        return list({(s, t) for s, t in edges if s != t})

    def _build_arch_graph(self, chunks: list[dict]) -> dict:
        """
        Build file-level architecture graph from real import statements.
        Edges come from "import X" / "from X import Y" in the actual code.
        Only includes repo-internal dependencies (filters out numpy, torch, etc.)
        Capped at 10 most important files to keep the diagram readable.
        """
        _SKIP_NAMES = {"__init__.py", "setup.py", "conftest.py"}
        _SKIP_PREFIXES = ("test_", ".")

        # Collect all Python files in the repo
        all_files = {c["file"] for c in chunks if c["file"].endswith(".py")}

        # Build per-file node info
        file_info = {}
        for chunk in chunks:
            fp = chunk["file"]
            if not fp.endswith(".py"):
                continue
            filename = fp.split("/")[-1]
            if filename in _SKIP_NAMES or any(filename.startswith(p) for p in _SKIP_PREFIXES):
                continue
            if fp not in file_info:
                file_info[fp] = {"classes": [], "functions": [], "imports": []}
            if chunk["type"] == "class" and chunk["name"] and "." not in chunk["name"]:
                file_info[fp]["classes"].append(chunk["name"])
            elif chunk["type"] == "function" and chunk["name"] and "." not in chunk["name"]:
                file_info[fp]["functions"].append(chunk["name"])
            elif chunk["type"] == "module":
                file_info[fp]["imports"].extend(chunk.get("imports", []))

        # Resolve imports to real edges between repo files
        import_edges = self._resolve_imports(chunks, all_files)

        # Score each file by importance:
        #   classes are worth 3 points (more architectural significance),
        #   functions worth 1 point, being imported by others worth 2 points.
        # This ensures the 10-node cap keeps the most meaningful files.
        import_targets = {}
        for src, tgt in import_edges:
            import_targets[tgt] = import_targets.get(tgt, 0) + 2

        def importance(fp):
            info = file_info[fp]
            return len(info["classes"]) * 3 + len(info["functions"]) + import_targets.get(fp, 0)

        # Keep the 10 most important files
        top_files = sorted(file_info.keys(), key=importance, reverse=True)[:10]
        top_set   = set(top_files)

        # Build nodes for top files only
        nodes = {}
        for fp in top_files:
            info     = file_info[fp]
            filename = fp.split("/")[-1]
            items    = (info["classes"] + info["functions"])[:4]
            nodes[fp] = {
                "id":          fp.replace("/", "_").replace(".", "_"),
                "label":       filename,
                "type":        "module",
                "file":        fp,
                "description": "",
                "items":       items,
            }

        # Map filepath → node id
        fp_to_id = {fp: n["id"] for fp, n in nodes.items()}

        # Build edges — only between files that made the top-10 cut
        edges = []
        seen  = set()
        for src, tgt in import_edges:
            if src not in top_set or tgt not in top_set:
                continue
            sid = fp_to_id.get(src)
            tid = fp_to_id.get(tgt)
            if sid and tid and (sid, tid) not in seen:
                seen.add((sid, tid))
                edges.append({"source": sid, "target": tid, "label": "imports"})

        return {"nodes": list(nodes.values()), "edges": edges}

    def _build_dataflow_graph(self, chunks: list[dict]) -> dict:
        """
        Build data flow graph from the call graph (who calls what).

        We trace from entry-point functions outward using real 'calls' data.
        The LLM then annotates what data looks like at each stage.
        Edges are real function calls extracted from AST — not invented.
        """
        # Build call graph: name → {file, calls}
        fn_map = {}
        for chunk in chunks:
            if chunk["type"] in ("function", "class") and chunk["name"]:
                name = chunk["name"].split(".")[-1]  # strip class prefix
                if name not in fn_map:
                    fn_map[name] = {"file": chunk["file"], "calls": set()}
                fn_map[name]["calls"].update(chunk.get("calls", []))

        # Find entry points: functions that are called by others but also call others
        # (i.e., middle-of-chain functions), plus top-level starters
        all_called = {c for info in fn_map.values() for c in info["calls"]}
        # Entry points = functions NOT called by anyone else (they're the starts)
        entry_points = [n for n in fn_map if n not in all_called and fn_map[n]["calls"]]

        # If no clear entry, fall back to functions with the most outgoing calls
        if not entry_points:
            entry_points = sorted(fn_map.keys(), key=lambda n: len(fn_map[n]["calls"]), reverse=True)

        # BFS from entry points, up to 8 nodes
        visited = []
        queue   = entry_points[:3]
        seen    = set()
        while queue and len(visited) < 8:
            node = queue.pop(0)
            if node in seen or node not in fn_map:
                continue
            seen.add(node)
            visited.append(node)
            # Add called functions to queue (sorted by name for determinism)
            for callee in sorted(fn_map[node]["calls"]):
                if callee in fn_map and callee not in seen:
                    queue.append(callee)

        if not visited:
            return {}

        nodes = []
        for name in visited:
            info = fn_map[name]
            # Classify as input/transform/output based on position in the chain
            callers_count = sum(1 for n in fn_map if name in fn_map[n]["calls"])
            if callers_count == 0:
                ntype = "input"
            elif not fn_map[name]["calls"].intersection(set(visited)):
                ntype = "output"
            else:
                ntype = "transform"

            nodes.append({
                "id":          name,
                "label":       name,
                "type":        ntype,
                "file":        info["file"].split("/")[-1],
                "description": "",
                "items":       [name],
            })

        node_ids = {n["id"] for n in nodes}
        edges = []
        seen_edges = set()
        for n in nodes:
            for callee in sorted(fn_map.get(n["id"], {}).get("calls", [])):
                if callee in node_ids and (n["id"], callee) not in seen_edges:
                    seen_edges.add((n["id"], callee))
                    edges.append({"source": n["id"], "target": callee, "label": "calls"})

        return {"nodes": nodes, "edges": edges}

    def _enrich_nodes(self, repo: str, diagram_type: str, graph: dict, chunks: list[dict] | None = None) -> dict:
        """
        Ask the LLM to write a short description for each node.

        The graph structure (nodes + edges) comes from static analysis and is
        already accurate. The LLM's only job here is to write readable descriptions
        — it cannot invent or change the structure.

        chunks: the full list from _list_chunks, used to supply actual source code
        snippets per node. Without this, the LLM only sees the node name and file
        and has to guess what the component does — the most common source of
        inaccurate descriptions. With snippets, descriptions are grounded in real code.
        """
        import json as _json

        nodes = graph.get("nodes", [])
        if not nodes:
            return graph

        # Two lookups so we can attach real code for both diagram types:
        #
        # snippet_by_name  — name → snippet (class diagrams: node id == class name)
        # snippets_by_file — filepath → top snippets joined (architecture diagrams:
        #                    node id is a sanitised filepath, not a symbol name)
        SNIPPET_LEN      = 400   # chars per individual snippet
        FILE_SNIPPET_LEN = 600   # chars for combined file summary (top 2 chunks)

        snippet_by_name:  dict[str, str]       = {}
        snippets_by_file: dict[str, list[str]] = {}
        if chunks:
            for c in chunks:
                name = c.get("name", "")
                fp   = c.get("file", "")
                text = (c.get("text") or "").strip()
                if name and text and name not in snippet_by_name:
                    snippet_by_name[name] = text[:SNIPPET_LEN]
                if fp and text:
                    snippets_by_file.setdefault(fp, []).append(text[:200])

        # Build node list, attaching a code snippet where available
        node_parts = []
        for n in nodes:
            nid   = n["id"]
            label = n["label"]
            ftype = n.get("type", "")
            ffile = n.get("file", "")  # full filepath for arch nodes, short name for class nodes

            # Try name-based lookup first (class diagrams), then file-based (arch diagrams)
            snippet = snippet_by_name.get(nid) or snippet_by_name.get(label)
            if not snippet and ffile:
                file_snippets = snippets_by_file.get(ffile, [])
                if file_snippets:
                    snippet = "\n---\n".join(file_snippets[:2])[:FILE_SNIPPET_LEN]

            if snippet:
                node_parts.append(
                    f'### {nid} ({ftype}) in {ffile}\n{snippet}'
                )
            else:
                node_parts.append(
                    f'- id="{nid}" label="{label}" file="{ffile}" type="{ftype}"'
                )

        node_list = "\n\n".join(node_parts)

        prompt = (
            f'Repository: {repo}\n'
            f'Diagram type: {diagram_type}\n\n'
            f'These components were identified by static code analysis. '
            f'Source code snippets are included where available — read them carefully.\n\n'
            f'{node_list}\n\n'
            f'For each component, write a 1-sentence description explaining what it does.\n'
            f'Base your descriptions on the source code, not the names alone.\n'
            f'Return ONLY this JSON (no markdown):\n'
            f'{{"descriptions": {{"<id>": "<1 sentence description>", ...}}}}'
        )

        try:
            # Don't use json_mode=True here — gemma-4-31b-it via the Gemini OpenAI-compat
            # endpoint may not support response_format=json_object and will throw silently.
            # The prompt already says "Return ONLY this JSON" and _parse_json handles fences.
            raw  = self._gen.generate(_ENRICH_SYSTEM, prompt, temperature=0.0)
            data = _parse_json(raw)
            descriptions = data.get("descriptions", {})
            # Build a lowercase lookup so LLM key capitalisation differences don't break matching
            # (e.g. LLM returns "value" but node id is "Value")
            desc_lower = {k.lower(): v for k, v in descriptions.items()}
            # Apply descriptions to nodes — exact match first, then case-insensitive fallback
            for n in nodes:
                desc = descriptions.get(n["id"]) or desc_lower.get(n["id"].lower())
                if desc:
                    n["description"] = desc
            matched = sum(1 for n in nodes if n.get("description"))
            print(f"DiagramService: enriched {matched}/{len(nodes)} nodes with descriptions")
        except Exception as e:
            print(f"DiagramService: enrichment failed (non-fatal): {e}")
            # Descriptions stay empty — diagram still shows accurate structure

        return graph

    # ── LLM-based builders ────────────────────────────────────────────────────

    def _build_sequence_from_llm(self, repo: str, chunks: list[dict]) -> dict:
        """Generate sequence diagram via LLM, grounded with real function names."""
        import json as _json

        # Extract real function names for grounding
        fn_names = sorted({
            c["name"] for c in chunks
            if c["type"] == "function" and c["name"] and "." not in c["name"]
        })[:30]
        fn_hint = ", ".join(fn_names) if fn_names else "(none found)"

        sample = chunks[:80]
        chunk_summary = "\n".join(
            f"- {c['name']} ({c['type']}) in {c['file']}"
            for c in sample if c["name"]
        )

        # Append real function names to ground the LLM
        grounded_summary = chunk_summary + f"\n\nReal function names in this repo: {fn_hint}"
        prompt = _JSON_PROMPTS["sequence"].format(repo=repo, chunk_summary=grounded_summary)
        raw = self._gen.generate(_DIAGRAM_SYSTEM, prompt, temperature=0.0, json_mode=True)

        try:
            data = _parse_json(raw)
        except ValueError:
            return {}

        actors = set(data.get("actors", []))
        data["messages"] = [
            m for m in data.get("messages", [])
            if m.get("from") in actors and m.get("to") in actors
        ]
        return data

    def _build_diagram_from_llm(self, repo: str, diagram_type: str, chunks: list[dict]) -> dict:
        """LLM fallback for non-Python repos where static analysis yields nothing."""
        import json as _json

        sample = chunks[:80]
        chunk_summary = "\n".join(
            f"- {c['name']} ({c['type']}) in {c['file']}"
            for c in sample if c["name"]
        )
        prompt = _JSON_PROMPTS[diagram_type].format(repo=repo, chunk_summary=chunk_summary)
        raw = self._gen.generate(_DIAGRAM_SYSTEM, prompt, temperature=0.0, json_mode=True)

        try:
            data = _parse_json(raw)
        except ValueError:
            return {}

        valid_ids = {n["id"] for n in data.get("nodes", [])}
        data["edges"] = [
            e for e in data.get("edges", [])
            if e.get("source") in valid_ids and e.get("target") in valid_ids
        ]
        return data

    # ── Private helpers ────────────────────────────────────────────────────────

    def _list_chunks(self, repo: str) -> list[dict]:
        """
        Scroll Qdrant for all chunks belonging to this repo.

        Returns dicts with all AST-extracted fields needed for static graph
        building: name, type, file, calls, imports, base_classes.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = []
        offset  = None
        filt    = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))])

        while True:
            points, offset = self._store.client.scroll(
                collection_name=self._store.collection,
                scroll_filter=filt,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                pay = p.payload or {}
                results.append({
                    "name":         pay.get("name") or pay.get("function_name") or "",
                    "type":         pay.get("chunk_type", "function"),
                    "file":         pay.get("filepath", ""),
                    "text":         pay.get("text", ""),   # actual source code — used by build_tour for accuracy
                    "calls":        pay.get("calls", []),
                    "imports":      pay.get("imports", []),
                    "base_classes": pay.get("base_classes", []),
                })
            if offset is None:
                break

        return results

"""
graph_service.py — Build a call graph from the indexed code chunks.

═══════════════════════════════════════════════════════════════
WHAT IS A CALL GRAPH?
═══════════════════════════════════════════════════════════════

A call graph is a directed graph where:
  - Nodes  = functions and classes
  - Edges  = "A calls B" relationships

Example:
  train() → forward() → relu() → max()
              ↓
           backward() → topological_sort()

Reading a call graph reveals:
  - Entry points  (nodes with no incoming edges — things called from outside)
  - Hub functions (nodes with many incoming edges — highly reused utilities)
  - Leaf functions (nodes with no outgoing edges — implementation primitives)
  - Cycles         (mutual recursion — usually a design smell)

═══════════════════════════════════════════════════════════════
HOW WE BUILD IT
═══════════════════════════════════════════════════════════════

During ingestion, `_CallExtractor` (in code_chunker.py) walks each
function's AST and records the names of every function it calls.
These names are stored in the `calls` field of each Qdrant point.

At graph request time, GraphService:
  1. Fetches all points for the repo from Qdrant (via scroll_repo)
  2. Builds a name → node_id lookup index
  3. For each node, resolves its `calls` list to actual node IDs
  4. Returns {nodes, edges} in the format D3.js expects

Edge resolution is fuzzy: a call to "backward" will match a node
named "Value.backward" (class.method format from _split_class).
This handles the common pattern of methods being called without
their class qualifier.

═══════════════════════════════════════════════════════════════
GRAPH DATA FORMAT (what the API returns)
═══════════════════════════════════════════════════════════════

{
  "nodes": [
    {
      "id":         "micrograd/engine.py::Value.backward",   # unique key
      "name":       "Value.backward",
      "filepath":   "micrograd/engine.py",
      "chunk_type": "function",
      "start_line": 82,
      "end_line":   101,
      "calls_count": 2,     # outgoing edges (how much it calls)
      "caller_count": 5,    # incoming edges (how much it's called) → node size
    }
  ],
  "edges": [
    {
      "source": "micrograd/engine.py::Value.backward",
      "target": "micrograd/engine.py::Value._topological_sort",
    }
  ],
  "stats": {
    "node_count": 42,
    "edge_count": 87,
    "python_only": true    # call extraction only works for Python
  }
}
"""

from collections import defaultdict
from ingestion.qdrant_store import QdrantStore


class GraphService:
    """
    Builds a call graph for a given repo from the indexed Qdrant data.

    The graph is built lazily — we scroll through all stored points and
    construct edges in memory. For a typical repo (100–500 functions),
    this takes <1 second. Results are not cached — call data changes when
    the repo is re-indexed.
    """

    def __init__(self, store: QdrantStore):
        self.store = store

    def build_graph(self, repo: str) -> dict:
        """
        Build and return the call graph for a repo.

        Steps:
          1. Fetch all points (name, filepath, chunk_type, calls, line numbers)
          2. Build a name → node_id index (handles "foo" matching "Class.foo")
          3. Resolve call names to node IDs, emit edges
          4. Annotate nodes with in-degree (caller_count) for D3 sizing
        """
        # ── Step 1: fetch all points ──────────────────────────────────────────
        points = self.store.scroll_repo(
            repo,
            with_payload=["name", "filepath", "chunk_type", "calls", "start_line", "end_line"],
        )

        # Filter to named chunks (functions and classes) — skip module/text
        named = [
            p for p in points
            if p.get("name") and p.get("chunk_type") in ("function", "class")
        ]

        if not named:
            return {"nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0}}

        # ── Step 2: build name lookup index ──────────────────────────────────
        # node_id format: "filepath::name"
        # We build TWO lookup tables:
        #   exact:  "Value.backward" → node_id
        #   suffix: "backward"       → node_id  (for unqualified calls)
        #
        # The suffix index allows "forward()" in train.py to match
        # "Model.forward" in model.py — this is the common case.
        exact_lookup:  dict[str, str] = {}   # full name → id
        suffix_lookup: dict[str, list[str]] = defaultdict(list)  # leaf → [ids]

        nodes: dict[str, dict] = {}

        for p in named:
            name      = p["name"]
            filepath  = p["filepath"]
            node_id   = f"{filepath}::{name}"

            nodes[node_id] = {
                "id":           node_id,
                "name":         name,
                "filepath":     filepath,
                "chunk_type":   p.get("chunk_type", "function"),
                "start_line":   p.get("start_line", 0),
                "end_line":     p.get("end_line", 0),
                "caller_count": 0,  # filled in step 4
            }

            exact_lookup[name] = node_id

            # For "Class.method" → also index the leaf "method"
            leaf = name.rsplit(".", 1)[-1]
            suffix_lookup[leaf].append(node_id)

        # ── Step 3: resolve calls → edges ────────────────────────────────────
        edges: list[dict] = []
        seen_edges: set[tuple] = set()

        for p in named:
            source_id = f"{p['filepath']}::{p['name']}"
            calls = p.get("calls") or []

            for call_name in calls:
                # Try exact match first, then suffix match
                target_id = exact_lookup.get(call_name)

                if not target_id:
                    candidates = suffix_lookup.get(call_name, [])
                    if len(candidates) == 1:
                        target_id = candidates[0]
                    elif len(candidates) > 1:
                        # Multiple candidates — prefer same file
                        same_file = [c for c in candidates if c.startswith(p["filepath"])]
                        target_id = same_file[0] if same_file else candidates[0]

                if target_id and target_id != source_id:
                    edge_key = (source_id, target_id)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({"source": source_id, "target": target_id})

        # ── Step 4: annotate nodes with in-degree ────────────────────────────
        # caller_count (in-degree) is used by D3 to size nodes:
        # functions called by many places are rendered larger.
        for edge in edges:
            if edge["target"] in nodes:
                nodes[edge["target"]]["caller_count"] += 1

        # Convert nodes dict to list, sorted by caller_count desc (hub nodes first)
        node_list = sorted(nodes.values(), key=lambda n: n["caller_count"], reverse=True)

        return {
            "nodes": node_list,
            "edges": edges,
            "stats": {
                "node_count":  len(node_list),
                "edge_count":  len(edges),
                "python_only": True,
            },
        }

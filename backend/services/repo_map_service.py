"""
repo_map_service.py — Build and cache a compact metadata map of an indexed repo.

WHY THIS EXISTS
───────────────
Every agent session starts cold — the agent rediscovers "trainer.py is the entry
point" or "GPT is in model.py" on every question, wasting 2-4 tool-call turns
before it can even start answering.

A RepoMapService scans Qdrant chunk metadata (no LLM calls, no text vectors) and
builds a compact JSON map: entry files, key classes, per-file breakdown. This is
injected into the agent's user message at session start so it already knows the
repo layout — the same way Claude Code loads CLAUDE.md before any conversation.

TOKEN COST
──────────
~300 tokens per session — roughly one search_code call.
Zero LLM API calls to generate — pure Qdrant metadata scan.

STORAGE
───────
Maps are saved to backend/repo_maps/{owner}_{name}.json.
Call invalidate(repo) after re-ingestion to force a rebuild.
"""

import json
from datetime import datetime
from pathlib import Path

_MAPS_DIR = Path(__file__).parent.parent / "repo_maps"

# Filenames that strongly suggest entry points
_ENTRY_NAMES = {
    "main.py", "app.py", "server.py", "index.py", "run.py",
    "train.py", "model.py", "agent.py", "pipeline.py", "cli.py",
}


class RepoMapService:
    """Builds, caches, and formats compact repo metadata maps from Qdrant."""

    def __init__(self, qdrant_store):
        self._store = qdrant_store
        _MAPS_DIR.mkdir(exist_ok=True)

    def _map_path(self, repo: str) -> Path:
        safe = repo.replace("/", "_").replace(" ", "_")
        return _MAPS_DIR / f"{safe}.json"

    def get_or_build(self, repo: str) -> dict:
        """Return cached map if it exists; otherwise scan Qdrant and build one."""
        path = self._map_path(repo)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass  # corrupt file — rebuild
        return self._build_and_save(repo)

    def invalidate(self, repo: str) -> None:
        """Delete the cached map — call after re-ingestion so stale data isn't served."""
        path = self._map_path(repo)
        if path.exists():
            path.unlink()

    def _build_and_save(self, repo: str) -> dict:
        """
        Scan Qdrant chunk payloads and assemble a compact repo map.

        Only reads lightweight metadata fields (filepath, name, chunk_type, imports) —
        NOT the text or embedding fields — so it completes quickly even for large repos.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        raw_chunks, offset = [], None
        filt = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))])

        while True:
            points, offset = self._store.client.scroll(
                collection_name=self._store.collection,
                scroll_filter=filt,
                limit=500,
                offset=offset,
                with_payload=["filepath", "name", "chunk_type", "imports"],
                with_vectors=False,
            )
            for p in points:
                pay = p.payload or {}
                raw_chunks.append({
                    "file":    pay.get("filepath", ""),
                    "name":    pay.get("name") or pay.get("function_name") or "",
                    "type":    pay.get("chunk_type", "function"),
                    "imports": pay.get("imports", []),
                })
            if offset is None:
                break

        if not raw_chunks:
            return {}

        # Group by file
        files: dict[str, dict] = {}
        for c in raw_chunks:
            fp = c["file"]
            if not fp:
                continue
            if fp not in files:
                files[fp] = {"classes": [], "functions": [], "imports": []}
            if c["type"] == "class" and c["name"] and "." not in c["name"]:
                if c["name"] not in files[fp]["classes"]:
                    files[fp]["classes"].append(c["name"])
            elif c["type"] == "function" and c["name"] and "." not in c["name"]:
                if c["name"] not in files[fp]["functions"]:
                    files[fp]["functions"].append(c["name"])
            files[fp]["imports"].extend(c.get("imports", []))

        # Score files: known entry-point names + how often imported by other files
        import_counts: dict[str, int] = {}
        for fp, info in files.items():
            for imp in info["imports"]:
                stem = imp.split(".")[-1]
                for other_fp in files:
                    if other_fp.split("/")[-1].replace(".py", "") == stem:
                        import_counts[other_fp] = import_counts.get(other_fp, 0) + 1

        def _score(fp: str) -> int:
            s = import_counts.get(fp, 0) * 2
            s += len(files[fp]["classes"]) * 2 + len(files[fp]["functions"])
            if fp.split("/")[-1] in _ENTRY_NAMES:
                s += 6
            return s

        top_files = sorted(files.keys(), key=_score, reverse=True)[:12]

        entry_files = [fp for fp in files if fp.split("/")[-1] in _ENTRY_NAMES][:5]

        key_classes: list[str] = []
        for fp in top_files:
            for cls in files[fp]["classes"]:
                if cls not in key_classes:
                    key_classes.append(cls)

        repo_map = {
            "repo":         repo,
            "updated_at":   datetime.utcnow().isoformat(),
            "total_chunks": len(raw_chunks),
            "total_files":  len(files),
            "entry_files":  entry_files,
            "key_classes":  key_classes[:12],
            "files": {
                fp: {
                    "classes":   files[fp]["classes"][:6],
                    "functions": files[fp]["functions"][:8],
                }
                for fp in top_files
            },
        }

        try:
            self._map_path(repo).write_text(json.dumps(repo_map, indent=2))
        except Exception as e:
            print(f"RepoMapService: could not save map for {repo}: {e}")

        return repo_map

    def format_for_prompt(self, repo_map: dict) -> str:
        """
        Render the map as a compact text block for the agent's context.

        Kept under 400 tokens: file names, class names, top functions per file.
        Gives the agent the repo "shape" so it can skip the list_files step
        and go straight to search_symbol or search_code on the right files.
        """
        if not repo_map:
            return ""

        repo    = repo_map.get("repo", "")
        chunks  = repo_map.get("total_chunks", 0)
        n_files = repo_map.get("total_files", 0)
        entry   = repo_map.get("entry_files", [])
        classes = repo_map.get("key_classes", [])
        files   = repo_map.get("files", {})

        lines = [
            f"╔══ REPO MAP: {repo} ({chunks} chunks, {n_files} files) ══╗",
        ]
        if entry:
            lines.append(f"  Entry files : {', '.join(f.split('/')[-1] for f in entry[:5])}")
        if classes:
            lines.append(f"  Key classes : {', '.join(classes[:10])}")
        lines.append("  Key files   :")
        for fp, info in list(files.items())[:8]:
            fname = fp.split("/")[-1]
            parts = []
            if info["classes"]:
                parts.append(f"classes: {', '.join(info['classes'][:3])}")
            if info["functions"]:
                fns = info["functions"][:4]
                extra = f" +{len(info['functions'])-4}" if len(info["functions"]) > 4 else ""
                parts.append(f"fns: {', '.join(fns)}{extra}")
            desc = "  |  ".join(parts) if parts else ""
            lines.append(f"    {fname:<28}  {desc}" if desc else f"    {fname}")
        lines.append("╚══ Skip list_files — use this map to target searches directly ══╝")
        return "\n".join(lines)

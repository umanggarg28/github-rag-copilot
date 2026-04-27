"""
scripts/prebake_repos.py — Generate the canonical artifact set for one
or more repos using the premium tier (Claude Sonnet 4.6).

For each repo the CLI ensures:
  - the repo is ingested with contextual retrieval (force re-index if missing)
  - tour data is generated and persisted to Qdrant
  - architecture and class diagrams are generated and persisted
  - README is generated and persisted
  - the repo_map is built and persisted

All generation calls go through the premium client when ANTHROPIC_API_KEY
is set, so the cached artifacts represent the highest quality this app
can produce. Once cached, every subsequent visitor reads them from Qdrant
without re-running an LLM.

Usage:
  python -m scripts.prebake_repos                                # default Karpathy set
  python -m scripts.prebake_repos owner/repo other/repo          # specific repos
  python -m scripts.prebake_repos --force karpathy/nanoGPT       # rebuild even if cached

Environment:
  ANTHROPIC_API_KEY — required for premium quality. Without it the script
                       runs against the free cascade with a warning.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import settings  # noqa: E402
from backend.services.generation     import GenerationService     # noqa: E402
from backend.services.diagram_service import DiagramService        # noqa: E402
from backend.services.readme_service  import ReadmeService         # noqa: E402
from backend.services.repo_map_service import RepoMapService       # noqa: E402
from backend.services.ingestion_service import IngestionService    # noqa: E402
from ingestion.embedder      import Embedder                        # noqa: E402
from ingestion.qdrant_store  import QdrantStore                     # noqa: E402


# Default set: Karpathy's well-known learning-oriented repos. These are the
# repos most users land on for tutorials / understanding fundamental code.
# Extend or override via CLI args.
DEFAULT_REPOS = [
    "karpathy/autoresearch",
    "karpathy/micrograd",
    "karpathy/nanochat",
    "karpathy/nanoGPT",
]

# Diagram types we cache. "architecture" + "class" cover the two non-tour
# diagram views surfaced in the UI.
DIAGRAM_TYPES = ["architecture", "class"]


def repo_indexed(store: QdrantStore, repo: str) -> bool:
    """Return True if Qdrant has any chunks for this repo."""
    try:
        return store.count(repo=repo) > 0
    except Exception:
        return False


def ingest(repo: str, store: QdrantStore, gen: GenerationService, embedder: Embedder) -> bool:
    """Re-ingest a repo via GitHub with force=True so contextual retrieval
    runs. Even when the repo is already indexed we re-run — premium prebake
    must end with premium-quality contextual descriptions on every chunk,
    not just whatever the previous (possibly free-tier) ingestion left
    behind. The Voyage embeddings are deduplicated by content hash so this
    isn't as expensive as it sounds: only changed/new chunks pay the
    embed cost; only chunks needing fresh contextual retrieval pay the
    LLM cost."""
    already = repo_indexed(store, repo)
    if already:
        print(f"  ▸ re-ingesting {repo} ({store.count(repo=repo)} chunks already indexed)…")
    else:
        print(f"  ▸ ingesting {repo}…")
    ingestion = IngestionService(store=store, embedder=embedder, gen=gen)
    repo_url = f"https://github.com/{repo}"
    try:
        # force=True triggers contextual retrieval enrichment. Because
        # premium_mode is on, gen.generate() routes those calls to the
        # premium client → claude-sonnet-4-6. progress callback prints
        # sparse milestones to stdout for visibility.
        last_step = [""]
        def on_progress(step: str, detail: str) -> None:
            if step != last_step[0]:
                print(f"     · {step}")
                last_step[0] = step
        result = ingestion.ingest(repo_url, force=True, progress=on_progress)
        print(f"  ✓ ingested ({result.get('chunks_stored', '?')} chunks)")
        return True
    except Exception as e:
        print(f"  ✗ ingestion crashed: {e}")
        return False


def bake_tour(repo: str, diagram_svc: DiagramService, force: bool) -> bool:
    """Run the tour pipeline; persist to Qdrant via the service's own cache logic."""
    if not force and diagram_svc._load_tour(repo) is not None:
        print("  ✓ tour cached — skipping (use --force to rebuild)")
        return True
    print("  ▸ tour…")
    last_stage = None
    try:
        for event in diagram_svc.build_tour_stream(repo, force=force):
            stage = event.get("stage")
            if stage and stage != last_stage:
                print(f"     · {stage} ({int((event.get('progress') or 0) * 100)}%)")
                last_stage = stage
            if stage == "error":
                print(f"  ✗ tour failed: {event.get('error')}")
                return False
        print("  ✓ tour cached")
        return True
    except Exception as e:
        print(f"  ✗ tour crashed: {e}")
        return False


def bake_diagram(repo: str, diagram_type: str, diagram_svc: DiagramService, force: bool) -> bool:
    if not force and diagram_svc._load_diagram(repo, diagram_type) is not None:
        print(f"  ✓ {diagram_type} diagram cached — skipping (use --force to rebuild)")
        return True
    print(f"  ▸ {diagram_type} diagram…")
    try:
        for event in diagram_svc.build_diagram_stream(repo, diagram_type, force=force):
            stage = event.get("stage")
            if stage == "error":
                print(f"  ✗ {diagram_type} diagram failed: {event.get('error')}")
                return False
        print(f"  ✓ {diagram_type} diagram cached")
        return True
    except Exception as e:
        print(f"  ✗ {diagram_type} crashed: {e}")
        return False


def bake_readme(repo: str, readme_svc: ReadmeService, store: QdrantStore, force: bool) -> bool:
    if not force and store.load_artifact(repo, "readme"):
        print("  ✓ readme cached — skipping (use --force to rebuild)")
        return True
    print("  ▸ readme…")
    try:
        for event in readme_svc.build_readme_stream(repo, force=force):
            if event.get("stage") == "error":
                print(f"  ✗ readme failed: {event.get('error')}")
                return False
        print("  ✓ readme cached")
        return True
    except Exception as e:
        print(f"  ✗ readme crashed: {e}")
        return False


def bake_repo_map(repo: str, repo_map_svc: RepoMapService, force: bool) -> bool:
    if force:
        repo_map_svc.invalidate(repo)
    print("  ▸ repo_map…")
    try:
        repo_map_svc.get_or_build(repo)
        print("  ✓ repo_map cached")
        return True
    except Exception as e:
        print(f"  ✗ repo_map crashed: {e}")
        return False


def bake_one(
    repo:        str,
    store:       QdrantStore,
    gen:         GenerationService,
    embedder:    Embedder,
    diagram_svc: DiagramService,
    readme_svc:  ReadmeService,
    repo_map_svc: RepoMapService,
    force:       bool,
) -> bool:
    print(f"\n=== {repo} ===")
    started = time.monotonic()
    if not ingest(repo, store, gen, embedder):
        return False
    bake_repo_map(repo, repo_map_svc, force)
    bake_tour(repo, diagram_svc, force)
    for dtype in DIAGRAM_TYPES:
        bake_diagram(repo, dtype, diagram_svc, force)
    bake_readme(repo, readme_svc, store, force)
    elapsed = time.monotonic() - started
    print(f"  ⏱  {elapsed:.1f}s")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-bake artifact cache for canonical repos.")
    parser.add_argument("repos", nargs="*", help="Specific repos to bake (default: Karpathy set).")
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even if already cached.")
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS

    if not settings.anthropic_api_key:
        print("⚠ ANTHROPIC_API_KEY not set — running against the free cascade.")
        print("  Cached artifacts will not represent premium quality.")
    else:
        print(f"Premium tier: enabled (model is configured in GenerationService).")

    store    = QdrantStore()
    embedder = Embedder()
    gen      = GenerationService()
    gen.premium_mode = True   # whole script runs at premium quality

    diagram_svc  = DiagramService(store, gen)
    repo_map_svc = RepoMapService(store)
    readme_svc   = ReadmeService(repo_map_svc, gen, store)

    print(f"\nBaking {len(repos)} repo(s) with premium_mode=True\n")
    ok = 0
    for repo in repos:
        if bake_one(repo, store, gen, embedder, diagram_svc, readme_svc, repo_map_svc, args.force):
            ok += 1

    print(f"\nDone: {ok}/{len(repos)} baked.")
    return 0 if ok == len(repos) else 1


if __name__ == "__main__":
    sys.exit(main())

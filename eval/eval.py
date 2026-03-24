"""
eval.py — Retrieval quality evaluation for the GitHub RAG Copilot.

═══════════════════════════════════════════════════════════════
WHY AN EVAL HARNESS?
═══════════════════════════════════════════════════════════════

Without measurement, you can't improve. The three retrieval modes
(semantic, keyword, hybrid) produce different rankings — but which
is actually better for code questions? This eval harness answers that.

Three metrics:

  Hit Rate @ k  (also called Recall@k)
  ──────────────────────────────────────
  For each test case: did ANY expected file appear in the top-k results?
  Answers: "Does our retrieval find the RIGHT file at all?"
  Example: k=3, expected=["engine.py"], top-3 results include engine.py → hit=1

  Mean Reciprocal Rank  (MRR)
  ──────────────────────────────────────
  For each test case: what rank was the FIRST correct result?
  Score = 1/rank. Rank 1 → 1.0, Rank 2 → 0.5, Rank 3 → 0.33, miss → 0.
  Average across all test cases = MRR.
  Answers: "When we find it, do we find it FIRST?"
  High hit@3 but low MRR means we find it but bury it under noise.

  Precision @ k
  ──────────────────────────────────────
  Of the top-k results, what fraction matched the expected files?
  Answers: "Are our top results relevant, or full of noise?"

═══════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════

  # Run eval on micrograd (must be indexed first):
  python -m eval.eval --repo karpathy/micrograd

  # Compare all three modes:
  python -m eval.eval --repo karpathy/micrograd --modes hybrid semantic keyword

  # Use custom test cases:
  python -m eval.eval --repo owner/repo --cases eval/test_cases/my_cases.json

  # More results per query:
  python -m eval.eval --repo karpathy/micrograd --top-k 5

═══════════════════════════════════════════════════════════════
INTERPRETING RESULTS
═══════════════════════════════════════════════════════════════

  hit@3 > 0.8 = good retrieval
  MRR   > 0.6 = good ranking (top results are relevant)
  MRR   < 0.4 = results are found but buried — re-rank or tune top_k

  If hybrid beats both semantic and keyword on MRR, it confirms that
  RRF fusion is working correctly and worth the extra complexity.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.retrieval import RetrievalService


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """
    One evaluation test case.

    A case is "hit" if any result's filepath contains one of the expected_files
    OR any result's name matches one of expected_names.
    File matching is substring-based — "engine.py" matches "micrograd/engine.py".
    """
    question: str
    expected_files: list[str] = field(default_factory=list)
    expected_names: list[str] = field(default_factory=list)

    def is_hit(self, result: dict) -> bool:
        """Return True if this result satisfies the expected conditions."""
        filepath = result.get("filepath", "").lower()
        name     = result.get("name", "").lower()

        for ef in self.expected_files:
            if ef.lower() in filepath:
                return True
        for en in self.expected_names:
            if en.lower() == name:
                return True
        return False


@dataclass
class CaseResult:
    """Metrics for one test case."""
    question: str
    hit:          bool   # any expected result in top-k
    rank:         int    # rank of FIRST correct result (0 = not found)
    reciprocal_rank: float  # 1/rank or 0.0
    precision_at_k:  float  # fraction of top-k that were relevant
    top_results:  list[dict] = field(default_factory=list)


# ── Core eval logic ────────────────────────────────────────────────────────────

def run_eval(
    retrieval: RetrievalService,
    cases: list[EvalCase],
    repo: Optional[str],
    mode: str,
    top_k: int,
) -> list[CaseResult]:
    """
    Run all test cases against the retrieval service.

    Args:
        retrieval: Initialized RetrievalService
        cases:     List of EvalCase to evaluate
        repo:      Repo filter ('owner/name') or None for all repos
        mode:      'hybrid', 'semantic', or 'keyword'
        top_k:     Number of results to retrieve per case

    Returns:
        List of CaseResult with per-case metrics
    """
    results = []

    for case in cases:
        hits = retrieval.search(
            query=case.question,
            top_k=top_k,
            repo_filter=repo,
            mode=mode,
        )

        first_hit_rank = 0
        hit_count = 0
        for rank, r in enumerate(hits, start=1):
            if case.is_hit(r):
                hit_count += 1
                if first_hit_rank == 0:
                    first_hit_rank = rank

        results.append(CaseResult(
            question        = case.question,
            hit             = first_hit_rank > 0,
            rank            = first_hit_rank,
            reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0,
            precision_at_k  = hit_count / top_k,
            top_results     = hits,
        ))

    return results


def compute_summary(results: list[CaseResult], top_k: int) -> dict:
    """Aggregate per-case metrics into dataset-level scores."""
    n = len(results)
    return {
        f"hit@{top_k}": round(sum(r.hit for r in results) / n, 3),
        "mrr":          round(sum(r.reciprocal_rank for r in results) / n, 3),
        f"p@{top_k}":   round(sum(r.precision_at_k for r in results) / n, 3),
        "n_cases":      n,
    }


# ── Output formatting ──────────────────────────────────────────────────────────

def print_report(
    mode: str,
    summary: dict,
    results: list[CaseResult],
    top_k: int,
    verbose: bool = False,
):
    """Print a human-readable eval report."""
    k = top_k
    hit_key = f"hit@{k}"
    p_key   = f"p@{k}"

    print(f"\n{'─'*60}")
    print(f"  Mode: {mode.upper():<10}  |  {results[0].top_results[0]['repo'] if results and results[0].top_results else 'all repos'}")
    print(f"{'─'*60}")
    print(f"  Hit@{k}  : {summary[hit_key]:.3f}  ({sum(r.hit for r in results)}/{summary['n_cases']} cases hit)")
    print(f"  MRR     : {summary['mrr']:.3f}")
    print(f"  P@{k}    : {summary[p_key]:.3f}")
    print(f"{'─'*60}")

    if verbose:
        for r in results:
            status = "✓" if r.hit else "✗"
            rank_str = f"rank={r.rank}" if r.rank > 0 else "miss"
            print(f"\n  {status} [{rank_str}]  {r.question[:60]}")
            if not r.hit and r.top_results:
                # Show what we got instead
                for i, res in enumerate(r.top_results[:3], 1):
                    print(f"      {i}. {res.get('filepath','')} — {res.get('name','')}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality for an indexed GitHub repo."
    )
    parser.add_argument(
        "--repo", required=True,
        help="Repo slug to evaluate (e.g. karpathy/micrograd). Must be indexed."
    )
    parser.add_argument(
        "--cases", default=None,
        help="Path to JSON test cases file. Defaults to eval/test_cases/<repo-name>.json"
    )
    parser.add_argument(
        "--modes", nargs="+", default=["hybrid", "semantic", "keyword"],
        choices=["hybrid", "semantic", "keyword"],
        help="Retrieval modes to compare (default: all three)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of results to retrieve per query (default: 3)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-case results including misses"
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Write results as JSON to FILE (e.g. eval_results.json). "
             "Useful for CI: git diff eval_results.json shows regressions."
    )
    args = parser.parse_args()

    # ── Load test cases ────────────────────────────────────────────────────────
    if args.cases:
        cases_path = Path(args.cases)
    else:
        repo_name  = args.repo.split("/")[-1]
        cases_path = Path(__file__).parent / "test_cases" / f"{repo_name}.json"

    if not cases_path.exists():
        print(f"Error: test cases file not found: {cases_path}")
        print(f"Create it with format: [{{'question': '...', 'expected_files': ['...']}}]")
        sys.exit(1)

    raw_cases = json.loads(cases_path.read_text())
    cases = [EvalCase(**c) for c in raw_cases]
    print(f"\nLoaded {len(cases)} test cases from {cases_path}")
    print(f"Repo filter: {args.repo}  |  top_k={args.top_k}")

    # ── Initialize retrieval ───────────────────────────────────────────────────
    print("\nInitializing retrieval service (loading embedding model)...")
    t0 = time.time()
    retrieval = RetrievalService()
    print(f"  Ready in {time.time()-t0:.1f}s")

    # ── Run eval for each mode ─────────────────────────────────────────────────
    all_summaries = {}
    for mode in args.modes:
        results = run_eval(
            retrieval=retrieval,
            cases=cases,
            repo=args.repo,
            mode=mode,
            top_k=args.top_k,
        )
        summary = compute_summary(results, args.top_k)
        all_summaries[mode] = summary
        print_report(mode, summary, results, args.top_k, args.verbose)

    # ── JSON output for CI ────────────────────────────────────────────────────
    if args.output:
        import json as _json
        output = {
            "repo":   args.repo,
            "top_k":  args.top_k,
            "n_cases": len(cases),
            "results": {
                mode: {
                    "hit_at_k": s[f"hit@{args.top_k}"],
                    "mrr":      s["mrr"],
                    "p_at_k":   s[f"p@{args.top_k}"],
                }
                for mode, s in all_summaries.items()
            }
        }
        Path(args.output).write_text(_json.dumps(output, indent=2))
        print(f"\nResults written to {args.output}")

    # ── Comparison table ───────────────────────────────────────────────────────
    if len(args.modes) > 1:
        k = args.top_k
        print(f"\n{'═'*60}")
        print(f"  Comparison Summary  (top_k={k}, n={len(cases)} cases)")
        print(f"{'═'*60}")
        print(f"  {'Mode':<10} | {'Hit@'+str(k):<8} | {'MRR':<8} | {'P@'+str(k):<8}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for mode, s in all_summaries.items():
            hit = s[f'hit@{k}']
            mrr = s['mrr']
            p   = s[f'p@{k}']
            best_mrr = max(v['mrr'] for v in all_summaries.values())
            marker = " ◀ best MRR" if mrr == best_mrr else ""
            print(f"  {mode:<10} | {hit:<8.3f} | {mrr:<8.3f} | {p:<8.3f}{marker}")
        print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()

"""
demo_query.py — End-to-end query test (no server needed).

Assumes demo_ingestion.py has already been run (karpathy/micrograd is indexed).

What this tests:
  1. RetrievalService: hybrid search returning ranked code chunks
  2. GenerationService: classify query type, build prompt, call LLM
  3. format_context: numbered source blocks with citations

Run:
  python demo_query.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrieval.retrieval import RetrievalService
from backend.services.generation import GenerationService, classify_query

QUESTIONS = [
    # Technical — low temperature, precise answer expected
    ("how does backward() work in micrograd?", "hybrid"),
    # Creative — higher temperature, more explanatory
    ("explain intuitively what a Value node is", "hybrid"),
    # Keyword test — exact identifier
    ("show me the __add__ method", "keyword"),
]


def main():
    print("=== GitHub RAG Copilot — Query Demo ===\n")

    retrieval = RetrievalService()
    generation = GenerationService()

    for question, mode in QUESTIONS:
        print(f"{'─'*60}")
        q_type = classify_query(question)
        print(f"Q: {question}")
        print(f"   Mode: {mode}  |  Query type: {q_type}")

        results = retrieval.search(
            query=question,
            top_k=4,
            repo_filter="karpathy/micrograd",
            mode=mode,
        )

        print(f"\n  Retrieved {len(results)} chunks:")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['filepath']}:{r['start_line']}–{r['end_line']}  "
                  f"({r['chunk_type']}: {r['name'] or '—'})  score={r['score']}")

        if not results:
            print("  No results — is karpathy/micrograd ingested? Run demo_ingestion.py first.")
            continue

        context = retrieval.format_context(results)
        print(f"\n  Generating answer ({q_type} mode)...")
        answer = generation.answer(question, context, q_type)
        print(f"\nAnswer:\n{answer}\n")

    print("=== Demo complete ===")


if __name__ == "__main__":
    main()

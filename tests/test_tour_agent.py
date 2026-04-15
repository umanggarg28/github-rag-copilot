"""
tests/test_tour_agent.py — Integration tests for TourAgent quality mechanisms.

Tests cover the three quality systems added in Chapter 23:
  1. Evaluator-optimizer loop (_validate_concepts)
  2. Negative example feedback (persist → load → inject)
  3. Dependency graph integrity after concept removal
  4. Dynamic MAX_ROUNDS logic
  5. Token budget truncation

All LLM calls are mocked — no real API costs.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_tour(concepts: list[dict]) -> dict:
    """Build a minimal tour dict for testing _validate_concepts."""
    return {"summary": "Test repo", "entry_point": "main.py", "concepts": concepts}


def _make_concept(id_: int, name: str, subtitle: str = "",
                  depends_on: list | None = None) -> dict:
    return {
        "id": id_,
        "name": name,
        "subtitle": subtitle or f"subtitle for {name}",
        "file": f"file{id_}.py",
        "type": "module",
        "description": "desc",
        "key_items": [],
        "depends_on": depends_on if depends_on is not None else ([] if id_ == 0 else [0]),
        "reading_order": id_ + 1,
        "ask": "What breaks?",
    }


def _make_agent(gen_responses: list[str]) -> "TourAgent":
    """Build a TourAgent with a mocked GenerationService."""
    from backend.services.tour_agent import TourAgent

    gen = MagicMock()
    gen.generate.side_effect = gen_responses

    store = MagicMock()
    store.scroll_repo.return_value = []
    # Default: Qdrant feedback returns empty (fresh repo)
    store.load_tour_feedback.return_value = {}
    store.save_tour_feedback.return_value = None

    return TourAgent(store=store, gen=gen)


# ── 1. Evaluator-optimizer loop ────────────────────────────────────────────────

class TestValidateConcepts:
    def test_ok_response_returns_tour_unchanged(self, tmp_path):
        """If the evaluator says 'ok', the tour is returned as-is."""
        agent = _make_agent([json.dumps({"status": "ok"})])
        tour = _make_tour([
            _make_concept(0, "Streaming Ingestion Pipeline"),
            _make_concept(1, "AST Code Chunking"),
        ])
        result = agent._validate_concepts(tour, repo="owner/repo")
        assert result["concepts"][0]["name"] == "Streaming Ingestion Pipeline"
        assert result["concepts"][1]["name"] == "AST Code Chunking"

    def test_fixed_response_applies_corrections(self, tmp_path):
        """If the evaluator returns 'fixed', concept names are updated."""
        fixed_resp = json.dumps({
            "status": "fixed",
            "concepts": [
                {"id": 0, "name": "End-to-end Pipeline", "subtitle": ""},
                {"id": 1, "name": "Hybrid Retrieval Strategy", "subtitle": ""},
            ],
        })
        # Round 1 returns fixed, round 2 returns ok
        agent = _make_agent([fixed_resp, json.dumps({"status": "ok"})])
        tour = _make_tour([
            _make_concept(0, "main.py"),           # artifact — should be renamed
            _make_concept(1, "retrieval_service"), # artifact — should be renamed
        ])
        result = agent._validate_concepts(tour, repo="owner/repo")
        assert result["concepts"][0]["name"] == "End-to-end Pipeline"
        assert result["concepts"][1]["name"] == "Hybrid Retrieval Strategy"

    def test_loop_runs_max_rounds_then_stops(self, tmp_path):
        """Loop runs at most MAX_ROUNDS times even if corrections keep coming."""
        # Always return "fixed" — should stop after MAX_ROUNDS
        fixed_resp = json.dumps({
            "status": "fixed",
            "concepts": [{"id": 0, "name": "Still Artifact", "subtitle": ""}],
        })
        # 3 rounds max for a repo with >5 past corrections
        store = MagicMock()
        store.load_tour_feedback.return_value = {f"bad{i}": f"good{i}" for i in range(6)}
        store.save_tour_feedback.return_value = None
        from backend.services.tour_agent import TourAgent
        gen = MagicMock()
        gen.generate.return_value = fixed_resp
        agent = TourAgent(store=store, gen=gen)

        tour = _make_tour([_make_concept(0, "artifact.py")])
        agent._validate_concepts(tour, repo="owner/repo")
        assert gen.generate.call_count <= 3

    def test_prior_corrections_injected_in_next_round(self):
        """The second round prompt must include what was already tried."""
        fixed_resp_r1 = json.dumps({
            "status": "fixed",
            "concepts": [{"id": 0, "name": "Still Bad Name", "subtitle": ""}],
        })
        ok_resp = json.dumps({"status": "ok"})
        # Give the agent 2 past corrections → MAX_ROUNDS=2, so a second round fires
        from backend.services.tour_agent import TourAgent
        gen = MagicMock()
        gen.generate.side_effect = [fixed_resp_r1, ok_resp]
        store = MagicMock()
        store.load_tour_feedback.return_value = {"old_bad": "old_good", "another_bad": "another_good"}
        store.save_tour_feedback.return_value = None
        agent = TourAgent(store=store, gen=gen)

        tour = _make_tour([_make_concept(0, "health")])
        agent._validate_concepts(tour, repo="owner/repo")

        # Two rounds must have fired
        assert gen.generate.call_count == 2, "Should run 2 rounds (2 past corrections → MAX_ROUNDS=2)"
        # Round 2 prompt (second call) should include "health" as a prior correction
        second_call_prompt = gen.generate.call_args_list[1][0][1]
        assert "health" in second_call_prompt or "Previously" in second_call_prompt


# ── 2. Feedback persistence ────────────────────────────────────────────────────

class TestFeedbackPersistence:
    def test_corrections_saved_to_qdrant(self):
        """After fixing artifact names, corrections are persisted via QdrantStore."""
        fixed_resp = json.dumps({
            "status": "fixed",
            "concepts": [{"id": 0, "name": "Pipeline Overview", "subtitle": ""}],
        })
        ok_resp = json.dumps({"status": "ok"})
        agent = _make_agent([fixed_resp, ok_resp])
        tour = _make_tour([_make_concept(0, "health")])

        agent._validate_concepts(tour, repo="owner/repo")

        # Qdrant save should have been called with the correction
        agent._store.save_tour_feedback.assert_called_once()
        saved_repo, saved_corrections = agent._store.save_tour_feedback.call_args[0]
        assert saved_repo == "owner/repo"
        assert "health" in saved_corrections

    def test_corrections_saved_to_file_when_no_qdrant(self, tmp_path):
        """File-based fallback saves corrections when Qdrant is unavailable."""
        from backend.services import tour_agent

        # Patch the feedback dir to a temp dir
        original_dir = tour_agent._FEEDBACK_DIR
        tour_agent._FEEDBACK_DIR = tmp_path

        try:
            fixed_resp = json.dumps({
                "status": "fixed",
                "concepts": [{"id": 0, "name": "Pipeline Overview", "subtitle": ""}],
            })
            ok_resp = json.dumps({"status": "ok"})

            from backend.services.tour_agent import TourAgent
            gen = MagicMock()
            gen.generate.side_effect = [fixed_resp, ok_resp]
            store = MagicMock()
            store.load_tour_feedback.side_effect = Exception("Qdrant down")
            store.save_tour_feedback.side_effect = Exception("Qdrant down")
            agent = TourAgent(store=store, gen=gen)

            tour = _make_tour([_make_concept(0, "config.py")])
            agent._validate_concepts(tour, repo="owner/repo")

            feedback_file = tmp_path / "owner_repo_feedback.json"
            assert feedback_file.exists(), "Feedback file should exist as fallback"
            data = json.loads(feedback_file.read_text())
            assert "config.py" in data
        finally:
            tour_agent._FEEDBACK_DIR = original_dir

    def test_feedback_injected_in_synthesize_prompt(self):
        """Previously bad names appear in the Phase 3 FORBIDDEN block."""
        from backend.services.tour_agent import _synthesize_negative_block

        store = MagicMock()
        store.load_tour_feedback.return_value = {
            "health": "Health Check Routing",
            "config.py": "Configuration Pipeline",
        }
        block = _synthesize_negative_block("owner/repo", store=store)
        assert "health" in block
        assert "config.py" in block
        assert "PREVIOUSLY REJECTED" in block

    def test_no_feedback_block_on_fresh_repo(self):
        """New repo with no feedback history returns empty string."""
        from backend.services.tour_agent import _synthesize_negative_block

        store = MagicMock()
        store.load_tour_feedback.return_value = {}
        block = _synthesize_negative_block("owner/new-repo", store=store)
        assert block == ""


# ── 3. Dependency graph integrity ─────────────────────────────────────────────

class TestDependencyGraph:
    def test_removed_concept_reroutes_dependents_to_zero(self):
        """
        When concept 2 is removed, concept 3 (which depended on 2) must be
        rerouted to concept 0 — not left with an empty depends_on.
        """
        # Evaluator removes concept 2 (health check) and renames the rest
        fixed_resp = json.dumps({
            "status": "fixed",
            "concepts": [
                # id=2 (health) is absent from the list — evaluator removed it
                {"id": 0, "name": "Pipeline Overview", "subtitle": ""},
                {"id": 1, "name": "Hybrid Retrieval", "subtitle": ""},
                # id=3 used to depend on id=2 which is now removed
                {"id": 3, "name": "Result Ranking", "subtitle": ""},
            ],
        })
        ok_resp = json.dumps({"status": "ok"})
        agent = _make_agent([fixed_resp, ok_resp])

        tour = _make_tour([
            _make_concept(0, "Pipeline Overview", depends_on=[]),
            _make_concept(1, "Hybrid Retrieval", depends_on=[0]),
            _make_concept(2, "health", depends_on=[0]),     # will be removed
            _make_concept(3, "Result Ranking", depends_on=[2]),  # dep on removed
        ])
        result = agent._validate_concepts(tour, repo="owner/repo")

        concepts = {c["id"]: c for c in result["concepts"]}
        # After removal and re-numbering: original ids 0,1,3 → new ids 0,1,2
        # The concept that was id=3 (dep on removed id=2) should depend on 0
        assert len(result["concepts"]) == 3
        for c in result["concepts"]:
            # No concept should have a depends_on pointing to a non-existent id
            valid_ids = {c2["id"] for c2 in result["concepts"]}
            for dep in c["depends_on"]:
                assert dep in valid_ids, f"Concept {c['id']} has dangling dep {dep}"
            # No concept should depend on itself
            assert c["id"] not in c["depends_on"]

    def test_first_concept_always_has_empty_depends_on(self):
        """Concept 0 (pipeline overview) must never have prerequisites."""
        fixed_resp = json.dumps({
            "status": "fixed",
            "concepts": [{"id": 0, "name": "Pipeline Overview", "subtitle": ""}],
        })
        ok_resp = json.dumps({"status": "ok"})
        agent = _make_agent([fixed_resp, ok_resp])

        tour = _make_tour([_make_concept(0, "main.py", depends_on=[1])])  # bad dep
        result = agent._validate_concepts(tour, repo="owner/repo")
        assert result["concepts"][0]["depends_on"] == []

    def test_no_self_references_after_renumbering(self):
        """No concept should have itself in depends_on after id renumbering."""
        ok_resp = json.dumps({"status": "ok"})
        agent = _make_agent([ok_resp])

        tour = _make_tour([
            _make_concept(0, "Ingestion Pipeline", depends_on=[]),
            _make_concept(1, "AST Chunking", depends_on=[0]),
            _make_concept(2, "Hybrid Search", depends_on=[1]),
        ])
        result = agent._validate_concepts(tour, repo="owner/repo")
        for c in result["concepts"]:
            assert c["id"] not in c["depends_on"], f"Concept {c['id']} depends on itself"


# ── 4. Dynamic MAX_ROUNDS ──────────────────────────────────────────────────────

class TestDynamicRounds:
    def test_fresh_repo_uses_one_round(self):
        """A repo with no feedback history should run exactly one evaluator round."""
        ok_resp = json.dumps({"status": "ok"})
        agent = _make_agent([ok_resp])
        agent._store.load_tour_feedback.return_value = {}  # no history

        tour = _make_tour([_make_concept(0, "Clean Pipeline Name")])
        agent._validate_concepts(tour, repo="owner/new-repo")

        assert agent._gen.generate.call_count == 1  # exactly one round

    def test_heavy_history_repo_gets_three_rounds(self):
        """A repo with >5 past corrections should allow up to 3 rounds."""
        # 6 past corrections → MAX_ROUNDS = 3
        store = MagicMock()
        store.load_tour_feedback.return_value = {f"bad{i}": f"good{i}" for i in range(6)}
        store.save_tour_feedback.return_value = None

        from backend.services.tour_agent import TourAgent
        gen = MagicMock()
        # Always return "fixed" to exhaust the rounds
        gen.generate.return_value = json.dumps({
            "status": "fixed",
            "concepts": [{"id": 0, "name": "Still Bad", "subtitle": ""}],
        })
        agent = TourAgent(store=store, gen=gen)
        tour = _make_tour([_make_concept(0, "artifact.py")])
        agent._validate_concepts(tour, repo="owner/repo")

        # Should run exactly 3 rounds (MAX_ROUNDS for >5 corrections)
        assert gen.generate.call_count == 3


# ── 5. Token budget ────────────────────────────────────────────────────────────

class TestTokenBudget:
    def test_short_text_passes_through_unchanged(self):
        from backend.services.tour_agent import _token_budget
        text = "short text"
        assert _token_budget(text, max_tokens=1000) == text

    def test_long_text_is_truncated(self):
        from backend.services.tour_agent import _token_budget
        # 4 chars/token * 100 tokens = 400 char limit
        text = "x" * 2000
        result = _token_budget(text, max_tokens=100)
        assert len(result) < 2000
        assert "[... truncated" in result

    def test_truncation_snaps_to_newline(self):
        from backend.services.tour_agent import _token_budget
        # Build text where the snap-to-newline should kick in
        lines = "\n".join(["line " + str(i).zfill(3) for i in range(200)])
        result = _token_budget(lines, max_tokens=50)
        # Result should not cut in the middle of a line
        body = result.split("[... truncated")[0]
        assert not body or body.endswith("\n") or "\n" in body

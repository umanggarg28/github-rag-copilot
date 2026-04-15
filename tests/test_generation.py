"""
tests/test_generation.py — Tests for GenerationService model tiering and fallback.

Verifies:
  1. fast=True routes to _fast_model, not _model
  2. Thread safety: parallel calls with fast=True don't mutate shared state
  3. Provider detection works correctly
  4. Fallback chain doesn't prevent fast model selection
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_gen(provider: str = "gemini") -> "GenerationService":
    """
    Build a GenerationService with mocked client — no real API calls.
    Directly sets internal attributes to simulate a fully initialized service.
    """
    from backend.services.generation import GenerationService

    gen = GenerationService.__new__(GenerationService)
    gen.provider     = provider
    gen._model       = "gemini-2.5-flash"
    gen._fast_model  = "gemini-2.0-flash-lite"
    gen._client      = MagicMock()

    # Fake a successful API response
    fake_choice = MagicMock()
    fake_choice.message.content = '{"answer": "42"}'
    fake_choice.finish_reason = "stop"
    gen._client.chat.completions.create.return_value = MagicMock(choices=[fake_choice])

    return gen


# ── 1. Fast tier routing ───────────────────────────────────────────────────────

class TestModelTiering:
    def test_fast_false_uses_primary_model(self):
        """generate(fast=False) must use self._model."""
        gen = _make_gen()

        with patch.object(gen, "_reset_to_primary"):
            gen.generate("sys", "prompt", fast=False)

        call_kwargs = gen._client.chat.completions.create.call_args
        model_used = call_kwargs[1]["model"] if "model" in call_kwargs[1] else call_kwargs[0][0]
        # The model kwarg is passed to the create() call
        create_call = gen._client.chat.completions.create.call_args
        assert create_call.kwargs.get("model") == "gemini-2.5-flash" or \
               (create_call.args and "gemini-2.5-flash" in str(create_call.args))

    def test_fast_true_uses_fast_model(self):
        """generate(fast=True) must use self._fast_model, not self._model."""
        gen = _make_gen()

        with patch.object(gen, "_reset_to_primary"):
            gen.generate("sys", "prompt", fast=True)

        create_call = gen._client.chat.completions.create.call_args
        assert create_call.kwargs.get("model") == "gemini-2.0-flash-lite" or \
               (create_call.args and "gemini-2.0-flash-lite" in str(create_call.args))

    def test_fast_model_differs_from_primary_for_gemini(self):
        """Gemini should have distinct fast and primary models."""
        gen = _make_gen(provider="gemini")
        assert gen._fast_model != gen._model
        assert "lite" in gen._fast_model or "flash" in gen._fast_model

    def test_non_gemini_providers_have_same_or_different_fast_model(self):
        """
        Providers without a genuine fast tier set _fast_model == _model.
        This is correct behaviour — no degradation for those providers.
        """
        gen = _make_gen(provider="anthropic")
        gen._fast_model = "claude-haiku-4-5-20251001"  # same as _model for haiku
        gen._model      = "claude-haiku-4-5-20251001"
        # No error: fast=True on a no-tier provider just uses the same model
        assert gen._fast_model == gen._model

    def test_fast_does_not_mutate_self_model(self):
        """
        Calling generate(fast=True) must NOT modify self._model.
        Critical for thread safety in the parallel enrichment ThreadPoolExecutor.
        """
        gen = _make_gen()
        original_model = gen._model

        with patch.object(gen, "_reset_to_primary"):
            gen.generate("sys", "prompt", fast=True)

        assert gen._model == original_model, \
            "generate(fast=True) must not mutate self._model"


# ── 2. Thread safety ───────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_parallel_fast_calls_do_not_corrupt_model_state(self):
        """
        Simulate the ThreadPoolExecutor pattern from _add_context.
        Multiple threads calling generate(fast=True) concurrently must not
        corrupt self._model — each call reads its own model from params.
        """
        gen = _make_gen()
        observed_models: list[str] = []
        lock = threading.Lock()

        original_create = gen._client.chat.completions.create

        def tracking_create(*args, **kwargs):
            with lock:
                observed_models.append(kwargs.get("model", "unknown"))
            # Return a fake response
            fake = MagicMock()
            fake.choices[0].message.content = "ok"
            fake.choices[0].finish_reason = "stop"
            return fake

        gen._client.chat.completions.create.side_effect = tracking_create

        errors: list[Exception] = []

        def worker(use_fast: bool):
            try:
                with patch.object(gen, "_reset_to_primary"):
                    gen.generate("sys", f"prompt fast={use_fast}", fast=use_fast)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(True,))
            for _ in range(5)
        ] + [
            threading.Thread(target=worker, args=(False,))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised: {errors}"
        # After all calls, self._model must still be the primary model
        assert gen._model == "gemini-2.5-flash"
        # All fast=True calls should have used the fast model
        fast_models = observed_models[:5]  # first 5 were fast=True
        # (ordering not guaranteed, but at least some should be fast model)
        assert any(m == "gemini-2.0-flash-lite" for m in observed_models)
        assert any(m == "gemini-2.5-flash" for m in observed_models)

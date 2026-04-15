"""
tests/test_ingestion.py — Tests for contextual retrieval prompt quality.

Verifies:
  1. _CONTEXT_SYSTEM prompt is terse (no failure scenarios, no verbose language)
  2. fast=True is passed for enrichment calls (model tiering)
  3. Anthropic caching path is taken when provider == 'anthropic'
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestContextualRetrievalPrompt:
    def test_context_system_is_terse(self):
        """
        The _CONTEXT_SYSTEM must instruct the model NOT to write failure scenarios.
        It should not ask for them or describe them as desired output.
        The word "failure" is allowed only in the context of forbidding it.
        """
        from backend.services.ingestion_service import _CONTEXT_SYSTEM

        lower = _CONTEXT_SYSTEM.lower()

        # The prompt must actively forbid failure-scenario language in the OUTPUT
        # (it may use the word "failure" to say "NEVER write failure scenarios")
        assert "failure scenario" in lower or "failure mode" in lower, \
            "_CONTEXT_SYSTEM must mention and forbid failure scenario language"

        # The prompt must FORBID verbose opening phrases in the model's output.
        # It may reference them to say "NEVER write X" — so we check the NEVER
        # clause exists, not that the phrases are absent entirely.
        assert "NEVER" in _CONTEXT_SYSTEM, "_CONTEXT_SYSTEM must use NEVER to forbid bad output"
        never_section = _CONTEXT_SYSTEM[_CONTEXT_SYSTEM.index("NEVER"):]
        assert "This chunk" in never_section or "This code" in never_section, \
            "NEVER clause must explicitly forbid 'This chunk' or 'This code' openings"

        # Must not ask for exhaustive multi-sentence output
        assert "3 sentences" not in lower and "three sentences" not in lower, \
            "_CONTEXT_SYSTEM should ask for 1-2 sentences, not 3+"

    def test_context_system_targets_retrieval(self):
        """The prompt must explicitly mention retrieval/search purpose."""
        from backend.services.ingestion_service import _CONTEXT_SYSTEM

        lower = _CONTEXT_SYSTEM.lower()
        assert any(term in lower for term in ["retrieval", "search", "embedding"]), \
            "_CONTEXT_SYSTEM must mention retrieval/search purpose"

    def test_context_system_asks_for_short_output(self):
        """The prompt must ask for 1-2 sentences, not a paragraph."""
        from backend.services.ingestion_service import _CONTEXT_SYSTEM

        # Should cap at 1-2 sentences
        assert "1-2" in _CONTEXT_SYSTEM or "one" in _CONTEXT_SYSTEM.lower()


class TestFastTierUsedForEnrichment:
    def test_enrich_one_calls_generate_with_fast_true(self):
        """
        _add_context must call gen.generate(... fast=True) for chunk enrichment.
        Using the fast model for enrichment preserves quota for synthesis.
        """
        from backend.services.ingestion_service import _add_context

        # Build a single chunk to enrich
        chunk = {
            "text":       "def forward(x): return x * 2",
            "filepath":   "model.py",
            "name":       "forward",
            "chunk_type": "function",
        }
        # _add_context expects list[dict] with "filepath"/"content" keys
        file_dicts = [{"filepath": "model.py",
                        "content": "def forward(x): return x * 2\n\ndef backward(x): pass\n"}]

        gen = MagicMock()
        gen.provider    = "gemini"
        gen.generate.return_value = "Implements forward pass in model.py"
        gen._fast_model = "gemini-2.0-flash-lite"

        chunks = [chunk]
        result = _add_context(chunks, file_dicts, gen)

        assert gen.generate.called, "gen.generate should have been called"
        call_kwargs = gen.generate.call_args
        # fast=True must appear in kwargs
        assert call_kwargs.kwargs.get("fast") is True or \
               (len(call_kwargs.args) >= 5 and call_kwargs.args[4] is True) or \
               "fast=True" in str(call_kwargs), \
            "Enrichment call must use fast=True to route to fast model tier"


class TestAnthropicCachingPath:
    def test_anthropic_provider_uses_caching_function(self):
        """
        When provider == 'anthropic', _add_context must use _anthropic_contextualise
        instead of gen.generate, to get prompt caching on the document block.
        """
        from backend.services.ingestion_service import _add_context

        chunk = {
            "text":       "class Retriever: pass",
            "filepath":   "retrieval.py",
            "name":       "Retriever",
            "chunk_type": "class",
        }
        file_dicts = [{"filepath": "retrieval.py",
                        "content": "class Retriever:\n    def search(self): pass\n"}]

        gen = MagicMock()
        gen.provider = "anthropic"
        gen._client  = MagicMock()
        gen._model   = "claude-haiku-4-5-20251001"

        # Mock the anthropic response structure
        mock_content = MagicMock()
        mock_content.text = "Retriever class implements vector search in retrieval.py"
        gen._client.messages.create.return_value = MagicMock(content=[mock_content])

        chunks = [chunk]
        result = _add_context(chunks, file_dicts, gen)

        # The Anthropic client's messages.create should be called (caching path)
        assert gen._client.messages.create.called, \
            "Anthropic caching path: _client.messages.create should be called"

        # gen.generate should NOT be called (wrong path for Anthropic)
        assert not gen.generate.called, \
            "gen.generate should not be called when provider is 'anthropic'"

    def test_anthropic_caching_passes_cache_control(self):
        """
        The document block in the Anthropic call must have cache_control set.
        Without it, the same document is re-processed for every chunk in the file.
        """
        from backend.services.ingestion_service import _add_context

        chunk = {
            "text":       "class Retriever: pass",
            "filepath":   "retrieval.py",
            "name":       "Retriever",
            "chunk_type": "class",
        }
        file_dicts = [{"filepath": "retrieval.py",
                        "content": "class Retriever:\n    def search(self): pass\n"}]

        gen = MagicMock()
        gen.provider = "anthropic"
        gen._client  = MagicMock()
        gen._model   = "claude-haiku-4-5-20251001"

        mock_content = MagicMock()
        mock_content.text = "Retriever in retrieval.py"
        gen._client.messages.create.return_value = MagicMock(content=[mock_content])

        _add_context([chunk], file_dicts, gen)

        create_call = gen._client.messages.create.call_args
        messages = create_call.kwargs.get("messages", [])
        assert messages, "messages argument must be provided"

        # Find the document content block and check for cache_control
        found_cache = False
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and "cache_control" in block:
                        found_cache = True
        assert found_cache, \
            "Document block must have cache_control for Anthropic prompt caching"

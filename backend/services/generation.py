"""
generation.py — LLM answer generation for code questions.

Two providers, one interface:
  - Groq (primary):  Fast inference via their API. Free tier is generous.
                     Models: llama-3.3-70b-versatile, mixtral-8x7b-32768
  - Anthropic (fallback): claude-haiku-4-5 — used if GROQ_API_KEY is unset.

Why two providers?
  Groq is fast and free but can hit rate limits. Having Anthropic as a fallback
  means the app keeps working. In production you'd add retry logic, but for a
  learning project provider-switching is simpler and teaches the same concept.

Conditional LLM parameters:
  Code questions are not all alike. "How does backpropagation work?" needs a
  precise, technical answer. "Can you explain this function like I'm ten?" needs
  an approachable, creative answer. We detect the query type and switch:

    Technical: temperature=0.1 (deterministic, accurate)
    Creative:  temperature=0.7 (expressive, natural)

  This is the same principle as a teacher who writes on a whiteboard vs draws
  an analogy — same knowledge, different delivery.

System prompt structure:
  The system prompt tells the LLM three things:
    1. What it is (a code assistant with access to retrieved source)
    2. How to format answers (code blocks, citations by source number)
    3. What NOT to do (hallucinate, answer outside the context)
"""

from pathlib import Path
import sys
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings

# DeepSeek-V3 via OpenRouter free tier — strong coding model, often beats GPT-4 on code tasks.
# The `:free` suffix means OpenRouter serves it at no cost (rate limited but generous).
_OPENROUTER_MODEL = "qwen/qwen3-coder:free"


def _openrouter_client(api_key: str):
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "GitHub RAG Copilot",
        },
    )


# ── LLM parameters per query type ─────────────────────────────────────────────
# Technical queries need precision; creative queries need warmth.
# max_tokens is higher for creative to allow longer explanations/analogies.

_PARAMS = {
    "technical": {"temperature": 0.1, "max_tokens": 1024},
    "creative":  {"temperature": 0.7, "max_tokens": 1536},
}

# ── Query classifier signals ───────────────────────────────────────────────────
# Weighted keyword lists. Higher weight = stronger signal for that type.
# Technical wins ties — better to be precise than wrong by being too creative.

_CREATIVE_SIGNALS: dict[str, int] = {
    "explain":      1,
    "intuition":    2,
    "intuitively":  3,
    "analogy":      3,
    "like i":       2,
    "simply":       2,
    "eli5":         3,
    "in plain":     2,
    "overview":     1,
    "what is":      1,
}

_TECHNICAL_SIGNALS: dict[str, int] = {
    "implement":    2,
    "debug":        2,
    "trace":        2,
    "step by step": 2,
    "formula":      3,
    "algorithm":    2,
    "complexity":   2,
    "optimize":     2,
    "refactor":     2,
    "how does":     1,
    "what does":    1,
    "show me":      1,
}


# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM_BASE = """You are a code assistant with access to retrieved source code snippets.
Answer questions based ONLY on the provided source code context.
If the context doesn't contain enough information, say so — do not hallucinate.

When referencing code, cite the source number like: "According to Source 2 (src/model.py, lines 45–72)..."
Format code in markdown code blocks with the appropriate language tag."""

_SYSTEM_TECHNICAL = _SYSTEM_BASE + """

Be precise and technical. Show exact function signatures, types, and return values when relevant.
Prefer short, accurate answers over long general explanations."""

_SYSTEM_CREATIVE = _SYSTEM_BASE + """

Explain clearly and accessibly. Use analogies where they help.
Build up from simple concepts to complex ones. A diagram described in text is fine."""


def classify_query(question: str) -> str:
    """
    Classify a question as 'technical' or 'creative' using weighted signals.

    Returns 'technical' if technical signals outweigh creative ones, or on tie.
    This biases toward precision — better to be technically correct.

    Example:
      "explain intuitively how attention works" → creative (intuitively=3 > nothing)
      "explain what BLEU score was used" → technical (BLEU=technical word; 'explain' alone
        scores 1 creative vs 0 technical, but if no technical signals present: creative)

    Wait — "explain what BLEU score was used" has 'what' in it which hits "what is" → creative 1.
    "explain" → creative 1. "BLEU" is not in signals. Technical=0. Creative=2 → creative.
    That's actually correct! It IS a creative "explain what" question.
    For "step by step trace of backprop" → technical=4, creative=0 → technical.
    """
    q = question.lower()
    creative_score  = sum(w for s, w in _CREATIVE_SIGNALS.items()  if s in q)
    technical_score = sum(w for s, w in _TECHNICAL_SIGNALS.items() if s in q)
    # Technical wins ties
    return "creative" if creative_score > technical_score else "technical"


class GenerationService:
    """
    Wraps Groq and Anthropic clients. Chooses provider based on available API keys.

    Usage:
      gen = GenerationService()
      answer = gen.answer(question, context, query_type)

      # Or streaming:
      for token in gen.stream(question, context, query_type):
          print(token, end="", flush=True)
    """

    def __init__(self):
        self.provider = self._init_provider()

    def _init_provider(self) -> str:
        """Pick a provider in priority order: Cerebras → OpenRouter → Groq → Gemini → Anthropic.

        Priority is speed + free-tier generosity:
          1. Cerebras (llama-3.3-70b) — 2600 tok/s, 1M tokens/day free, JSON mode supported
          2. OpenRouter (DeepSeek-V3) — best coding quality, free tier
          3. Groq (Llama 3.3 70B)    — fast, generous free tier, good quality
          4. Gemini 2.0 Flash         — Google's fast free model
          5. Anthropic (claude-haiku) — paid fallback

        Cerebras, Gemini, and OpenRouter use OpenAI-compatible endpoints so they share
        the same _groq_complete / _groq_stream code paths.
        """
        if settings.cerebras_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.cerebras_api_key,
                base_url="https://api.cerebras.ai/v1",
            )
            self._model  = "llama3.3-70b"
            print("Generation: using Cerebras (llama-3.3-70b) — 2600 tok/s free tier")
            return "cerebras"
        elif settings.openrouter_api_key:
            self._client = _openrouter_client(settings.openrouter_api_key)
            self._model  = _OPENROUTER_MODEL
            print(f"Generation: using OpenRouter ({self._model})")
            return "openrouter"
        elif settings.groq_api_key:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
            self._model  = "llama-3.3-70b-versatile"
            print(f"Generation: using Groq ({self._model})")
            return "groq"
        elif settings.gemini_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._model  = "gemini-2.0-flash"
            print(f"Generation: using Google Gemini ({self._model})")
            return "gemini"
        elif settings.anthropic_api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._model  = "claude-haiku-4-5-20251001"
            print(f"Generation: using Anthropic ({self._model})")
            return "anthropic"
        else:
            raise ValueError(
                "No LLM API key found. Set CEREBRAS_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in .env"
            )

    def _try_fallback(self) -> bool:
        """
        Switch to the next available provider when the current one is exhausted.

        Called at runtime when a credits/quota error is caught.
        Returns True if a fallback was found and initialized, False if we're already
        on the last provider.

        Priority: cerebras → openrouter (DeepSeek-V3) → groq → gemini → anthropic
        """
        if self.provider == "cerebras" and settings.openrouter_api_key:
            self._client  = _openrouter_client(settings.openrouter_api_key)
            self._model   = _OPENROUTER_MODEL
            self.provider = "openrouter"
            print(f"Generation: Cerebras limit hit — switched to OpenRouter ({_OPENROUTER_MODEL})")
            return True
        if self.provider in ("cerebras", "openrouter") and settings.groq_api_key:
            from groq import Groq
            self._client  = Groq(api_key=settings.groq_api_key)
            self._model   = "llama-3.3-70b-versatile"
            self.provider = "groq"
            print("Generation: OpenRouter limit hit — switched to Groq (llama-3.3-70b-versatile)")
            return True
        if self.provider in ("cerebras", "openrouter", "groq") and settings.gemini_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._model  = "gemini-2.0-flash"
            self.provider = "gemini"
            print("Generation: switched to Google Gemini (gemini-2.0-flash)")
            return True
        if self.provider in ("cerebras", "openrouter", "groq", "gemini") and settings.anthropic_api_key:
            import anthropic
            self._client  = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._model   = "claude-haiku-4-5-20251001"
            self.provider = "anthropic"
            print("Generation: switched to Anthropic (claude-haiku) as final fallback")
            return True
        return False

    def grade_answer(self, question: str, context: str, answer: str, query_type: str = "technical") -> dict:
        """
        Model-based grading: ask the LLM to evaluate the answer against the sources.

        WHY THIS EXISTS
        ───────────────
        RAG answers can silently hallucinate — the LLM sounds confident but makes
        claims not backed by any retrieved source. Without grading, the user has no
        signal about answer reliability beyond manually cross-checking citations.

        Model-based grading turns the LLM into its own fact-checker. A second, short
        call reads (question + sources + answer) and outputs a confidence verdict.
        This is the same technique used in production RAG evaluation pipelines.

        FREE-TIER DESIGN
        ────────────────
        The grading prompt is intentionally tiny:
          - Context is truncated to 2000 chars (enough to spot hallucinations)
          - Output is capped at 80 tokens (just the JSON verdict)
          - Uses the same provider already active — no second model needed
        This adds ~300ms after the last token, which is acceptable.

        Returns:
            {"confidence": "high|medium|low", "faithful": bool, "note": "one sentence"}
            On any failure: {"confidence": "unknown", "faithful": True, "note": ""}
        """
        # Creative queries (high-level summaries, explanations) require synthesis.
        # Grading them on faithfulness penalises correct answers that interpret code.
        # Only grade technical queries where every claim should be traceable to source.
        if query_type == "creative":
            return {"confidence": "unknown", "faithful": True, "note": ""}

        # Extract source headers only (file + function name per source) so the grader
        # sees ALL retrieved sources without blowing up the prompt with full code text.
        # Truncating to 1200 chars caused false "low" grades because sources 3-6 were
        # cut off — the answer would cite them but the grader couldn't see them.
        import re as _re
        source_headers = "\n".join(_re.findall(r"\[Source \d+[^\]]*\]", context))
        if not source_headers:
            source_headers = context[:2000]  # fallback if format unexpected

        _GRADE_SYSTEM = (
            "RAG grader. Does the answer match the sources? Return ONLY JSON."
        )
        _GRADE_PROMPT = (
            f"SOURCES RETRIEVED:\n{source_headers}\n\nANSWER:\n{answer[:1000]}\n\n"
            'Return: {"confidence":"high|medium|low","note":"one sentence"}\n'
            "high=answer claims match retrieved sources, medium=minor extrapolation, low=claims contradict or aren't in sources"
        )
        try:
            raw = self.generate(_GRADE_SYSTEM, _GRADE_PROMPT, temperature=0.0, json_mode=True, max_tokens=80)
            if not raw or not raw.strip():
                return {"confidence": "unknown", "faithful": True, "note": ""}
            import json as _json
            grade = _json.loads(raw.strip())
            # Normalise — guard against unexpected LLM field values
            grade["confidence"] = grade.get("confidence", "unknown").lower()
            if grade["confidence"] not in ("high", "medium", "low"):
                grade["confidence"] = "unknown"
            grade["faithful"] = bool(grade.get("faithful", True))
            grade["note"]     = str(grade.get("note", ""))[:200]
            return grade
        except Exception as e:
            print(f"Grading failed (non-fatal): {e}")
            return {"confidence": "unknown", "faithful": True, "note": ""}

    def _reset_to_primary(self) -> None:
        """Reset provider to OpenRouter/DeepSeek-V3 (primary) so recovered rate limits
        get used again. Called at the start of each public method — if DeepSeek is still
        exhausted, the fallback chain will kick in as normal."""
        if self.provider != "openrouter" and settings.openrouter_api_key:
            self._client  = _openrouter_client(settings.openrouter_api_key)
            self._model   = _OPENROUTER_MODEL
            self.provider = "openrouter"

    def generate(self, system: str, prompt: str, temperature: float = 0.2, json_mode: bool = False, max_tokens: int = 2048) -> str:
        """
        One-shot generation with a custom system prompt — no RAG context, no history.
        Used for structured tasks like diagram/tour generation where we control
        the full prompt ourselves.

        temperature=0.0 for factual/structured output (tour, concept graphs).
        temperature=0.2 for Mermaid diagrams (slight variation helps layout).

        json_mode=True: tells the provider to output valid JSON.
          - OpenAI-compatible (Groq, Gemini, OpenRouter): sets response_format=json_object
            which FORCES the model to output valid JSON — no markdown fences, no preamble.
          - Anthropic: no separate flag needed; the system prompt already says "Return ONLY
            valid JSON". Prompt caching (see _anthropic_complete) keeps this cheap to repeat.
        """
        # Only reset to primary on the first call, not on fallback retries.
        # Without this guard, the recursive retry call resets back to Groq,
        # which 429s again, which retries again — infinite recursion.
        if not getattr(self, '_in_fallback', False):
            self._reset_to_primary()
        messages = [{"role": "user", "content": prompt}]
        params   = {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode}
        try:
            if self.provider in ("cerebras", "groq", "gemini", "openrouter"):
                return self._groq_complete(system, messages, params)
            else:
                return self._anthropic_complete(system, messages, params)
        except Exception as e:
            if _is_exhausted(e) and self._try_fallback():
                self._in_fallback = True
                try:
                    return self.generate(system, prompt, temperature, json_mode)
                finally:
                    self._in_fallback = False
            raise

    def answer(self, question: str, context: str, query_type: str, history: list[dict] | None = None) -> str:
        """
        Generate a complete answer (non-streaming).

        Args:
            question:   The user's question
            context:    Retrieved code chunks formatted as a numbered source list
            query_type: 'technical' or 'creative' — selects system prompt + params
            history:    Prior conversation turns [{role, content}, ...] for follow-up questions
        """
        system   = _SYSTEM_TECHNICAL if query_type == "technical" else _SYSTEM_CREATIVE
        params   = _PARAMS[query_type]
        messages = _build_messages(question, context, history)

        try:
            if self.provider in ("cerebras", "groq", "gemini", "openrouter"):
                return self._groq_complete(system, messages, params)
            else:
                return self._anthropic_complete(system, messages, params)
        except Exception as e:
            if _is_exhausted(e) and self._try_fallback():
                return self.answer(question, context, query_type, history)  # retry with new provider
            raise

    def stream(self, question: str, context: str, query_type: str, history: list[dict] | None = None) -> Iterator[str]:
        """
        Generate a streaming answer, yielding tokens as they arrive.

        Streaming gives the UI token-by-token output — the user sees the answer
        being written in real time rather than waiting for the full response.
        FastAPI uses Server-Sent Events (SSE) to push these tokens to the browser.

        Args:
            question:   The user's question
            context:    Formatted context string from RetrievalService.format_context()
            query_type: 'technical' or 'creative'
            history:    Prior conversation turns [{role, content}, ...] for follow-up questions

        Yields:
            str: Individual text tokens as they are generated
        """
        if not getattr(self, '_in_fallback', False):
            self._reset_to_primary()
        system   = _SYSTEM_TECHNICAL if query_type == "technical" else _SYSTEM_CREATIVE
        params   = _PARAMS[query_type]
        messages = _build_messages(question, context, history)

        try:
            if self.provider in ("cerebras", "groq", "gemini", "openrouter"):
                yield from self._groq_stream(system, messages, params)
            else:
                yield from self._anthropic_stream(system, messages, params)
        except Exception as e:
            if _is_exhausted(e) and self._try_fallback():
                self._in_fallback = True
                try:
                    yield from self.stream(question, context, query_type, history)
                finally:
                    self._in_fallback = False
            else:
                raise

    # ── Groq / Gemini / OpenRouter implementation ─────────────────────────────
    # All three use the OpenAI SDK interface, so one implementation covers all.

    def _groq_complete(self, system: str, messages: list[dict], params: dict) -> str:
        kwargs: dict = dict(
            model=self._model,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        # Structured JSON output: instructs the model to emit ONLY a valid JSON
        # object. No markdown fences, no explanatory text — just the JSON.
        # This is far more reliable than asking nicely in the system prompt.
        if params.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _groq_stream(self, system: str, messages: list[dict], params: dict) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            stream=True,
        )
        yielded = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yielded = True
                yield delta
        if not yielded:
            # Model returned 200 OK but streamed no content (e.g. stepfun free tier).
            # Raise so the caller's exception handler can try the next provider.
            raise RuntimeError("Stream completed with no content from model")

    # ── Anthropic implementation ───────────────────────────────────────────────

    def _anthropic_complete(self, system: str, messages: list[dict], params: dict) -> str:
        # Prompt caching: wrapping the system prompt in a cache_control block
        # tells Anthropic to cache the compiled KV state for this text.
        # On the SECOND call with the same system prompt, Anthropic returns
        # the cached result instantly — no re-processing, ~90% cost reduction
        # for the cached tokens.
        #
        # Rules for caching to activate:
        #   - Content must be ≥ 1024 tokens (for claude-3-5+)
        #   - Same content must appear in the same position across calls
        #   - Cache TTL is 5 minutes (ephemeral) — enough for multi-turn chat
        #
        # Our system prompts are short (~50 tokens), so they won't cache on their
        # own. But for large code-context calls (diagram generation), the user
        # message is large and we mark the last user turn cacheable there too.
        # The real benefit kicks in when the same large prompt is reused across
        # multiple diagram types for the same repo.
        system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        response = self._client.messages.create(
            model=self._model,
            system=system_block,
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return response.content[0].text

    def _anthropic_stream(self, system: str, messages: list[dict], params: dict) -> Iterator[str]:
        system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        with self._client.messages.stream(
            model=self._model,
            system=system_block,
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        ) as stream:
            for text in stream.text_stream:
                yield text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_exhausted(e: Exception) -> bool:
    """Return True if this error means the provider is unavailable and we should try another.

    Includes 429 (rate limit) because each provider's rate limit is independent —
    if Gemini is rate-limited, OpenRouter likely isn't. The fallback chain is designed
    to only go to safe providers (Groq excluded from agent due to hermes bug), so
    triggering fallback on 429 is safe and gives the user an immediate response
    instead of "please wait a moment".
    """
    msg = str(e).lower()
    return any(kw in msg for kw in (
        "credit", "billing", "quota", "rate_limit", "rate limit",
        "resource_exhausted", "daily limit", "429", "no content from model",
    ))


def _build_messages(question: str, context: str, history: list[dict] | None = None) -> list[dict]:
    """
    Build the LLM message list: prior conversation turns + current question with code context.

    History turns are bare question/answer pairs — we don't re-attach the retrieved
    code context for them (that context is no longer relevant). Only the current
    question gets the freshly retrieved context injected.

    Context comes first in the current turn because:
      - LLMs attend better to information near the start of the prompt
      - The question at the end is a clear instruction after reading the context
    """
    messages = [{"role": h["role"], "content": h["content"]} for h in (history or [])]
    current_prompt = f"""Here is the relevant source code retrieved from the repository:

{context}

---

Question: {question}

Answer based on the source code above. Cite sources by number (e.g. "Source 1")."""
    messages.append({"role": "user", "content": current_prompt})
    return messages

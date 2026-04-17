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
import threading
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings

# Thread-local storage for the "quality-only" flag.
# When generate_quality() or generate_synthesis() is active on a thread,
# this flag tells _try_fallback() to skip Cerebras 8B for THAT thread only.
# Using threading.local() instead of a shared counter eliminates a race
# condition where concurrent Phase 2 workers could decrement each other's
# counter to zero mid-cascade, accidentally unblocking Cerebras.
_quality_local = threading.local()

# Qwen3-Coder via OpenRouter free tier — strong coding model from Alibaba (Apache 2.0 licence).
# The `:free` suffix means OpenRouter serves it at no cost (rate limited but generous).
_OPENROUTER_MODEL = "qwen/qwen3-coder:free"


def _openrouter_client(api_key: str):
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=45, max_retries=0,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Cartographer",
        },
    )


# ── Thought-token stripping ────────────────────────────────────────────────────
# Gemma 4 (and some other reasoning models) wrap their internal chain-of-thought
# in <thought>...</thought> tags before the final answer. These tags are useful
# for the model's reasoning but should NOT be shown to the user — they're noisy
# and often incomplete mid-stream.
#
# We strip them with a simple state machine over the token stream:
#   - "outside": normal text, yield tokens as-is
#   - "inside":  inside a <thought> block, buffer but don't yield
#
# Edge case: the opening/closing tag may be split across multiple tokens
# (e.g. "<" then "thought>"). We keep a small look-ahead buffer to handle this.

def _strip_thought_tokens(tokens: Iterator[str]) -> Iterator[str]:
    """Strip <thought>...</thought> blocks from a token stream.

    Gemma 4 and some other reasoning models wrap their chain-of-thought in
    <thought>...</thought> before (or instead of) the final answer. We strip
    the tags but keep the answer. If the entire response is inside a thought
    block with nothing after it (Gemma 4 sometimes does this when the answer
    IS the reasoning), we fall back to yielding the thought content so the
    user always gets a response.

    State machine:
      outside  → yield tokens, detect <thought> opening tag
      inside   → buffer and discard, detect </thought> closing tag
      fallback → at EOF, if no answer tokens were ever yielded, yield thought buf
    """
    OPEN  = "<thought>"
    CLOSE = "</thought>"

    buf           = ""   # look-ahead window for partial tag detection
    inside        = False
    thought_buf   = ""   # accumulates full thought content (for fallback)
    answer_yielded = False

    for tok in tokens:
        buf += tok

        while True:
            if inside:
                end = buf.find(CLOSE)
                if end != -1:
                    # Found closing tag — everything before it is thought content
                    thought_buf += buf[:end]
                    buf = buf[end + len(CLOSE):]
                    inside = False
                else:
                    # Discard confirmed-thought text, keep tail for partial-tag detection
                    keep = len(CLOSE) - 1
                    if len(buf) > keep:
                        thought_buf += buf[:-keep]
                        buf = buf[-keep:]
                    break
            else:
                start = buf.find(OPEN)
                if start != -1:
                    # Yield everything before the opening tag as answer text
                    if start > 0:
                        pre = buf[:start]
                        if pre.strip():
                            answer_yielded = True
                        yield pre
                    buf = buf[start + len(OPEN):]
                    inside = True
                else:
                    # No opening tag — safe to yield all but the partial-tag tail
                    keep = len(OPEN) - 1
                    if len(buf) > keep:
                        chunk = buf[:-keep]
                        if chunk.strip():
                            answer_yielded = True
                        yield chunk
                        buf = buf[-keep:]
                    break

    # ── End of stream ──────────────────────────────────────────────────────────
    if inside:
        # Stream ended mid-thought (no closing tag found).
        # Yield the accumulated thought content — it IS the answer.
        yield thought_buf + buf
        return  # skip the fallback check — we just yielded

    if buf:
        if buf.strip():
            answer_yielded = True
        yield buf

    # Fallback: model wrapped everything in <thought>...</thought> with nothing
    # substantive after (either empty or only whitespace). Rather than leaving
    # the answer blank, yield the thought content so the user always gets output.
    if not answer_yielded and thought_buf:
        yield thought_buf


# ── LLM parameters per query type ─────────────────────────────────────────────
# Technical queries need precision; creative queries need warmth.
# max_tokens is higher for creative to allow longer explanations/analogies.

_PARAMS = {
    "technical": {"temperature": 0.1, "max_tokens": 4096},
    "creative":  {"temperature": 0.7, "max_tokens": 4096},
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

_SYSTEM_BASE = """You are an expert code assistant answering questions about a specific codebase. You have access to retrieved source code snippets and answer based strictly on what those snippets contain.

NEVER hallucinate function names, file paths, or behaviours not present in the provided sources.
NEVER say a feature "likely" or "probably" works a certain way — either the source shows it or say it is not in the retrieved context.
NEVER write generic programming advice that ignores the actual code provided.

If the retrieved context does not contain enough information to answer, say so clearly rather than filling gaps with assumptions.

Cite every claim: "According to Source 2 (src/model.py, lines 45–72)..."
Format all code in fenced code blocks with the appropriate language tag."""

_SYSTEM_TECHNICAL = _SYSTEM_BASE + """

Be precise and technical. Show exact function signatures, types, and return values when relevant.
Prefer short, accurate answers. NEVER write multi-paragraph explanations for single-sentence questions."""

_SYSTEM_CREATIVE = _SYSTEM_BASE + """

Explain clearly and accessibly. Use analogies where they genuinely help understanding.
Build from simple to complex. A diagram described in text is fine.
NEVER sacrifice accuracy for approachability — analogies must match what the code actually does."""


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
    Wraps multiple LLM providers. Chooses the best available free provider on init,
    then falls back through the chain if rate limits or quota errors are hit.

    Priority: Gemini (gemma-4-31b-it) → Cerebras (llama-3.3-70b) → Anthropic (haiku)
              → OpenRouter (qwen3-coder) → Groq (llama-3.3-70b-versatile)

    Free providers (Gemini, Cerebras, OpenRouter, Groq) come first.
    Anthropic is kept as a paid fallback — only used if all free providers fail.

    One shared instance handles all tasks: Q&A streaming, diagram generation,
    README generation, and contextual enrichment. Gemini gemma-4-31b-it is the
    strongest free model available and handles all task types well.

    Usage:
      gen = GenerationService()
      answer = gen.answer(question, context, query_type)
      for token in gen.stream(question, context, query_type):
          print(token, end="", flush=True)
    """

    def __init__(self):
        self.provider = self._init_provider()

    def _init_provider(self) -> str:
        """Pick the best available provider.

        Priority: Gemini → Cerebras → Anthropic haiku → OpenRouter → Groq
        """
        _TIMEOUT = 30  # seconds — prevents any provider from hanging indefinitely
        if settings.gemini_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=_TIMEOUT, max_retries=0,
            )
            self._model      = "gemini-2.5-flash"
            # gemini-2.0-flash-lite: ~4x faster and lighter than 2.5-flash.
            # Used for high-volume enrichment calls (one per chunk at ingest time)
            # so the stronger model's quota is preserved for synthesis tasks.
            self._fast_model = "gemini-2.0-flash-lite"
            print("Generation: using Gemini 2.5 Flash via Google Gemini API")
            return "gemini"
        elif settings.sambanova_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.sambanova_api_key,
                base_url="https://api.sambanova.ai/v1",
                timeout=_TIMEOUT, max_retries=0,
            )
            # DeepSeek-V3.1 — best open-source model on SambaNova free tier (200K tok/day).
            # Meta-Llama-3.1-405B-Instruct was deprecated Apr 2026; DeepSeek-V3.1 is the
            # replacement and outperforms it on coding + reasoning benchmarks.
            self._model      = "DeepSeek-V3.1"
            self._fast_model = self._model   # no lighter SambaNova tier available
            print("Generation: using SambaNova (DeepSeek-V3.1) — free tier")
            return "sambanova"
        elif settings.cerebras_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.cerebras_api_key,
                base_url="https://api.cerebras.ai/v1",
                timeout=_TIMEOUT, max_retries=0,
            )
            # llama3.1-8b: only confirmed-available model on Cerebras free tier (Apr 2026).
            # llama3.3-70b and gpt-oss-120b both return 404 on the free tier.
            self._model      = "llama3.1-8b"
            self._fast_model = "llama3.1-8b"
            print("Generation: using Cerebras (llama3.1-8b) — free tier")
            return "cerebras"
        elif settings.anthropic_api_key:
            import anthropic
            self._client     = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0)
            self._model      = "claude-haiku-4-5-20251001"
            self._fast_model = self._model   # haiku is already the fast Anthropic tier
            print(f"Generation: using Anthropic ({self._model})")
            return "anthropic"
        elif settings.openrouter_api_key:
            self._client     = _openrouter_client(settings.openrouter_api_key)
            self._model      = _OPENROUTER_MODEL
            self._fast_model = self._model
            print(f"Generation: using OpenRouter ({self._model})")
            return "openrouter"
        elif settings.mistral_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.mistral_api_key,
                base_url="https://api.mistral.ai/v1",
                timeout=_TIMEOUT, max_retries=0,
            )
            # mistral-small-latest: free tier, 1B tok/month, strong at structured output.
            self._model      = "mistral-small-latest"
            self._fast_model = self._model
            print("Generation: using Mistral (mistral-small-latest) — free tier")
            return "mistral"
        elif settings.groq_api_key:
            from groq import Groq
            self._client     = Groq(api_key=settings.groq_api_key)
            self._model      = "llama-3.3-70b-versatile"
            self._fast_model = self._model
            print(f"Generation: using Groq ({self._model})")
            return "groq"
        else:
            raise ValueError(
                "No LLM API key found. Set GEMINI_API_KEY, CEREBRAS_API_KEY, SAMBANOVA_API_KEY, "
                "MISTRAL_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY in .env"
            )

    def _try_fallback(self) -> bool:
        """
        Switch to the next available provider when the current one is exhausted.

        Called at runtime when a credits/quota error is caught.
        Returns True if a fallback was found and initialized, False if we're already
        on the last provider.

        Records a 60-second exhaustion window for the failed provider so that
        _reset_to_primary() doesn't immediately reset parallel calls back to it.
        Without this, 12 parallel ReAct calls all 429 on Gemini simultaneously
        and cascade all the way to Groq, exhausting its daily token budget.

        Each candidate provider is checked against _exhausted_until before being
        selected. This allows generate_synthesis() to block specific providers
        (e.g. Cerebras 8B) for a single high-stakes call by writing a far-future
        timestamp into the dict — without needing a separate code path.
        """
        import time
        # Mark the current provider as exhausted for 60 seconds
        if not hasattr(self, '_exhausted_until'):
            self._exhausted_until = {}
        self._exhausted_until[self.provider] = time.monotonic() + 60
        failed = self.provider
        print(f"Generation: {failed} rate-limited — trying next provider")

        # Cascade order — Google tier first (same key), then SambaNova, then others.
        # Gemma 4 is placed before SambaNova because it shares the Gemini API key:
        # when Gemini 2.5 Flash exhausts, Gemma 4 31B is a free same-key extension.
        # Only when both Google models are exhausted do we spend SambaNova quota.
        # _all must match this order exactly — the _all[:_all.index(X)] pattern
        # means "allow any provider that comes before X in _all" to fall through to X.
        _all = ("gemini", "gemma4", "sambanova", "cerebras", "anthropic", "openrouter", "mistral", "groq")

        # Helper: is a provider currently blocked by an exhaustion window?
        # This is what allows generate_synthesis() to block Cerebras (or any provider)
        # for a single call by writing to _exhausted_until — without this check,
        # _try_fallback() blindly switches to it regardless.
        now = time.monotonic()
        def _is_blocked(name: str) -> bool:
            return self._exhausted_until.get(name, 0) > now

        # Gemma 4 31B — same GEMINI_API_KEY, separate rate-limit bucket from Gemini 2.5 Flash.
        # Skipped when _skip_thinking=True: Gemma 4 is a reasoning model that prepends a THINK
        # block to every response — this eats the entire token budget on compact-JSON tasks.
        if (self.provider in _all[:_all.index("gemma4")]
                and settings.gemini_api_key
                and not _is_blocked("gemma4")
                and not getattr(self, '_skip_thinking', False)):
            from openai import OpenAI
            self._client  = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=90, max_retries=0,  # Gemma 4 31B is slow; no SDK retries
            )
            self._model      = "gemma-4-31b-it"
            self._fast_model = "gemma-4-31b-it"
            self.provider    = "gemma4"
            print("Generation: switched to Gemma 4 31B (same Gemini key)")
            return True
        if getattr(self, '_skip_thinking', False) and self.provider in _all[:_all.index("gemma4")] and settings.gemini_api_key:
            print("Generation: skipping Gemma 4 (thinking model) — falling through to SambaNova")
        # SambaNova DeepSeek-V3.1 — excellent at code, best quality after Google tier (200K tok/day)
        if self.provider in _all[:_all.index("sambanova")] and settings.sambanova_api_key and not _is_blocked("sambanova"):
            from openai import OpenAI
            self._client  = OpenAI(api_key=settings.sambanova_api_key, base_url="https://api.sambanova.ai/v1", timeout=30, max_retries=0)
            self._model   = "DeepSeek-V3.1"
            self.provider = "sambanova"
            print("Generation: switched to SambaNova (DeepSeek-V3.1)")
            return True
        # Cerebras llama3.1-8b is blocked during generate_quality() and generate_synthesis() calls.
        # _quality_local.quality_only is a per-thread boolean set by those methods.
        # Using threading.local() instead of a shared counter eliminates a race condition
        # where Worker B finishing its quality call would decrement a shared counter to 0
        # while Worker A is still mid-cascade, accidentally unblocking Cerebras for A.
        _in_synthesis = getattr(_quality_local, 'quality_only', False)
        if self.provider in _all[:_all.index("cerebras")] and settings.cerebras_api_key and not _is_blocked("cerebras") and not _in_synthesis:
            from openai import OpenAI
            self._client  = OpenAI(api_key=settings.cerebras_api_key, base_url="https://api.cerebras.ai/v1", timeout=30, max_retries=0)
            # llama3.1-8b: only confirmed-available model on Cerebras free tier as of Apr 2026
            # (llama3.3-70b and gpt-oss-120b both return 404 on free tier)
            self._model   = "llama3.1-8b"
            self.provider = "cerebras"
            print("Generation: switched to Cerebras (llama3.1-8b)")
            return True
        if (_is_blocked("cerebras") or _in_synthesis) and self.provider in _all[:_all.index("cerebras")]:
            print("Generation: skipping Cerebras (blocked for synthesis) — falling through")
        if self.provider in _all[:_all.index("anthropic")] and settings.anthropic_api_key and not _is_blocked("anthropic"):
            import anthropic
            self._client  = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0)
            self._model   = "claude-haiku-4-5-20251001"
            self.provider = "anthropic"
            print("Generation: switched to Anthropic (claude-haiku-4-5)")
            return True
        if self.provider in _all[:_all.index("openrouter")] and settings.openrouter_api_key and not _is_blocked("openrouter"):
            self._client  = _openrouter_client(settings.openrouter_api_key)
            self._model   = _OPENROUTER_MODEL
            self.provider = "openrouter"
            print(f"Generation: switched to OpenRouter ({_OPENROUTER_MODEL})")
            return True
        if self.provider in _all[:_all.index("mistral")] and settings.mistral_api_key and not _is_blocked("mistral"):
            from openai import OpenAI
            self._client  = OpenAI(api_key=settings.mistral_api_key, base_url="https://api.mistral.ai/v1", timeout=30, max_retries=0)
            self._model   = "mistral-small-latest"
            self.provider = "mistral"
            print("Generation: switched to Mistral (mistral-small-latest)")
            return True
        if self.provider in _all[:_all.index("groq")] and settings.groq_api_key and not _is_blocked("groq"):
            from groq import Groq
            self._client  = Groq(api_key=settings.groq_api_key)
            self._model   = "llama-3.3-70b-versatile"
            self.provider = "groq"
            print("Generation: switched to Groq (llama-3.3-70b-versatile) — last resort")
            return True
        print(f"Generation: all providers exhausted — no fallback available")
        return False

    # Providers whose architecture emits a THINK/reasoning block before the answer.
    # These are incompatible with compact-JSON tasks (forced DONE, short synthesis)
    # because the thinking block consumes the entire token budget before the JSON starts.
    _THINKING_PROVIDERS = frozenset({"gemma4"})

    def generate_quality(self, system: str, prompt: str, **kwargs) -> str:
        """
        Quality-gated generation: blocks Cerebras llama3.1-8b and waits for strong
        providers to recover rather than falling through to a weak model.

        WHY THIS EXISTS
        ───────────────
        Phase 2 investigations ask the model to summarise tool call results (read_file,
        grep, search_symbol) into a coherent concept description. An 8B model (Cerebras)
        does not faithfully report what the tools returned — it hallucinates file paths,
        function names, and behaviours not present in the code. The tour then contains
        fabricated content that looks plausible but is completely wrong.

        The right response when only a weak model is available is to wait for a strong
        model to recover, not to produce fast but hallucinated output. A slow correct
        tour is better than a fast wrong one.

        BEHAVIOUR
        ─────────
        1. Waits for Gemini's rate-limit window to expire if it is currently blocked
           (same logic as generate_synthesis).
        2. Clears Gemini/SambaNova exhaustion windows so they are retried fresh.
        3. Sets _quality_local.quality_only=True (per-thread) so _try_fallback() skips Cerebras 8B.
        4. Skips thinking models (Gemma 4) since the THINK block wastes budget.
        5. If all quality providers are exhausted (Gemini, SambaNova, Anthropic,
           OpenRouter, Mistral, Groq), raises RuntimeError rather than using Cerebras.
           The caller should treat this as a skipped investigation, not a crash.
        """
        import time
        if not hasattr(self, '_exhausted_until'):
            self._exhausted_until = {}

        # Wait for Gemini to recover if its window is still active.
        gemini_wait = self._exhausted_until.get('gemini', 0) - time.monotonic()
        if gemini_wait > 0:
            wait = min(gemini_wait + 2, 65)
            print(f"Generation: quality call waiting {wait:.0f}s for Gemini to recover…")
            time.sleep(wait)

        # Clear windows so the cascade retries strong providers.
        self._exhausted_until.pop('gemini', None)
        self._exhausted_until.pop('gemma4', None)
        self._exhausted_until.pop('sambanova', None)

        # Block Cerebras for this thread via the thread-local flag.
        # Using _quality_local (threading.local) instead of a shared counter ensures
        # that each concurrent Phase 2 worker blocks Cerebras independently.
        # A shared counter races: Worker B finishing its call decrements to 0 while
        # Worker A is mid-cascade, accidentally unblocking Cerebras for Worker A.
        old_quality = getattr(_quality_local, 'quality_only', False)
        _quality_local.quality_only = True
        self._skip_thinking = True
        try:
            return self.generate(system, prompt, **kwargs)
        finally:
            self._skip_thinking = False
            _quality_local.quality_only = old_quality

    def generate_non_thinking(self, system: str, prompt: str, **kwargs) -> str:
        """
        Like generate() but skips thinking models (e.g. Gemma 4) in the entire fallback chain.

        Gemma 4 unconditionally prepends a THINK block to every response. This is architectural
        — prompt instructions cannot suppress it. On compact-JSON tasks (forced DONE, synthesis)
        the THINK block consumes the entire token budget before the JSON even starts.

        This sets _skip_thinking=True for the duration of the call. _try_fallback() checks
        this flag and bypasses Gemma 4, falling directly to SambaNova (or the next available
        non-thinking provider). The flag is cleared in a finally block so it never leaks
        into subsequent calls.
        """
        self._skip_thinking = True
        try:
            return self.generate(system, prompt, **kwargs)
        finally:
            self._skip_thinking = False

    def generate_synthesis(self, system: str, prompt: str, **kwargs) -> str:
        """
        Like generate_non_thinking() but engineered for Phase 3 synthesis quality.

        WHY THIS IS NEEDED
        ──────────────────
        Phase 1 MAP runs up to 16 ReAct rounds. Phase 2 INVESTIGATE runs up to 11
        parallel investigations with 3 workers, each up to 4 rounds. Together they
        exhaust Gemini's 15 RPM free-tier quota, triggering a 60-second exhaustion
        window in _try_fallback(). If Phase 3 synthesis runs while that window is
        still active, the call cascades all the way to Cerebras llama3.1-8b — an
        8B model that cannot reliably produce 3000-token structured JSON.

        THREE INTERVENTIONS
        ───────────────────
        1. Wait for Gemini/SambaNova to recover: simply clearing the exhaustion window
           doesn't help — Gemini still 429s on the next call if it hasn't recovered.
           We sleep until the window expires (max 65s) so the strongest model is
           genuinely available when synthesis starts.

        2. Block Cerebras for this call only: llama3.1-8b cannot reliably write 3000-
           token JSON. We mark it exhausted for 3600s and restore it in finally so
           Phase 1/2 tool calls can still use it. _try_fallback() now checks
           _exhausted_until for ALL providers (not just Gemini) so the block takes
           effect in the cascade.

        3. Skip thinking models: Gemma 4's THINK block eats the token budget before
           the JSON even starts. _skip_thinking=True causes _try_fallback() to jump
           over Gemma 4 to SambaNova.

        Acceptable synthesis providers in priority order:
          Gemini 2.5 Flash → SambaNova DeepSeek-V3.1 → Anthropic Haiku →
          OpenRouter Qwen3-Coder → Mistral Small → Groq llama-3.3-70b
          (Cerebras llama3.1-8b: blocked; Gemma 4: skipped — thinking model)
        """
        import time
        if not hasattr(self, '_exhausted_until'):
            self._exhausted_until = {}

        # 1. Wait for Gemini to recover if it's in a short exhaustion window.
        #    Phase 1/2 can exhaust Gemini's 15 RPM free tier across parallel calls.
        #    Synthesis is ONE sequential call — waiting up to 65s here is acceptable
        #    and ensures we get the strongest model rather than silently falling to 8B.
        #    We don't simply clear the window: clearing it and immediately retrying
        #    still 429s if Gemini hasn't actually recovered yet, which sends the call
        #    down the entire cascade again.
        gemini_blocked_until = self._exhausted_until.get('gemini', 0)
        wait_secs = gemini_blocked_until - time.monotonic()
        if wait_secs > 0:
            wait_secs = min(wait_secs + 2, 65)  # 2s buffer; cap at 65s
            print(f"Generation: synthesis waiting {wait_secs:.0f}s for Gemini to recover…")
            time.sleep(wait_secs)

        # Similarly wait for SambaNova if it's our only strong fallback after Gemini.
        # SambaNova DeepSeek-V3.1 is the best option after Gemini; waiting for it is
        # preferable to falling to an 8B model.
        # After the Gemini wait (if any), check SambaNova — only wait if Gemini is
        # now clear (its timestamp is in the past) so we don't double-wait.
        gemini_still_blocked = self._exhausted_until.get('gemini', 0) > time.monotonic()
        sambanova_blocked_until = self._exhausted_until.get('sambanova', 0)
        wait_secs = sambanova_blocked_until - time.monotonic()
        if wait_secs > 0 and not gemini_still_blocked:
            wait_secs = min(wait_secs + 2, 65)
            print(f"Generation: synthesis waiting {wait_secs:.0f}s for SambaNova to recover…")
            time.sleep(wait_secs)

        # 2. Clear Google exhaustion windows (we either waited them out or they expired).
        self._exhausted_until.pop('gemini', None)
        self._exhausted_until.pop('gemma4', None)
        self._exhausted_until.pop('sambanova', None)

        # 3. Block Cerebras for this thread using threading.local().
        #    WHY threading.local() instead of a shared counter: Phase 3 synthesis is
        #    sequential, but generate_quality() calls from Phase 2 workers also set this
        #    flag concurrently. A shared counter would race: Worker B finishing its call
        #    decrements to 0 while Worker A is mid-cascade, unblocking Cerebras for A.
        #    threading.local() is per-thread — each thread's flag is independent.
        old_quality = getattr(_quality_local, 'quality_only', False)
        _quality_local.quality_only = True

        # 4. Skip thinking models — Gemma 4's THINK block eats the 3000-token budget.
        self._skip_thinking = True
        try:
            # 5. Retry loop: if all providers are exhausted (Phase 2 burned through all
            #    60s windows right before synthesis runs), wait one full window cycle and
            #    retry. Synthesis is the ONE place where a 67s wait is acceptable — it's
            #    a single sequential call and "all providers exhausted" is recoverable.
            for attempt in range(3):
                try:
                    return self.generate(system, prompt, **kwargs)
                except RuntimeError as e:
                    if "rate-limited" not in str(e).lower() or attempt >= 2:
                        raise
                    wait = 67  # 60s window + 7s buffer — enough for ALL providers to clear
                    print(f"Generation: synthesis all-exhausted — waiting {wait}s for providers to recover (attempt {attempt + 1}/3)…")
                    time.sleep(wait)
                    # Clear every exhaustion window; they've all expired by now.
                    self._exhausted_until.clear()
        finally:
            self._skip_thinking = False
            _quality_local.quality_only = old_quality

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
        """Reset to Gemini (primary) so recovered rate limits get used again.

        Called at the start of each public method — if Gemini is still in its
        60-second exhaustion window (set by _try_fallback after a 429), we skip
        the reset so parallel calls don't all hammer Gemini simultaneously.

        Without the exhaustion window, 12 parallel ReAct calls all reset to Gemini,
        all 429, all cascade to Groq — exhausting its daily token limit.
        """
        import time
        now = time.monotonic()
        exhausted_until = getattr(self, '_exhausted_until', {})
        if exhausted_until.get("gemini", 0) > now:
            # Gemini still rate-limited — stay on current fallback provider
            return
        if self.provider not in ("gemini", "gemma4") and settings.gemini_api_key:
            from openai import OpenAI
            self._client  = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=30, max_retries=0,
            )
            self._model   = "gemini-2.5-flash"
            self.provider = "gemini"
            print(f"Generation: reset to Gemini (primary)")

    def generate(self, system: str, prompt: str, temperature: float = 0.2,
                 json_mode: bool = False, max_tokens: int = 2048,
                 fast: bool = False) -> str:
        """
        One-shot generation with a custom system prompt — no RAG context, no history.
        Used for structured tasks like diagram/tour generation where we control
        the full prompt ourselves.

        temperature=0.0 for factual/structured output (tour, concept graphs).
        temperature=0.2 for Mermaid diagrams (slight variation helps layout).

        fast=True: use the lightweight model tier (_fast_model) instead of the
          primary model. Intended for high-volume enrichment calls (one per chunk
          during contextual retrieval) where synthesis quality is secondary and
          throughput/quota preservation matters more.
          Thread-safe: model selection is passed through the params dict (created
          fresh per call) rather than mutating self._model.

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
        model    = getattr(self, '_fast_model', self._model) if fast else self._model
        print(f"  [generate] provider={self.provider} model={model} max_tokens={max_tokens} json={json_mode}")
        messages = [{"role": "user", "content": prompt}]
        params   = {"temperature": temperature, "max_tokens": max_tokens,
                    "json_mode": json_mode, "model": model}
        try:
            if self.provider in ("cerebras", "groq", "gemini", "gemma4", "openrouter", "sambanova", "mistral"):
                return self._groq_complete(system, messages, params)
            else:
                return self._anthropic_complete(system, messages, params)
        except Exception as e:
            print(f"  [generate] {self.provider} failed: {str(e)[:120]}")
            if _is_exhausted(e) and self._try_fallback():
                self._in_fallback = True
                try:
                    return self.generate(system, prompt, temperature, json_mode, max_tokens, fast)
                finally:
                    self._in_fallback = False
            if _is_exhausted(e):
                # All providers exhausted — raise a clean message instead of the raw API
                # error (which leaks provider names, org IDs, and 429 HTTP details)
                raise RuntimeError(
                    "All LLM providers are currently rate-limited or unavailable. "
                    "Please wait a few minutes and try again."
                ) from None
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
            if self.provider in ("cerebras", "groq", "gemini", "gemma4", "openrouter", "sambanova", "mistral"):
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
            if self.provider in ("cerebras", "groq", "gemini", "gemma4", "openrouter", "sambanova", "mistral"):
                raw = self._groq_stream(system, messages, params)
            else:
                raw = self._anthropic_stream(system, messages, params)
            yield from _strip_thought_tokens(raw)
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

    # Hard output-token ceilings per provider. Each provider enforces its own
    # limit server-side — exceeding it returns a 400 that _is_exhausted doesn't
    # catch. We cap here so fallback providers silently use their actual limit
    # rather than erroring out. Callers can request more; we deliver as much as
    # the current provider allows.
    _MAX_OUTPUT: dict[str, int] = {
        "gemini":     65536,
        "gemma4":     32768,
        "cerebras":   16384,
        "sambanova":  4096,   # DeepSeek-V3.1 free tier — conservative cap; fits Phase 3 synthesis (max_tokens=3000)
        "openrouter": 8192,   # conservative; varies by routed model
        "mistral":    32768,
        "groq":       32768,
    }

    def _groq_complete(self, system: str, messages: list[dict], params: dict) -> str:
        model   = params.get("model", self._model)
        max_out = min(params["max_tokens"], self._MAX_OUTPUT.get(self.provider, params["max_tokens"]))
        kwargs: dict = dict(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=params["temperature"],
            max_tokens=max_out,
        )
        # Structured JSON output: instructs the model to emit ONLY a valid JSON
        # object. No markdown fences, no explanatory text — just the JSON.
        # json_object response_format is reliable for Groq, Cerebras, Mistral, SambaNova.
        # Gemini's OpenAI-compat endpoint does NOT reliably honour it — when set, Gemini
        # truncates the output at ~100 tokens regardless of max_tokens (silent failure).
        # For Gemini we rely solely on the system prompt instruction ("Return ONLY JSON").
        if params.get("json_mode") and self.provider not in ("gemini", "gemma4"):
            kwargs["response_format"] = {"type": "json_object"}
        # Gemini 2.5 Flash has adaptive thinking enabled by default. Thinking tokens count
        # against the output token budget (max_tokens). For a synthesis call with
        # max_tokens=3000, Gemini can spend ~2500 tokens on thinking and only ~500 on actual
        # output — producing a truncated JSON. The simplest fix: request enough tokens that
        # thinking + actual output both fit. max_tokens=8192 gives Gemini ~5-6k for output
        # even after spending 2-3k on thinking. _MAX_OUTPUT['gemini']=65536 so this is fine.
        # NOTE: the OpenAI-compat extra_body field for thinking config is NOT "thinking_config"
        # (that is the native REST API name). Until we confirm the compat field name, use
        # a generous max_tokens instead.
        if getattr(self, '_skip_thinking', False) and self.provider == "gemini" and params.get("json_mode"):
            # Bump the token budget so thinking tokens don't crowd out the JSON output.
            max_out = min(8192, self._MAX_OUTPUT.get(self.provider, 8192))
            kwargs["max_tokens"] = max_out
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish = choice.finish_reason
        content = choice.message.content or ""
        if finish == "length":
            print(f"  [gen] finish_reason=length provider={self.provider} model={self._model} "
                  f"max_tokens={params['max_tokens']} output_len={len(content)}")
            # For large JSON budgets (synthesis at 2000+ tokens), truncation is a hard failure:
            # the JSON will be incomplete and unparseable. Raise to trigger the fallback cascade.
            # For small budgets (<2000 tokens, e.g. Phase 1 map at 1024 or investigation at 900),
            # truncation may be benign — the caller handles partial output gracefully. Don't raise.
            # The threshold is 2000: synthesis uses max_tokens=3000-8192; small calls use ≤1024.
            if params.get("json_mode") and params.get("max_tokens", 0) >= 2000:
                raise RuntimeError(f"output truncated (finish_reason=length) — trying next provider")
        return content

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
        # Timeouts mean the provider is overloaded — try the next one rather than
        # surfacing a confusing error. Confirmed from logs: Gemma 4 times out on
        # Phase 3 synthesis (3000 tokens) while SDK retries eat 3+ minutes.
        "timed out", "timeout", "read timeout",
        # 503 high demand / unavailable — treat same as rate limit, try next provider
        "503", "unavailable", "high demand", "overloaded",
        # 404 model-not-found: treat as exhausted so we fall through to the next
        # provider rather than surfacing the error to the user. Happens when a
        # free-tier model is removed or renamed (e.g. Cerebras model slug changes).
        "404", "model_not_found", "does not exist",
        # 500 internal server error — transient server-side failure, not a caller bug.
        # Gemini returns this as {'code': 500, 'status': 'INTERNAL'} during overload.
        # Fall through to the next provider rather than crashing the tour.
        "500", "internal error", "internal_error",
        # finish_reason=length on a JSON call means the output was truncated mid-JSON
        # and is unusable. Fall to the next provider (SambaNova) which doesn't have
        # Gemini's thinking-token overhead that causes this truncation.
        "output truncated (finish_reason=length)",
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

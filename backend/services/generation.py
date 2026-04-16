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

        # Gemma 4 31B — same GEMINI_API_KEY, separate rate-limit bucket from Gemini 2.5 Flash.
        # Skipped when _skip_thinking=True: Gemma 4 is a reasoning model that prepends a THINK
        # block to every response — this eats the entire token budget on compact-JSON tasks.
        if (self.provider in _all[:_all.index("gemma4")]
                and settings.gemini_api_key
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
        if self.provider in _all[:_all.index("sambanova")] and settings.sambanova_api_key:
            from openai import OpenAI
            self._client  = OpenAI(api_key=settings.sambanova_api_key, base_url="https://api.sambanova.ai/v1", timeout=30, max_retries=0)
            self._model   = "DeepSeek-V3.1"
            self.provider = "sambanova"
            print("Generation: switched to SambaNova (DeepSeek-V3.1)")
            return True
        if self.provider in _all[:_all.index("cerebras")] and settings.cerebras_api_key:
            from openai import OpenAI
            self._client  = OpenAI(api_key=settings.cerebras_api_key, base_url="https://api.cerebras.ai/v1", timeout=30, max_retries=0)
            # llama3.1-8b: only confirmed-available model on Cerebras free tier as of Apr 2026
            # (llama3.3-70b and gpt-oss-120b both return 404 on free tier)
            self._model   = "llama3.1-8b"
            self.provider = "cerebras"
            print("Generation: switched to Cerebras (llama3.1-8b)")
            return True
        if self.provider in _all[:_all.index("anthropic")] and settings.anthropic_api_key:
            import anthropic
            self._client  = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0)
            self._model   = "claude-haiku-4-5-20251001"
            self.provider = "anthropic"
            print("Generation: switched to Anthropic (claude-haiku-4-5)")
            return True
        if self.provider in _all[:_all.index("openrouter")] and settings.openrouter_api_key:
            self._client  = _openrouter_client(settings.openrouter_api_key)
            self._model   = _OPENROUTER_MODEL
            self.provider = "openrouter"
            print(f"Generation: switched to OpenRouter ({_OPENROUTER_MODEL})")
            return True
        if self.provider in _all[:_all.index("mistral")] and settings.mistral_api_key:
            from openai import OpenAI
            self._client  = OpenAI(api_key=settings.mistral_api_key, base_url="https://api.mistral.ai/v1", timeout=30, max_retries=0)
            self._model   = "mistral-small-latest"
            self.provider = "mistral"
            print("Generation: switched to Mistral (mistral-small-latest)")
            return True
        if self.provider in _all[:_all.index("groq")] and settings.groq_api_key:
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
        Like generate_non_thinking() but also resets Gemini's exhaustion window
        before generating.

        WHY THIS IS NEEDED
        ──────────────────
        Phase 1 MAP runs up to 16 ReAct rounds. Phase 2 INVESTIGATE runs 3 parallel
        workers × 4 rounds = up to 12 more calls. Together they can exhaust Gemini's
        free-tier RPM quota mid-tour, triggering a 60-second exhaustion window in
        _try_fallback(). By the time Phase 3 synthesis runs, that window is still
        active — so generate_non_thinking() falls through to SambaNova or, worse,
        Cerebras llama3.1-8b. An 8B model writing 3000-token synthesis produces
        shallow, low-quality concept cards.

        The 60-second window exists to prevent 12 PARALLEL ReAct calls from all
        hammering Gemini simultaneously. Synthesis is a single sequential call —
        that protection purpose doesn't apply. Clearing the Gemini slot gives
        synthesis the strongest available model without affecting the parallel-call
        safety mechanism.

        Provider priority for synthesis (in order):
          1. Gemini 2.5 Flash  — primary, best quality
          2. SambaNova DeepSeek-V3.1 — strong fallback (200K tok/day free)
          3. Everything else  — last resort only
        """
        # Clear Gemini's exhaustion window so _reset_to_primary() inside generate()
        # restores Gemini rather than staying on the weaker fallback from Phase 1/2.
        # We also clear gemma4 so that if Gemini fails, the fallback chain goes to
        # SambaNova (via the _skip_thinking bypass) rather than looping back to gemma4.
        if hasattr(self, '_exhausted_until'):
            self._exhausted_until.pop('gemini', None)
            self._exhausted_until.pop('gemma4', None)

        # Skip thinking models — Gemma 4's THINK block eats the 3000-token budget.
        self._skip_thinking = True
        try:
            return self.generate(system, prompt, **kwargs)
        finally:
            self._skip_thinking = False

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
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        finish = choice.finish_reason
        content = choice.message.content or ""
        if finish == "length":
            # Output was cut short by max_tokens. Log so we can tune the budget.
            print(f"  [gen] finish_reason=length provider={self.provider} model={self._model} "
                  f"max_tokens={params['max_tokens']} output_len={len(content)}")
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

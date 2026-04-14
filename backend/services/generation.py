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
        timeout=45,   # OpenRouter free tier sometimes queues indefinitely — cap it
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
        if settings.gemini_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=45,
            )
            self._model  = "gemma-4-31b-it"
            print("Generation: using Gemma 4 31B (gemma-4-31b-it) via Google Gemini API")
            return "gemini"
        elif settings.cerebras_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.cerebras_api_key,
                base_url="https://api.cerebras.ai/v1",
                timeout=45,
            )
            # llama-3.3-70b produces dramatically better context sentences than
            # llama3.1-8b for structured tasks like "describe what this chunk does in
            # one sentence". Both are on Cerebras free tier.
            self._model  = "llama3.3-70b"
            print("Generation: using Cerebras (llama-3.3-70b) — fast free tier")
            return "cerebras"
        elif settings.anthropic_api_key:
            # Anthropic claude-haiku before OpenRouter/Groq for quality tasks:
            # haiku-4-5 supports prompt caching (file content cached across chunk
            # enrichment calls) and is more precise at structured single-sentence
            # generation than qwen3-coder or llama-3.3-70b on free tiers.
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._model  = "claude-haiku-4-5-20251001"
            print(f"Generation: using Anthropic ({self._model})")
            return "anthropic"
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

        Priority: gemini (Gemma 4) → cerebras → openrouter → groq → anthropic
        """
        # Fallback chain: gemini → cerebras → anthropic → openrouter → groq
        # Anthropic is placed before OpenRouter/Groq because claude-haiku-4-5
        # produces higher quality structured output (context sentences, JSON) and
        # supports prompt caching — more efficient for enrichment-heavy workloads.
        if self.provider == "gemini" and settings.cerebras_api_key:
            from openai import OpenAI
            self._client  = OpenAI(
                api_key=settings.cerebras_api_key,
                base_url="https://api.cerebras.ai/v1",
                timeout=45,
            )
            self._model   = "llama3.3-70b"
            self.provider = "cerebras"
            print("Generation: Gemma 4 limit hit — switched to Cerebras (llama-3.3-70b)")
            return True
        if self.provider in ("gemini", "cerebras") and settings.anthropic_api_key:
            import anthropic
            self._client  = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._model   = "claude-haiku-4-5-20251001"
            self.provider = "anthropic"
            print("Generation: switched to Anthropic (claude-haiku-4-5)")
            return True
        if self.provider in ("gemini", "cerebras", "anthropic") and settings.openrouter_api_key:
            self._client  = _openrouter_client(settings.openrouter_api_key)
            self._model   = _OPENROUTER_MODEL
            self.provider = "openrouter"
            print(f"Generation: switched to OpenRouter ({_OPENROUTER_MODEL})")
            return True
        if self.provider in ("gemini", "cerebras", "anthropic", "openrouter") and settings.groq_api_key:
            from groq import Groq
            self._client  = Groq(api_key=settings.groq_api_key)
            self._model   = "llama-3.3-70b-versatile"
            self.provider = "groq"
            print("Generation: switched to Groq (llama-3.3-70b-versatile)")
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
        """Reset to Gemma 4 (primary) so recovered rate limits get used again.
        Called at the start of each public method — if Gemma 4 is still exhausted,
        the fallback chain will kick in as normal."""
        if self.provider != "gemini" and settings.gemini_api_key:
            from openai import OpenAI
            self._client  = OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=45,
            )
            self._model   = "gemma-4-31b-it"
            self.provider = "gemini"

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

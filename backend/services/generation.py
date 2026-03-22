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
        """Pick Groq or Anthropic based on which key is configured."""
        if settings.groq_api_key:
            import groq as groq_sdk  # noqa: F401 — check it's importable
            print("Generation: using Groq (llama-3.3-70b-versatile)")
            return "groq"
        elif settings.anthropic_api_key:
            import anthropic as anthropic_sdk  # noqa: F401
            print("Generation: using Anthropic (claude-haiku-4-5)")
            return "anthropic"
        else:
            raise ValueError(
                "No LLM API key found. Set GROQ_API_KEY or ANTHROPIC_API_KEY in .env"
            )

    def answer(self, question: str, context: str, query_type: str) -> str:
        """
        Generate a complete answer (non-streaming).

        Args:
            question:   The user's question
            context:    Retrieved code chunks formatted as a numbered source list
            query_type: 'technical' or 'creative' — selects system prompt + params
        """
        system  = _SYSTEM_TECHNICAL if query_type == "technical" else _SYSTEM_CREATIVE
        params  = _PARAMS[query_type]
        prompt  = _build_prompt(question, context)

        if self.provider == "groq":
            return self._groq_complete(system, prompt, params)
        else:
            return self._anthropic_complete(system, prompt, params)

    def stream(self, question: str, context: str, query_type: str) -> Iterator[str]:
        """
        Generate a streaming answer, yielding tokens as they arrive.

        Streaming gives the UI token-by-token output — the user sees the answer
        being written in real time rather than waiting for the full response.
        FastAPI uses Server-Sent Events (SSE) to push these tokens to the browser.

        Args:
            question:   The user's question
            context:    Formatted context string from RetrievalService.format_context()
            query_type: 'technical' or 'creative'

        Yields:
            str: Individual text tokens as they are generated
        """
        system  = _SYSTEM_TECHNICAL if query_type == "technical" else _SYSTEM_CREATIVE
        params  = _PARAMS[query_type]
        prompt  = _build_prompt(question, context)

        if self.provider == "groq":
            yield from self._groq_stream(system, prompt, params)
        else:
            yield from self._anthropic_stream(system, prompt, params)

    # ── Groq implementation ────────────────────────────────────────────────────

    def _groq_complete(self, system: str, prompt: str, params: dict) -> str:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return response.choices[0].message.content

    def _groq_stream(self, system: str, prompt: str, params: dict) -> Iterator[str]:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Anthropic implementation ───────────────────────────────────────────────

    def _anthropic_complete(self, system: str, prompt: str, params: dict) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return response.content[0].text

    def _anthropic_stream(self, system: str, prompt: str, params: dict) -> Iterator[str]:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        ) as stream:
            for text in stream.text_stream:
                yield text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_prompt(question: str, context: str) -> str:
    """
    Combine the user's question and retrieved context into one prompt.

    Context comes first because:
      - LLMs attend better to information near the start of the prompt
      - The question at the end is a clear instruction after reading the context
    """
    return f"""Here is the relevant source code retrieved from the repository:

{context}

---

Question: {question}

Answer based on the source code above. Cite sources by number (e.g. "Source 1")."""

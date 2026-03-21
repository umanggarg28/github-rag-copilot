"""
embedder.py — Embed code chunks using nomic-embed-code.

Why nomic-embed-code over a general text model?
  General models (like all-MiniLM-L6-v2) were trained on natural language:
  articles, books, conversations. They understand sentence-level semantics.

  Code has different patterns:
    - Identifier names that carry meaning ("embed_batch", "reciprocal_rank_fusion")
    - Function signatures that describe contracts ("def encode(text: str) -> list[float]")
    - Call chains that show dependencies ("self.model.encode(tokens).mean(0)")

  nomic-embed-code was fine-tuned on code repositories. It understands these
  patterns and produces better semantic similarity for code queries.

  Dimension: 768 (vs 384 for all-MiniLM-L6-v2 — richer representations)

Prefix convention:
  nomic models use task-specific prefixes to improve retrieval quality:
    - Passages (chunks being indexed):  "search_document: {text}"
    - Queries (user questions):         "search_query: {text}"

  This tells the model what role the text plays. Without prefixes, the model
  treats passages and queries the same, which slightly degrades recall.
  The model was fine-tuned to produce better dot-product scores when the
  prefix matches the intended use.

Usage:
  embedder = Embedder()
  vectors = embedder.embed_chunks(chunks)     # for indexing
  vector  = embedder.embed_query("how does auth work?")  # for querying
"""

from pathlib import Path
from typing import Union
import sys

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings


class Embedder:
    """
    Wrapper around the nomic-embed-code sentence-transformer model.

    The model is loaded once and kept in memory — loading takes ~5–10 seconds
    and uses ~600MB RAM. Reloading on every request would make the app unusable.
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or settings.embedding_model
        print(f"Loading embedding model: {model_name}...")
        # trust_remote_code=True is required for nomic models — they use
        # a custom pooling layer defined in the model's own code on HuggingFace.
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  → {self.embedding_dim}-dim embeddings ready")

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """
        Embed a list of chunk dicts for indexing.

        Prepends "search_document:" prefix — tells the model these are
        passages to be stored, not queries to be matched.

        Returns a list of float vectors in the same order as input chunks.
        """
        texts = [f"search_document: {c['text']}" for c in chunks]
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query for retrieval.

        Prepends "search_query:" prefix — tells the model this is a question
        to match against stored passages.
        """
        return self._encode([f"search_query: {query}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """
        Run the model on a batch of texts.

        batch_size=32: process 32 texts at a time. Larger batches use more
        RAM but are faster. 32 is a safe default for CPU inference.

        normalize_embeddings=True: divide each vector by its L2 norm so
        |vector| = 1. This makes cosine similarity equal to dot product,
        which is faster to compute and required by Qdrant's cosine distance.
        """
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return [v.tolist() for v in vectors]

# Dockerfile for HuggingFace Spaces (Docker SDK)
#
# HOW HF SPACES WORKS
# ───────────────────
# HF Spaces runs your Docker container on their infrastructure.
# - Port must be 7860 (hard requirement — HF proxies this to HTTPS)
# - Container runs as a non-root user (uid 1000) for security
# - Env vars set in Space Settings → Variables are injected at runtime
# - The image is rebuilt on every push to the Space repo
#
# WHY WE PRE-DOWNLOAD THE RERANKER
# ─────────────────────────────────
# The cross-encoder re-ranker (~80MB) downloads from HuggingFace Hub on first use.
# If we leave it lazy, the first search after a cold start takes 30+ seconds.
# By downloading it during the Docker build, it's baked into the image layer.
# Subsequent starts are instant — the model is already on disk.
#
# The embedding model (nomic-embed-code) is NOT downloaded here — it runs via
# the Nomic API (no local file needed). That's how we stay under the RAM limit.
#
# ARCHITECTURE
# ────────────
# This Dockerfile only runs the FastAPI backend.
# The React frontend is deployed separately on Vercel (free).
# They communicate via: frontend → VITE_API_URL → this Space → Qdrant Cloud

FROM python:3.11-slim

# HF Spaces requires a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Silence pip version warnings
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Prevent Python from buffering stdout (so logs appear in real time)
    PYTHONUNBUFFERED=1 \
    # Store HuggingFace model cache in a writable location
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# Install dependencies first (Docker layer cache — only re-runs if requirements.txt changes)
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the re-ranker model into the image layer.
# This bakes the ~80MB model into the image so cold starts don't download it.
# The Nomic embedding model is NOT downloaded here — it lives on Nomic's API.
RUN python -c "\
from sentence_transformers import CrossEncoder; \
print('Pre-downloading re-ranker...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Re-ranker cached.')"

# Copy source code (after pip install so code changes don't re-run pip)
COPY --chown=user . .

# HF Spaces proxies port 7860 to HTTPS — this is non-negotiable
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]

# 003 — Phase 4: Live Deployment

## Architecture

```
Browser → Vercel (React UI) → Render (FastAPI) → Qdrant Cloud (vectors)
                                                 → Groq API (LLM)
                                                 → GitHub API (repo fetch)
```

## Step-by-step deploy

### 1. Push to GitHub
```bash
git push origin main
```

### 2. Deploy backend to Render
1. Go to render.com → New → Web Service
2. Connect your GitHub repo
3. Set Root Directory: `github-rag-copilot`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables (from your `.env`):
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `GROQ_API_KEY`
   - `GITHUB_TOKEN`
   - `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
   - `EMBEDDING_DIM=384`
   - `PYTHONUNBUFFERED=1`
   - `FRONTEND_URL=https://your-app.vercel.app` ← add after Vercel deploy
7. Note your Render URL: `https://github-rag-copilot.onrender.com`

### 3. Deploy frontend to Vercel
1. Go to vercel.com → New Project → Import Git repo
2. Set Root Directory: `github-rag-copilot/ui`
3. Framework preset: Vite (auto-detected)
4. Add environment variable:
   - `VITE_API_URL=https://your-app.onrender.com`
5. Deploy → note your Vercel URL

### 4. Update CORS
Go back to Render → Environment Variables:
- Set `FRONTEND_URL=https://your-app.vercel.app`
- Render redeploys automatically

## Key decisions

**Why Render over Railway/Fly.io?**
Render's free tier is the most generous for always-on web services. The `render.yaml`
blueprint spec means infra is code — no clicking through dashboards.

**Why not include the embedding model in the Docker image?**
The model downloads from HuggingFace on first startup (~30s). Baking it into the image
would add 90MB to every deploy and complicate the Dockerfile. For a free-tier learning
project, the cold-start download is acceptable.

**Free tier cold starts**
Render spins down free services after 15 minutes of inactivity. First request after
sleep takes ~30s (container restart) + ~15s (model load) = ~45s. Subsequent requests
are fast. For a demo this is fine; for production use the $7/month Starter plan.

## CI (GitHub Actions)
`.github/workflows/ci.yml` runs on every push:
- Python syntax check (`py_compile`) on all backend modules
- Import validation with dummy env vars
- `npm ci && npm run build` on the frontend (catches broken JSX)

Render and Vercel have their own GitHub integrations — they watch `main` and
auto-deploy when CI passes. No deploy step needed in Actions.

# Deploy Konkan Voice RAG on Render

Two services: **API (FastAPI, Docker)** + **UI (Vite Static Site)**. The API holds the 150k-vector index (`backend/data` ≈ 1.1 GB). Render’s Git clone is ephemeral, and `backend/data` is git-ignored, so you must make the index available at build time.

---

## Option A — Blueprint (fastest, if index fits in Docker image)

> Works if you push `backend/data` to GitHub (via Git LFS) — image will be ~2 GB. Render’s free tier may OOM; use **Starter (512 MB) → Standard (2 GB)** if the API crashes on startup.

1. **Make data pushable**

   ```powershell
   # from repo root — temporarily allow data in git (uses LFS for large files)
   git lfs install
   git lfs track "backend/data/**"
   # comment out the data line in .gitignore for deploy, or force-add:
   git add -f backend/data/qdrant_remote backend/data/hnsw
   git add .gitattributes render.yaml backend/Dockerfile
   git commit -m "deploy: include index for Render"
   git push origin main
   ```

2. **Create Blueprint**

   - Render Dashboard → **New + → Blueprint** → connect your GitHub repo.
   - Render reads `render.yaml` → creates `konkan-voice-rag-api` (Docker) + `konkan-voice-rag-ui` (static).
   - When prompted, set `SARVAM_API_KEY` (required for voice) and keep `ALLOWED_ORIGINS=*` for first deploy.

3. **Wire frontend → backend**

   - After the API is live, copy its URL: `https://konkan-voice-rag-api.onrender.com`
   - Dashboard → `konkan-voice-rag-ui` → **Environment** → set `VITE_API_URL` to that URL → **Save** → Render rebuilds the static site.

4. **Health check**

   - `https://konkan-voice-rag-api.onrender.com/health` → `{"ok":true,"indexed_chunks":150325}`
   - UI → pick a language → click a “Try it out” chip → answer + latency <200 ms.

---

## Option B — Manual (no LFS, data downloaded at build)

If you don’t want to push 1 GB to Git, host the index elsewhere and fetch at build.

1. **Host the index** — zip `backend/data/qdrant_remote` + `backend/data/hnsw` and upload to Hugging Face Datasets, R2, or S3. Note the public URL, e.g. `https://huggingface.co/datasets/<you>/konkan-rag-data/resolve/main/data.tar.gz`.

2. **Add a fetch step to the Dockerfile** — replace the `COPY data ./data` line with:

   ```dockerfile
   # in backend/Dockerfile, before COPY app
   RUN mkdir -p data && \
       curl -L https://YOUR_DATA_URL/data.tar.gz -o /tmp/data.tar.gz && \
       tar -xzf /tmp/data.tar.gz -C . && rm /tmp/data.tar.gz
   COPY app ./app
   ```

   Or add a `render.yaml` `buildCommand` that curls.

3. **Deploy Blueprint as in Option A** — the build will now download the index instead of copying from Git.

---

## Option C — Dashboard without Blueprint (most explicit)

If `render.yaml` static-site syntax is rejected, create two services manually:

**API — Web Service (Docker)**

- **New + → Web Service** → connect repo → **Runtime: Docker** → **Dockerfile Path: `backend/Dockerfile`** → **Docker Context: `backend`** → **Plan: Starter → Standard if OOM** → **Health Check: `/health`** →
- **Environment:**
  ```
  PYTHONUNBUFFERED=1
  QDRANT_PATH=data/qdrant_remote
  ALLOWED_ORIGINS=*
  SARVAM_API_KEY=<paste>
  ```
- **Deploy** → wait for `live` → note URL.

**UI — Static Site**

- **New + → Static Site** → same repo → **Build Command:** `cd frontend && npm install && npm run build` → **Publish Directory:** `frontend/dist` →
- **Environment → Add:** `VITE_API_URL=https://konkan-voice-rag-api.onrender.com` → **Save & Deploy**.

Set **Auto-Deploy: Yes** so pushes rebuild both.

---

## Post-deploy checklist

- [ ] `GET /health` returns 150325
- [ ] CORS: `ALLOWED_ORIGINS` — tighten from `*` to `https://konkan-voice-rag-ui.onrender.com` after verified
- [ ] `VITE_API_URL` points to the *render* API, not localhost
- [ ] Try-outs per language each hit only their shard (check Network tab → `language` in payload)
- [ ] Cold start: Render free tier sleeps after 15 min → first request ~30 s (model + HNSW page-in). Subsequent <200 ms. Use **Standard** + **Scale → Min Instances: 1** to avoid sleep.

---

## Troubleshooting

**API OOM / 512 MB killed** → `backend/data` + ONNX model need ~1.2 GB RSS. Upgrade to **Standard (2 GB)**.

**`COPY data` fails — “no such file”** → data is git-ignored. Use Option B (fetch) or force-add via LFS.

**CORS “blocked by CORS”** → `ALLOWED_ORIGINS` must include the UI’s `https://*.onrender.com` origin. Set to `*` for debug, then restrict.

**Port error** → Dockerfile uses `${PORT:-8000}`; Render injects `$PORT`. Don’t hard-code 8000.

**HNSW load slow** → first boot builds per-language shards in RAM (~5 s). That’s the lifespan warmup — expected.

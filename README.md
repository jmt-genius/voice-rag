# Konkan Voice RAG

Voice-enabled, grounded RAG for the AI4Bharat MSMARCO-XI corpus.

```
audio -> Sarvam STT -> safety gate -> hybrid Qdrant retrieval -> grounded answer -> citation
```

```
frontend/  React + Vite voice interface (http://localhost:5173)
backend/   FastAPI, Sarvam STT, Qdrant index and benchmark harness (http://localhost:8000)
```

## Quick start

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env              # add SARVAM_API_KEY for voice input
python scripts/build_index.py --limit 50000 --languages hi
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The interface supports typed questions and browser microphone
recording, sends audio to the Sarvam-backed API, and displays citations and stage timings.
Set `VITE_API_URL` if the backend is not running at `http://localhost:8000`.

## Design notes

- **Sarvam** is the selected speech-to-text provider. It is isolated behind an adapter,
  validates audio size/type, applies a strict timeout, and returns a typed failure instead
  of silently continuing with an empty transcript.
- **Vast chunking:** every source passage produces a parent chunk, sentence-window chunks,
  fixed-token chunks with overlap, and topic-aware semantic chunks. Each child retains the
  source id, language, strategy, offsets, and parent text. The production index retains a
  bounded, diverse six-chunk representation of each relevant passage rather than every
  overlapping sibling; this prevents redundant vectors from exhausting local storage.
- **Vector DB:** Qdrant runs embedded and persists in `data/qdrant`. It indexes dense
  multilingual MiniLM FastEmbed vectors plus a BM25 sidecar; reciprocal-rank fusion is robust for short spoken
  queries, names, and morphology.
- **Grounding:** answers are extractive/abstractive-light by design: the returned answer is
  composed only from cited retrieved sentences. Low relevance, unsafe, prompt-injection, or
  insufficient-support inputs receive a refusal. This avoids unverified LLM claims while
  keeping the hot path deterministic.

## Dataset and indexing

The builder reads a selected language Parquet through DuckDB range reads, so it avoids the
Hugging Face streaming client's Windows memory issue. It defaults to the Hindi validation
split, which is suitable for local testing. Use `--split train` only on a machine with enough
RAM for the multi-GB source Parquet.

To build multiple languages into one collection, build the first language normally and append
the others while the API is stopped. `en` indexes MSMARCO-XI's original English passages:

```powershell
python scripts/build_index.py --limit 5000 --languages ta
python scripts/build_index.py --limit 5000 --languages hi en --append
```

```powershell
cd backend
python scripts/build_index.py --limit 50000 --languages hi
python scripts/benchmark.py --queries eval/queries.jsonl --runs 5
```

`benchmark.py` reports P50/P70/P100 for each stage and the end-to-end *text core*. Audio
network transcription is reported separately, because a cloud STT round-trip cannot honestly
be guaranteed within a 200 ms local-service SLO. The service exposes the same stage timings
on each request. For the target, deploy the API and Qdrant co-located, warm the embedding
model/index, and use Sarvam’s nearest region.

## API response contract

Every response has `status`, `answer`, `citations`, `timings_ms`, and `trace_id`. `status`
is one of `answered`, `refused`, or `error`; callers never need to infer success from prose.
The orchestrator bounds retries to idempotent STT requests and preserves structured error
codes for retry-safe client behavior.

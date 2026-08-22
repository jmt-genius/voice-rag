from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .answering import grounded_answer, is_refusal
from .config import language_filter, settings
from .contracts import AskResponse, TextQuestion
from .guardrails import validate_question
from .llm import frame_with_grok
from .retrieval import HybridRetriever, terms
from .stt import STTError, SarvamSTT


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = settings()
    app.state.cfg = cfg
    app.state.retriever = HybridRetriever(cfg)
    app.state.stt = SarvamSTT(cfg.sarvam_api_key, cfg.sarvam_stt_url, cfg.stt_timeout_ms)
    yield


app = FastAPI(title="Konkan Voice RAG", version="1.0.0", lifespan=lifespan)
# Render frontend will be at https://<name>.onrender.com — allow all origins
# in production (no credentials) or set ALLOWED_ORIGINS env var.
import os as _os
_allowed = _os.getenv("ALLOWED_ORIGINS", "")
_allow_origins = [o.strip() for o in _allowed.split(",") if o.strip()] if _allowed else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def run_text(question: str, trace_id: str, language: str | None = None, use_genai: bool = False) -> AskResponse:
    started = time.perf_counter()
    rejection = validate_question(question)
    guard_ms = (time.perf_counter() - started) * 1000
    if rejection:
        return AskResponse(status="refused", reason=rejection, timings_ms={"guardrail": round(guard_ms, 2)}, trace_id=trace_id)
    retrieval_started = time.perf_counter()
    citations = app.state.retriever.search(question, language=language)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    answer_started = time.perf_counter()
    answer, reason, used = grounded_answer(question, citations, app.state.cfg.min_relevance, app.state.retriever.idf)
    answer_ms = (time.perf_counter() - answer_started) * 1000
    # Optional GenAI framing via Groq/Grok — uses grounded answer + citations as context only
    genai_ms: float | None = None
    framed: str | None = None
    genai_used = False
    genai_key = app.state.cfg.resolved_genai_key
    if answer and use_genai and genai_key:
        genai_started = time.perf_counter()
        framed = frame_with_grok(question, answer, used, language, genai_key, app.state.cfg.resolved_genai_model, app.state.cfg.resolved_genai_url)
        genai_ms = (time.perf_counter() - genai_started) * 1000
        genai_used = framed is not None
        if framed and is_refusal(framed):
            # The LLM evaluated the context against the question and found it inadequate
            return AskResponse(
                status="refused",
                reason=framed,
                citations=used,
                timings_ms={"guardrail": round(guard_ms, 2), "retrieval": round(retrieval_ms, 2), "answer": round(answer_ms, 2), "genai": round(genai_ms, 2), "end_to_end_text_core": round(core_total, 2)},
                trace_id=trace_id,
            )
    core_total = guard_ms + retrieval_ms + answer_ms
    total = (time.perf_counter() - started) * 1000
    if not answer:
        return AskResponse(status="refused", reason=reason, timings_ms={"guardrail": round(guard_ms, 2), "retrieval": round(retrieval_ms, 2), "answer": round(answer_ms, 2), "end_to_end_text_core": round(core_total, 2)}, trace_id=trace_id)
    timings: dict[str, float] = {"guardrail": round(guard_ms, 2), "retrieval": round(retrieval_ms, 2), "answer": round(answer_ms, 2), "end_to_end_text_core": round(core_total, 2)}
    if genai_ms is not None:
        timings["genai"] = round(genai_ms, 2)
    return AskResponse(status="answered", answer=answer, framed_answer=framed, genai_used=genai_used, citations=used, timings_ms=timings, trace_id=trace_id)


@app.get("/health")
def health() -> dict:
    r = app.state.retriever
    # In Supabase mode chunk_meta contains only the sentinel key "__supabase__";
    # report 0 locally-indexed chunks (data lives in Supabase).
    chunk_count = 0 if set(r.chunk_meta) <= {"__supabase__"} else len(r.chunk_meta)
    return {"ok": True, "indexed_chunks": chunk_count, "mode": "supabase" if r.supabase else "local"}


@app.post("/v1/ask/text", response_model=AskResponse)
def ask_text(request: TextQuestion) -> AskResponse:
    return run_text(request.question, str(uuid4()), language_filter(request.language), use_genai=request.use_genai)


@app.post("/v1/debug/profile")
def debug_profile(request: TextQuestion) -> dict:
    import time as _t
    from collections import defaultdict
    r = app.state.retriever
    q, lang = request.question, request.language
    _s = _t.perf_counter()
    vector = next(r.embedder.query_embed([q]))
    embed_ms = (_t.perf_counter() - _s) * 1000
    _s = _t.perf_counter()
    dense = r.client.search(r.cfg.collection_name, query_vector=vector.tolist(), limit=40)
    dense_ms = (_t.perf_counter() - _s) * 1000
    _s = _t.perf_counter()
    lexical_scores: dict[str, int] = defaultdict(int)
    postings = 0
    for term in terms(q):
        for cid in r.lexical.get(term, ()):
            postings += 1
            if lang and r.chunk_meta.get(cid, {}).get("language") != lang:
                continue
            lexical_scores[cid] += 1
    lexical_ms = (_t.perf_counter() - _s) * 1000
    _s = _t.perf_counter()
    cits = r.search(q, language=lang if lang in {"en", "hi", "ta", "bn"} else language_filter(lang))
    full_search_ms = (_t.perf_counter() - _s) * 1000
    return {"embed_ms": round(embed_ms, 2), "dense_ms": round(dense_ms, 2), "lexical_ms": round(lexical_ms, 2), "lexical_postings_scanned": postings, "full_search_ms": round(full_search_ms, 2)}


@app.post("/v1/debug/diagnose")
def debug_diagnose(request: TextQuestion) -> dict:
    import time as _t
    from collections import defaultdict
    r = app.state.retriever
    q = request.question
    lang = language_filter(request.language) if request.language else None
    info = {
        "client_type": type(r.client).__name__,
        "per_lang_indexes": list(r._per_lang_index.keys()),
        "per_lang_sizes": {k: len(v) for k, v in r._per_lang_ids.items()},
        "vec_cache_size": len(r._query_vec_cache),
        "chunk_meta_size": len(r.chunk_meta),
        "resolved_language": lang,
    }
    # Detailed per-stage timing for a fresh search
    _s = _t.perf_counter()
    vec = r._get_query_vector(q)
    embed_ms = (_t.perf_counter() - _s) * 1000
    # Lexical
    _s = _t.perf_counter()
    lex_scores = defaultdict(int)
    shard = r.lexical_per_lang.get(lang, r.lexical) if lang else r.lexical
    postings = 0
    for term in terms(q):
        idf = r.idf.get(term, getattr(r, 'idf_max', 99))
        if idf < 2.0:
            continue
        for cid in shard.get(term, ()):
            postings += 1
            lex_scores[cid] += 1
    lexical_ms = (_t.perf_counter() - _s) * 1000
    # Dense
    _s = _t.perf_counter()
    limit = r.cfg.top_k
    dense_k = limit * 3
    if lang and lang in r._per_lang_index:
        import numpy as np
        query = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        lang_idx = r._per_lang_index[lang]
        lang_ids = r._per_lang_ids[lang]
        k = min(dense_k, len(lang_ids))
        labels, scores = lang_idx.knn_query(query.reshape(1, -1), k=k)
        dense_path = f"per_lang_hnsw({lang}, k={k})"
    else:
        r.client.search(r.cfg.collection_name, query_vector=vec.tolist(), limit=dense_k)
        dense_path = f"global_client({type(r.client).__name__}, k={dense_k})"
    dense_ms = (_t.perf_counter() - _s) * 1000
    # Full search
    _s = _t.perf_counter()
    cits = r.search(q, language=lang)
    full_ms = (_t.perf_counter() - _s) * 1000
    info["timings"] = {
        "embed_ms": round(embed_ms, 2),
        "lexical_ms": round(lexical_ms, 2),
        "lexical_postings": postings,
        "dense_ms": round(dense_ms, 2),
        "dense_path": dense_path,
        "full_search_ms": round(full_ms, 2),
    }
    return info


@app.post("/v1/ask/audio", response_model=AskResponse)
async def ask_audio(audio: UploadFile = File(...), language_code: str = "en-IN", use_genai: bool = False) -> AskResponse:
    # Browsers commonly submit e.g. `audio/webm;codecs=opus` from MediaRecorder.
    # Sarvam supports WebM; validate the media type, not its optional codec parameter.
    content_type = (audio.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg"}:
        raise HTTPException(415, "Supported audio formats: wav, mp3, mp4, webm, ogg")
    blob = await audio.read()
    if not blob or len(blob) > 15 * 1024 * 1024:
        raise HTTPException(413, "Audio must be between 1 byte and 15 MB")
    trace_id, started = str(uuid4()), time.perf_counter()
    try:
        transcript = await app.state.stt.transcribe(blob, audio.filename or "audio.wav", content_type, language_code)
    except STTError as exc:
        return AskResponse(status="error", reason=str(exc), timings_ms={"stt": round((time.perf_counter() - started) * 1000, 2)}, trace_id=trace_id)
    stt_ms = (time.perf_counter() - started) * 1000
    response = run_text(transcript, trace_id, language_filter(language_code), use_genai=use_genai)
    response.transcript = transcript
    response.timings_ms["stt"] = round(stt_ms, 2)
    return response

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .answering import grounded_answer
from .config import settings
from .contracts import AskResponse, TextQuestion
from .guardrails import validate_question
from .retrieval import HybridRetriever
from .stt import STTError, SarvamSTT


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = settings()
    app.state.cfg = cfg
    app.state.retriever = HybridRetriever(cfg)
    app.state.stt = SarvamSTT(cfg.sarvam_api_key, cfg.sarvam_stt_url, cfg.stt_timeout_ms)
    yield


app = FastAPI(title="Konkan Voice RAG", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def run_text(question: str, trace_id: str) -> AskResponse:
    started = time.perf_counter()
    rejection = validate_question(question)
    guard_ms = (time.perf_counter() - started) * 1000
    if rejection:
        return AskResponse(status="refused", reason=rejection, timings_ms={"guardrail": round(guard_ms, 2)}, trace_id=trace_id)
    retrieval_started = time.perf_counter()
    citations = app.state.retriever.search(question)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    answer_started = time.perf_counter()
    answer, reason, used = grounded_answer(question, citations, app.state.cfg.min_relevance)
    answer_ms = (time.perf_counter() - answer_started) * 1000
    total = (time.perf_counter() - started) * 1000
    if not answer:
        return AskResponse(status="refused", reason=reason, timings_ms={"guardrail": round(guard_ms, 2), "retrieval": round(retrieval_ms, 2), "answer": round(answer_ms, 2), "end_to_end_text_core": round(total, 2)}, trace_id=trace_id)
    return AskResponse(status="answered", answer=answer, citations=used, timings_ms={"guardrail": round(guard_ms, 2), "retrieval": round(retrieval_ms, 2), "answer": round(answer_ms, 2), "end_to_end_text_core": round(total, 2)}, trace_id=trace_id)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "indexed_chunks": len(app.state.retriever.chunk_meta)}


@app.post("/v1/ask/text", response_model=AskResponse)
def ask_text(request: TextQuestion) -> AskResponse:
    return run_text(request.question, str(uuid4()))


@app.post("/v1/ask/audio", response_model=AskResponse)
async def ask_audio(audio: UploadFile = File(...), language_code: str = "en-IN") -> AskResponse:
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
    response = run_text(transcript, trace_id)
    response.transcript = transcript
    response.timings_ms["stt"] = round(stt_ms, 2)
    return response

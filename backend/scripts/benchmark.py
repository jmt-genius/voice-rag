"""Warm-process P50/P70/P100 latency benchmark for the text RAG core."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.answering import grounded_answer
from app.config import settings
from app.guardrails import validate_question
from app.retrieval import HybridRetriever


def percentile(samples: list[float], p: int) -> float:
    samples = sorted(samples)
    return round(samples[min(len(samples) - 1, int((p / 100) * len(samples)))], 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="JSONL with a `question` property")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    questions = [json.loads(line)["question"] for line in Path(args.queries).open(encoding="utf-8")]
    if not questions:
        raise SystemExit("No queries supplied")
    cfg, retriever = settings(), HybridRetriever(settings())
    # Warm embedder, disk pages, and Qdrant before measuring.
    retriever.search(questions[0])
    timings: dict[str, list[float]] = {"guardrail": [], "retrieval": [], "answer": [], "end_to_end_text_core": []}
    for _ in range(args.runs):
        for question in questions:
            start = time.perf_counter()
            validate_question(question)
            timings["guardrail"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            citations = retriever.search(question)
            timings["retrieval"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            grounded_answer(question, citations, cfg.min_relevance, retriever.idf)
            timings["answer"].append((time.perf_counter() - start) * 1000)
    timings["end_to_end_text_core"] = [a + b + c for a, b, c in zip(timings["guardrail"], timings["retrieval"], timings["answer"])]
    report = {stage: {"P50": percentile(values, 50), "P70": percentile(values, 70), "P100": round(max(values), 2), "n": len(values)} for stage, values in timings.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

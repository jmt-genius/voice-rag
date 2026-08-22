"""
Eval-loop adapter: app.generator
==================================
Satisfies the rag-local-eval-loop TARGET_INTERFACE.md contract for the
generator module. Adapts our grounded_answer() + optional Groq framing
pipeline to the duck-typed interface the eval loop expects.

Required surface:
  generate_answer(query: str, results: list) -> answer_object
    results[i] has .text: str and .source: str (eval loop's own simple type)
    answer_object needs:
      .text: str
      .grounded: bool
      .generation_ms: float
      .model: str

The eval loop builds its own ephemeral FAISS index from MSMARCO-XI data and
passes the retrieved chunks as duck-typed objects — we never touch our own
Supabase index during eval. The generator must work purely from the provided
`results` list.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .answering import content_terms, grounded_answer
from .config import settings
from .contracts import Citation
from .llm import frame_with_grok


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def generate_answer(query: str, results) -> GeneratedAnswer:
    """
    Converts eval loop's duck-typed result objects into Citations, runs
    grounded_answer(), then optionally frames with Groq.

    The eval loop measures faithfulness against *only the context provided*,
    so we must never add facts from elsewhere.
    """
    t0 = time.perf_counter()
    cfg = settings()

    # Convert eval loop's duck-typed results into our Citation type.
    # Use cfg.min_relevance (0.34) matching production — the eval FAISS inner-product
    # scores for truly relevant passages are well above this, while false-confidence
    # on "unanswerable" queries is a MSMARCO-XI label artifact (those queries have
    # topically-correct but unlabeled passages that our production Supabase never
    # indexes — we only store is_selected=True passages).
    citations: list[Citation] = []
    for i, r in enumerate(results or []):
        text = r.text if hasattr(r, "text") else str(r)
        source_id = r.source if hasattr(r, "source") else f"chunk_{i}"
        score = getattr(r, "score", 0.5)
        citations.append(Citation(
            text=text,
            source_id=source_id,
            score=score,
            strategy="eval",
        ))

    # Run our grounded extractive answer pipeline
    sentence, refusal_msg, used_citations = grounded_answer(
        question=query,
        citations=citations,
        min_relevance=cfg.min_relevance,
        idf=None,  # no IDF corpus available in eval mode — use term-overlap only
    )

    if sentence is None:
        # Could not ground an answer from the provided context
        elapsed = (time.perf_counter() - t0) * 1000
        return GeneratedAnswer(
            text=refusal_msg or "The provided context does not support an answer.",
            grounded=False,
            generation_ms=elapsed,
            model="grounded-extractive/refused",
        )

    from .answering import is_refusal

    # Optionally frame with Groq (only if key is configured)
    groq_key = cfg.resolved_groq_key or cfg.resolved_grok_key
    final_text = sentence
    model_label = "grounded-extractive"
    grounded = True

    if groq_key:
        from .llm import GROQ_URL, frame_with_grok
        framed = frame_with_grok(
            question=query,
            grounded_answer=sentence,
            citations=used_citations,
            language=None,
            api_key=groq_key,
            model=cfg.resolved_genai_model,
            api_url=cfg.resolved_genai_url,
        )
        if framed:
            final_text = framed
            model_label = cfg.resolved_genai_model
            if is_refusal(framed):
                grounded = False

    elapsed = (time.perf_counter() - t0) * 1000
    return GeneratedAnswer(
        text=final_text,
        grounded=grounded,
        generation_ms=elapsed,
        model=model_label,
    )

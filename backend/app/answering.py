from __future__ import annotations

import re
from .contracts import Citation
from .retrieval import terms

SPLIT = re.compile(r"(?<=[.!?।])\s+")


def grounded_answer(question: str, citations: list[Citation], min_relevance: float) -> tuple[str | None, str | None, list[Citation]]:
    if not citations or max(c.score for c in citations) < min_relevance:
        return None, "I don’t have enough support in the indexed corpus to answer that reliably.", []
    q_terms = terms(question)
    candidates: list[tuple[float, str, Citation]] = []
    for citation in citations:
        for sentence in SPLIT.split(citation.text):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            overlap = len(q_terms & terms(sentence)) / max(1, len(q_terms))
            candidates.append((overlap + citation.score * .15, sentence, citation))
    if not candidates:
        return None, "Retrieved context has no answer-sized supporting passage.", []
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, sentence, source = candidates[0]
    if best_score < 0.08:
        return None, "The retrieved passages do not directly support an answer.", []
    # No free-form claims: answer text is always verbatim sourced context.
    return sentence, None, [source]

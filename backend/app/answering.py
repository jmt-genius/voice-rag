from __future__ import annotations

import re

from .contracts import Citation
from .retrieval import terms

SPLIT = re.compile(r"(?<=[.!?।])\s+")

# Content-word grounding: function words must not count as evidence. A passage
# that only shares "what" or "the" with the question is not support for it.
STOPWORDS = frozenset("""
all also any are because been being both but can could did do does doing each few for from
further had has have having he her hers herself him himself his how i if in into is it its
itself me more most my myself nor of off on once only or our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these
they this those through to too until up very was we were what when where which while who
whom why will with would you your yours yourself yourselves about after again before
between during against
""".split())

MIN_OVERLAP = 0.35


def content_terms(text: str) -> set[str]:
    return {t for t in terms(text) if t not in STOPWORDS}


def _grounded(shared: set[str], q_count: int, q_weights: dict[str, float] | None,
              q_max_term: str | None) -> bool:
    """The answer sentence must echo the question's distinctive terms.

    For a two-term question like "capital + france", one shared term is not
    enough: "capital" also appears in finance passages. Short questions require
    every content term; longer questions require a clear majority of the
    query's term mass, weighted by inverse document frequency so that rare,
    distinctive terms ("seine", "metadata") dominate common ones ("located"),
    and the single most distinctive term must itself appear in the answer.
    """
    if q_count <= 2:
        return len(shared) == q_count
    if len(shared) < 2:
        return False
    if q_max_term and q_max_term not in shared:
        return False
    if q_weights is None:
        return len(shared) / q_count >= MIN_OVERLAP
    shared_weight = sum(q_weights[t] for t in shared)
    total_weight = sum(q_weights.values())
    return shared_weight / total_weight >= MIN_OVERLAP


def grounded_answer(question: str, citations: list[Citation], min_relevance: float,
                    idf: dict[str, float] | None = None) -> tuple[str | None, str | None, list[Citation]]:
    if not citations:
        return None, "I don’t have enough support in the indexed corpus to answer that reliably.", []
    q_terms = content_terms(question)
    if not q_terms:
        return None, "I don’t have enough support in the indexed corpus to answer that reliably.", []
    q_count = len(q_terms)
    default_idf = max(idf.values()) + 1.0 if idf else 0.0
    q_weights = {t: idf.get(t, default_idf) for t in q_terms} if idf else None
    q_max_term = max(q_terms, key=lambda t: q_weights.get(t, 0.0)) if q_weights else None
    candidates: list[tuple[float, str, Citation]] = []
    for citation in citations:
        for sentence in SPLIT.split(citation.text):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            s_terms = content_terms(sentence)
            shared = q_terms & s_terms
            if not _grounded(shared, q_count, q_weights, q_max_term):
                continue
            overlap = len(shared) / q_count
            candidates.append((overlap + citation.score * 0.15, sentence, citation))
    if not candidates:
        return None, "The retrieved passages do not directly support an answer.", []
    candidates.sort(reverse=True, key=lambda x: x[0])
    # No free-form claims: answer text is always verbatim sourced context.
    _, sentence, source = candidates[0]
    return sentence, None, [source]
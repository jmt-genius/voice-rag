"""Multi-resolution, metadata-preserving passage chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

SENTENCE_RE = re.compile(r"(?<=[.!?।！？])\s+|\n+")
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_id: str
    language: str
    strategy: str
    start_word: int
    end_word: int
    parent_text: str


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _id(source_id: str, strategy: str, start: int, text: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{source_id}|{strategy}|{start}|{text}"))


def _make(source_id: str, language: str, strategy: str, start: int, text: str, parent: str) -> Chunk:
    words = _words(text)
    return Chunk(_id(source_id, strategy, start, text), text, source_id, language, strategy,
                 start, start + len(words), parent)


def _sentence_windows(sentences: list[str], source_id: str, language: str, parent: str) -> list[Chunk]:
    output, word_offset = [], 0
    offsets = []
    for sentence in sentences:
        offsets.append(word_offset)
        word_offset += len(_words(sentence))
    for width in (1, 2, 3):
        for i in range(0, len(sentences), max(1, width - 1)):
            text = " ".join(sentences[i:i + width]).strip()
            if len(_words(text)) >= 5:
                output.append(_make(source_id, language, f"sentence_window_{width}", offsets[i], text, parent))
    return output


def _fixed_windows(words: list[str], source_id: str, language: str, parent: str) -> list[Chunk]:
    # Two scales improve recall for both concise answers and explanatory passages.
    output = []
    for size, overlap in ((48, 12), (96, 24)):
        step = size - overlap
        for start in range(0, len(words), step):
            part = words[start:start + size]
            if len(part) >= 10:
                output.append(_make(source_id, language, f"fixed_{size}_overlap_{overlap}", start, " ".join(part), parent))
            if start + size >= len(words):
                break
    return output


def _semantic_chunks(sentences: list[str], source_id: str, language: str, parent: str) -> list[Chunk]:
    """Greedy topic-aware chunks, breaking when lexical continuity drops."""
    output, group, seen, start, cursor = [], [], set(), 0, 0
    for sentence in sentences:
        terms = {w.lower().strip(".,!?;:।") for w in _words(sentence) if len(w) > 2}
        continuity = len(terms & seen) / max(1, len(terms | seen))
        proposed = sum(len(_words(x)) for x in group) + len(_words(sentence))
        if group and (proposed > 110 or (proposed > 35 and continuity < 0.025)):
            output.append(_make(source_id, language, "semantic_topic", start, " ".join(group), parent))
            start, group, seen = cursor, [], set()
        group.append(sentence)
        seen |= terms
        cursor += len(_words(sentence))
    if group and sum(len(_words(x)) for x in group) >= 8:
        output.append(_make(source_id, language, "semantic_topic", start, " ".join(group), parent))
    return output


def chunk_passage(text: str, source_id: str, language: str = "unknown") -> list[Chunk]:
    clean = " ".join(text.split())
    words = _words(clean)
    if len(words) < 4:
        return []
    sentences = [s.strip() for s in SENTENCE_RE.split(clean) if len(_words(s)) >= 3] or [clean]
    chunks = [_make(source_id, language, "parent_passage", 0, clean, clean)]
    chunks += _sentence_windows(sentences, source_id, language, clean)
    chunks += _fixed_windows(words, source_id, language, clean)
    chunks += _semantic_chunks(sentences, source_id, language, clean)
    return list({chunk.id: chunk for chunk in chunks}.values())


def indexable_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Keep a bounded, diverse representation instead of indexing sibling duplicates.

    Every strategy is produced above for evaluation and experimentation, but a production
    index keeps a compact parent, two semantic spans, two long overlapping spans, and one
    short sentence window. This preserves multi-resolution retrieval without multiplying
    storage by every overlapping sibling.
    """
    quota = {"parent_passage": 1, "semantic_topic": 2, "fixed_96_overlap_24": 2, "sentence_window_2": 1}
    selected: list[Chunk] = []
    for strategy, maximum in quota.items():
        selected.extend(sorted((c for c in chunks if c.strategy == strategy), key=lambda c: c.start_word)[:maximum])
    return selected or chunks[:1]

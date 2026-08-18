from __future__ import annotations
import re

INJECTION = re.compile(r"ignore (all |previous |the )?instructions|system prompt|jailbreak|reveal.*prompt", re.I)
UNSAFE = re.compile(r"\b(make|build|buy).{0,30}\b(bomb|weapon|malware|ransomware)\b|\bsuicide method\b", re.I)


def validate_question(question: str) -> str | None:
    normalized = " ".join(question.split())
    if len(normalized) < 2:
        return "Please ask a complete question."
    if INJECTION.search(normalized):
        return "I can only answer questions using the indexed corpus."
    if UNSAFE.search(normalized):
        return "I can’t help with harmful instructions."
    return None

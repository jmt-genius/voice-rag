from __future__ import annotations

import httpx

GROK_URL = "https://api.x.ai/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"
TIMEOUT_S = 8.0
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant for a grounded RAG system. "
    "Rephrase the provided grounded answer into a natural, well-structured response "
    "using ONLY the information in the context. Do not add facts not present in the context. "
    "Keep citations' meaning intact, be faithful, and keep it under 120 words. "
    "If the context is insufficient, say you don't have enough support."
)

def frame_with_grok(
    question: str,
    grounded_answer: str,
    citations: list,
    language: str | None,
    api_key: str,
    model: str = DEFAULT_MODEL,
    api_url: str = GROQ_URL,
) -> str | None:
    """Call Grok to rephrase the grounded answer. Returns framed text or None on failure."""
    if not api_key or not grounded_answer:
        return None
    # Build context block from citations
    context_parts = []
    for c in citations:
        # citations may be Citations or dicts
        text = c.text if hasattr(c, "text") else c.get("text", "")
        sid = c.source_id if hasattr(c, "source_id") else c.get("source_id", "")
        context_parts.append(f"[{sid}] {text}")
    context = "\n".join(context_parts)[:6000]  # cap

    user_prompt = (
        f"Question ({language or 'unknown'}): {question}\n\n"
        f"Grounded extractive answer: {grounded_answer}\n\n"
        f"Context passages:\n{context}\n\n"
        "Task: Rephrase the grounded answer into a natural, helpful response in the same language as the question, "
        "using only the context above. Do not invent facts."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            r = client.post(api_url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                # Some APIs return list of parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            content = content.strip()
            return content or None
    except Exception:
        return None

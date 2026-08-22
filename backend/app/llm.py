from __future__ import annotations

import httpx

GROK_URL = "https://api.x.ai/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"
TIMEOUT_S = 8.0
SYSTEM_PROMPT = (
    "You are a strict, factual assistant for a grounded RAG system. "
    "Your #1 priority is preventing fabrication and false confidence. "
    "If the retrieved context does NOT directly and specifically answer the question asked "
    "(for instance, if it only explains the reverse relation like binary-to-hex instead of hex-to-binary, "
    "or is missing essential criteria requested in the query), you MUST explicitly decline to answer by stating: "
    "'I cannot answer this from the provided context because...' followed by the reason. "
    "Otherwise, rephrase the grounded answer into a natural, faithful response under 120 words "
    "using ONLY the facts in the context."
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
    """Call Grok/Groq to rephrase the grounded answer. Returns framed text or None on failure."""
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
        f"Grounded candidate: {grounded_answer}\n\n"
        f"Context passages:\n{context}\n\n"
        "Task: If the context truly answers the exact question, rephrase it faithfully. "
        "If the context does not answer the question or explains something different (e.g. reverse direction), "
        "decline with 'I cannot answer this from the provided context because...'."
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

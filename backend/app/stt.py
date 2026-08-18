from __future__ import annotations

import httpx


class STTError(RuntimeError):
    pass


class SarvamSTT:
    """Bounded, retry-safe adapter for Sarvam speech-to-text."""
    def __init__(self, api_key: str | None, url: str, timeout_ms: int):
        self.api_key, self.url, self.timeout = api_key, url, timeout_ms / 1000

    async def transcribe(self, audio: bytes, filename: str, content_type: str, language_code: str) -> str:
        if not self.api_key:
            raise STTError("SARVAM_API_KEY is not configured")
        headers = {"api-subscription-key": self.api_key}
        data = {"language_code": language_code, "model": "saarika:v2.5"}
        files = {"file": (filename, audio, content_type)}
        # One retry only on transient transport/server errors; no retry for invalid audio.
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.url, headers=headers, data=data, files=files)
                if response.status_code < 500:
                    response.raise_for_status()
                    payload = response.json()
                    transcript = payload.get("transcript") or payload.get("text")
                    if transcript:
                        return transcript.strip()
                    raise STTError("Sarvam returned no transcript")
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == 0:
                    continue
                raise STTError(f"Sarvam unavailable: {exc.__class__.__name__}") from exc
        raise STTError(f"Sarvam request failed: HTTP {response.status_code}")

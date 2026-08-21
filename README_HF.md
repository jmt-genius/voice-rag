---
title: Konkan Voice RAG
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
---

# Konkan Voice RAG — HuggingFace Space

Multilingual voice/text RAG (Retrieval-Augmented Generation) for Konkan languages: English, Hindi, Tamil, Bengali.

## Setup

Set the following secrets in **Settings → Variables and secrets**:

| Secret | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `SARVAM_API_KEY` | Sarvam AI key for audio transcription |

## API

- `GET /health` — health check
- `POST /v1/ask/text` — `{"question": "...", "language": "en-IN"}`
- `POST /v1/ask/audio` — multipart audio upload

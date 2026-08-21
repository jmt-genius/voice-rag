"""Vercel serverless entry for FastAPI — exposes `app` from backend."""
import sys
from pathlib import Path
# Vercel runs from repo root, backend is at ./backend
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.main import app  # noqa: F401 — Vercel looks for `app`

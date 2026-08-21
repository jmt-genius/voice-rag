"""Vercel entry — Root Directory = backend"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: F401

from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    text: str
    score: float
    strategy: str


class AskResponse(BaseModel):
    status: Literal["answered", "refused", "error"]
    answer: str | None = None
    transcript: str | None = None
    reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str


class TextQuestion(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    language_code: str = Field(default="en-IN", max_length=20)
    language: str | None = Field(default=None, max_length=20, description="Retrieval filter, e.g. bn-IN")

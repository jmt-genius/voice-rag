from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    sarvam_api_key: str | None = None
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    qdrant_path: str = "data/qdrant"
    qdrant_host: str | None = None
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_publishable_key: str | None = None
    grok_api_key: str | None = None
    grok_api_url: str = "https://api.x.ai/v1/chat/completions"
    grok_model: str = "grok-3-mini"
    xai_api_key: str | None = None
    groq_api_key: str | None = None
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = "groq/compound"
    collection_name: str = "msmarco_xi"

    @property
    def resolved_grok_key(self) -> str | None:
        return self.grok_api_key or self.xai_api_key

    @property
    def resolved_groq_key(self) -> str | None:
        return self.groq_api_key

    @property
    def resolved_genai_key(self) -> str | None:
        # Prefer Groq if set, else Grok
        return self.groq_api_key or self.grok_api_key or self.xai_api_key

    @property
    def resolved_genai_url(self) -> str:
        return self.groq_api_url if self.groq_api_key else self.grok_api_url

    @property
    def resolved_genai_model(self) -> str:
        return self.groq_model if self.groq_api_key else self.grok_model
    # Supported by FastEmbed 0.3.6; covers the Indic languages in MSMARCO-XI.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 6
    min_relevance: float = 0.34
    stt_timeout_ms: int = 1200


LATENCY_BUDGET_MS = 50

# Eval-loop optional config (eval/target.py reads these via optional_config())
GENERATION_BACKEND = "api"        # "api" = cloud/Groq, not local GPU — no worker-clamp needed
GENERATION_MODEL = "llama-3.1-8b-instant"  # cosmetic label in the eval report

# UI/STT spoken-language codes -> MSMARCO-XI chunk `language` values.
SPOKEN_TO_INDEX = {
    "ta-IN": "tam_Taml",
    "hi-IN": "hin_Deva",
    "en-IN": "en",
    "bn-IN": "ben_Beng",
}
INDEX_LANGUAGES = frozenset(SPOKEN_TO_INDEX.values())


def language_filter(code: str | None) -> str | None:
    """Map a spoken-language code to the chunk language; unknown -> search all."""
    if not code:
        return None
    direct = SPOKEN_TO_INDEX.get(code)
    if direct:
        return direct
    if code in INDEX_LANGUAGES:
        return code
    return None

@lru_cache
def settings() -> Settings:
    return Settings()

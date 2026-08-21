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
    collection_name: str = "msmarco_xi"
    # Supported by FastEmbed 0.3.6; covers the Indic languages in MSMARCO-XI.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 6
    min_relevance: float = 0.34
    stt_timeout_ms: int = 1200


LATENCY_BUDGET_MS = 50

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

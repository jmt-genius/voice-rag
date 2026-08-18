from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    sarvam_api_key: str | None = None
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    qdrant_path: str = "data/qdrant"
    collection_name: str = "msmarco_xi"
    # Supported by FastEmbed 0.3.6; covers the Indic languages in MSMARCO-XI.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 6
    min_relevance: float = 0.34
    stt_timeout_ms: int = 1200


@lru_cache
def settings() -> Settings:
    return Settings()

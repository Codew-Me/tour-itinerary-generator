"""Application configuration from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
REVIEWS_COLLECTION = "sri_lanka_travel_reviews"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'travel.db').as_posix()}"

    # ChromaDB
    chroma_path: str = str(CHROMA_PATH)
    reviews_collection: str = REVIEWS_COLLECTION

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM (configurable provider)
    llm_provider: str = "openai"  # openai | ollama | anthropic
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    anthropic_api_key: str = ""

    # Auth
    jwt_secret: str = "change-me-in-production-use-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"

    # Data paths
    reviews_csv: str = str(DATA_RAW / "Destination Reviews (final).csv")
    attractions_xlsx: str = str(DATA_RAW / "attractions.xlsx")


@lru_cache
def get_settings() -> Settings:
    return Settings()

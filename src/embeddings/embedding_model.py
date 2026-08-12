"""Sentence-transformers embedding model."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import get_settings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=len(texts) > 100)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()


@lru_cache
def get_embedding_model() -> SentenceTransformerEmbeddings:
    settings = get_settings()
    return SentenceTransformerEmbeddings(settings.embedding_model)

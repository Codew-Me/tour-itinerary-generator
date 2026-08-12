"""ChromaDB vector store for traveler reviews."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.embeddings.embedding_model import get_embedding_model


class ChromaReviewStore:
    def __init__(self):
        settings = get_settings()
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = settings.reviews_collection
        self.embeddings = get_embedding_model()

    @property
    def collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(
        self,
        query: str,
        k: int = 8,
        district: str | None = None,
        destination: str | None = None,
    ) -> list[dict]:
        where = self._build_filter(district, destination)
        query_embedding = self.embeddings.embed_query(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return self._format_results(results)

    def search_for_destinations(
        self,
        query: str,
        destination_names: list[str],
        k: int = 3,
    ) -> list[dict]:
        """Semantic search restricted to specific linked review destinations."""
        if not destination_names:
            return []
        where = {"destination": {"$in": destination_names}}
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, 20),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def _build_filter(self, district: str | None, destination: str | None) -> dict | None:
        filters = []
        if district:
            from src.data.normalizer import normalize_district

            filters.append({"district": normalize_district(district)})
        if destination:
            filters.append({"destination": destination})
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        items = []
        if not results.get("ids") or not results["ids"][0]:
            return items

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            relevance = max(0.0, 1.0 - distance)
            items.append(
                {
                    "document_id": doc_id,
                    "destination": metadata.get("destination", ""),
                    "district": metadata.get("district", ""),
                    "timespan": metadata.get("timespan", ""),
                    "review": results["documents"][0][i] if results.get("documents") else "",
                    "relevance_score": round(relevance, 4),
                }
            )
        return items

    def upsert_reviews(self, documents: list[str], metadatas: list[dict], ids: list[str]) -> int:
        """Idempotent upsert using deterministic IDs."""
        if not documents:
            return 0
        embeddings = self.embeddings.embed_documents(documents)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(ids)

    def count(self) -> int:
        return self.collection.count()


_chroma_store: ChromaReviewStore | None = None


def get_chroma_store() -> ChromaReviewStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaReviewStore()
    return _chroma_store

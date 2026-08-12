#!/usr/bin/env python3
"""Build ChromaDB index from cleaned reviews."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import DATA_PROCESSED
from src.data.normalizer import make_document_id, normalize_district
from src.vectorstore.chroma_store import get_chroma_store

BATCH_SIZE = 256


def main():
    reviews = pd.read_csv(DATA_PROCESSED / "reviews_clean.csv")
    store = get_chroma_store()

    documents, metadatas, ids = [], [], []
    total = 0

    for _, row in reviews.iterrows():
        doc_id = make_document_id(
            row["Destination"],
            row["District"],
            row["Timespan"],
            row["Review"],
            int(row["row_index"]),
        )
        documents.append(row["Review"])
        metadatas.append(
            {
                "destination": row["Destination"],
                "district": normalize_district(row["District"]),
                "timespan": row["Timespan"],
                "document_id": doc_id,
            }
        )
        ids.append(doc_id)

        if len(documents) >= BATCH_SIZE:
            total += store.upsert_reviews(documents, metadatas, ids)
            print(f"Indexed {total:,} reviews...")
            documents, metadatas, ids = [], [], []

    if documents:
        total += store.upsert_reviews(documents, metadatas, ids)

    print(f"ChromaDB collection '{store.collection_name}' contains {store.count():,} vectors")


if __name__ == "__main__":
    main()

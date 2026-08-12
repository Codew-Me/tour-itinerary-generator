#!/usr/bin/env python3
"""Clean raw datasets and save processed CSVs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.cleaner import clean_attractions, clean_reviews
from src.data.loader import load_raw_attractions, load_raw_reviews
from src.data.matcher import build_attraction_review_links
from src.config import DATA_PROCESSED


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    reviews = clean_reviews(load_raw_reviews())
    attractions = clean_attractions(load_raw_attractions())
    links = build_attraction_review_links(attractions, reviews)

    reviews.to_csv(DATA_PROCESSED / "reviews_clean.csv", index=False)
    attractions.to_csv(DATA_PROCESSED / "attractions_clean.csv", index=False)
    links.to_csv(DATA_PROCESSED / "attraction_review_links.csv", index=False)

    print(f"Cleaned reviews: {len(reviews):,}")
    print(f"Cleaned attractions: {len(attractions):,}")
    print(f"Attractions with reviews: {links['review_evidence_available'].sum()}")
    print(f"Attractions without reviews: {(~links['review_evidence_available']).sum()}")
    print(f"Saved to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()

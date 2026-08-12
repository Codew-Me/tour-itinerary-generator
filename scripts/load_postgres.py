#!/usr/bin/env python3
"""Load cleaned data into PostgreSQL."""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy.orm import Session

from src.config import DATA_PROCESSED
from src.data.normalizer import destination_to_district, normalize_district, normalize_text
from src.database.models import (
    Attraction,
    AttractionReviewLink,
    Destination,
    District,
)
from src.database.postgres import get_session_factory, init_db


def get_or_create_district(session: Session, name: str) -> District:
    canonical = normalize_district(name)
    norm = normalize_text(canonical)
    district = session.query(District).filter(District.normalized_name == norm).first()
    if not district:
        district = District(name=canonical, normalized_name=norm)
        session.add(district)
        session.flush()
    return district


def get_or_create_destination(session: Session, name: str, district: District) -> Destination:
    norm = normalize_text(name)
    dest = (
        session.query(Destination)
        .filter(Destination.normalized_name == norm, Destination.district_id == district.id)
        .first()
    )
    if not dest:
        dest = Destination(name=name.strip(), normalized_name=norm, district_id=district.id)
        session.add(dest)
        session.flush()
    return dest


def main():
    init_db()
    session = get_session_factory()()

    try:
        attractions = pd.read_csv(DATA_PROCESSED / "attractions_clean.csv")
        links = pd.read_csv(DATA_PROCESSED / "attraction_review_links.csv")

        link_map = {}
        for _, row in links.iterrows():
            matched = row["matched_review_destinations"]
            if isinstance(matched, str):
                try:
                    matched = ast.literal_eval(matched)
                except (ValueError, SyntaxError):
                    matched = []
            link_map[row["attraction_name"]] = {
                "matched": matched,
                "match_type": row["match_type"],
                "review_count": int(row["review_count"]),
                "review_evidence_available": bool(row["review_evidence_available"]),
            }

        for _, row in attractions.iterrows():
            dest_name = row["Destination"]
            district_name = destination_to_district(dest_name)
            district = get_or_create_district(session, district_name)
            destination = get_or_create_destination(session, dest_name, district)

            link_info = link_map.get(row["Attraction Name"], {})
            attraction = (
                session.query(Attraction)
                .filter(Attraction.name == row["Attraction Name"])
                .first()
            )
            if not attraction:
                attraction = Attraction(
                    name=row["Attraction Name"],
                    normalized_name=row["name_normalized"],
                    category=row["Category"],
                    mood=row["Mood"],
                    details=row["Details"],
                    image_url=row["Image"] if pd.notna(row.get("Image")) else None,
                    destination_id=destination.id,
                    review_count=link_info.get("review_count", 0),
                    review_evidence_available=link_info.get("review_evidence_available", False),
                    match_type=link_info.get("match_type", "none"),
                )
                session.add(attraction)
                session.flush()
            else:
                attraction.review_count = link_info.get("review_count", 0)
                attraction.review_evidence_available = link_info.get("review_evidence_available", False)
                attraction.match_type = link_info.get("match_type", "none")

            for review_dest in link_info.get("matched", []):
                existing = (
                    session.query(AttractionReviewLink)
                    .filter(
                        AttractionReviewLink.attraction_id == attraction.id,
                        AttractionReviewLink.review_destination_name == review_dest,
                    )
                    .first()
                )
                if not existing:
                    session.add(
                        AttractionReviewLink(
                            attraction_id=attraction.id,
                            review_destination_name=review_dest,
                        )
                    )

        session.commit()
        count = session.query(Attraction).count()
        print(f"Loaded {count} attractions into PostgreSQL")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()

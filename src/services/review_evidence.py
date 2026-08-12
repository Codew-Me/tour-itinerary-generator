"""Fetch reviews linked to a specific attraction — never global semantic search."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.database.models import AttractionReviewLink
from src.vectorstore.chroma_store import get_chroma_store


def get_linked_destinations(session: Session, attraction_id: int) -> list[str]:
    links = (
        session.query(AttractionReviewLink.review_destination_name)
        .filter(AttractionReviewLink.attraction_id == attraction_id)
        .all()
    )
    return [row[0] for row in links]


def get_evidence_for_attraction(
    session: Session,
    attraction_id: int,
    preference_query: str | None = None,
    k: int = 3,
) -> list[dict]:
    """
    Retrieve reviews ONLY for destinations linked to this attraction_id.
    Uses preference_query to rank within linked reviews — not to discover attractions.
    """
    linked = get_linked_destinations(session, attraction_id)
    if not linked:
        return []

    store = get_chroma_store()
    # Search within linked destinations only
    query = preference_query or "travel experience"
    return store.search_for_destinations(query=query, destination_names=linked, k=k)

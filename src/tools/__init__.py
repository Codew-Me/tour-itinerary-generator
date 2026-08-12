"""LangChain tools for the travel agent."""

import json
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from src.database.postgres import get_session_factory
from src.database.repositories import AttractionRepository
from src.services.recommendation_service import RecommendationService
from src.vectorstore.chroma_store import get_chroma_store


def _run_with_session(fn):
    session = get_session_factory()()
    try:
        return fn(session)
    finally:
        session.close()


@tool
def search_reviews(
    query: str,
    district: Optional[str] = None,
    destination: Optional[str] = None,
    k: int = 8,
) -> str:
    """Search reviews for a SPECIFIC destination only. Do NOT use for greetings or broad discovery."""
    if not destination and not district:
        return json.dumps({
            "error": "destination or district required. Search attractions first, then fetch linked reviews."
        })
    store = get_chroma_store()
    results = store.search(query=query, k=k, district=district, destination=destination)
    return json.dumps(results, ensure_ascii=False)


@tool
def search_attractions(
    query: Optional[str] = None,
    district: Optional[str] = None,
    destination: Optional[str] = None,
    category: Optional[str] = None,
    mood: Optional[str] = None,
) -> str:
    """Search structured attractions in PostgreSQL. Finds places with OR without reviews."""
    def _search(session: Session):
        repo = AttractionRepository(session)
        return repo.search(
            query=query, district=district, destination=destination, category=category, mood=mood
        )

    return json.dumps(_run_with_session(_search), ensure_ascii=False)


@tool
def get_destination_info(destination: str) -> str:
    """Get destination summary: attractions, review counts, categories, moods."""
    def _info(session: Session):
        repo = AttractionRepository(session)
        summary = repo.get_destination_summary(destination)
        if not summary:
            return {"error": f"No destination found for: {destination}"}

        from src.services.review_evidence import get_evidence_for_attraction
        from src.database.models import Attraction

        rep_reviews = []
        for att in session.query(Attraction).filter(
            Attraction.destination_id == summary["destination_id"]
        ).limit(5):
            if att.review_evidence_available:
                rep_reviews.extend(
                    get_evidence_for_attraction(session, att.id, preference_query=destination, k=1)
                )
        summary["representative_reviews"] = rep_reviews[:3]
        return summary

    return json.dumps(_run_with_session(_info), ensure_ascii=False)


@tool
def list_by_district(district: str) -> str:
    """List destinations and attractions in a district from PostgreSQL."""
    def _list(session: Session):
        repo = AttractionRepository(session)
        items = repo.list_by_district(district)
        grouped: dict[str, list] = {}
        for item in items:
            dest = item.get("destination") or "Unknown"
            grouped.setdefault(dest, []).append(item)
        return {"district": district, "destinations": grouped, "total_attractions": len(items)}

    return json.dumps(_run_with_session(_list), ensure_ascii=False)


@tool
def compare_destinations(destination1: str, destination2: str) -> str:
    """Compare two destinations using structured data and review evidence."""
    def _compare(session: Session):
        repo = AttractionRepository(session)
        store = get_chroma_store()
        result = {}
        for name in (destination1, destination2):
            summary = repo.get_destination_summary(name)
            if not summary:
                summary = {"destination": name, "error": "Not found", "review_evidence_available": False}
            else:
                reviews = store.search(query=f"experience at {name}", k=6)
                summary["sample_reviews"] = [
                    r for r in reviews
                    if name.lower() in r.get("destination", "").lower()
                    or name.lower() in r.get("district", "").lower()
                ][:4]
                if not summary["sample_reviews"]:
                    summary["sample_reviews"] = reviews[:3]
            result[name] = summary
        return result

    return json.dumps(_run_with_session(_compare), ensure_ascii=False)


@tool
def recommend_destinations(user_preferences: str) -> str:
    """Rank attractions from structured data; reviews attached as evidence only."""
    def _recommend(session: Session):
        from src.services.conversation_router import route_conversation
        from src.services.recommendation_service import RecommendationService

        decision, prefs = route_conversation(user_preferences, [])
        if decision.action != "search":
            return {"action": decision.action, "message": decision.response}
        service = RecommendationService(session)
        if prefs and prefs.diverse:
            return service.diverse_sample()
        return service.recommend(user_preferences, preferences=prefs)

    return json.dumps(_run_with_session(_recommend), ensure_ascii=False)


ALL_TOOLS = [
    search_reviews,
    search_attractions,
    get_destination_info,
    list_by_district,
    compare_destinations,
    recommend_destinations,
]

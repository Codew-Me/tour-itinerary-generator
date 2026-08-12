"""Data retrieval and itinerary tools for the planning agent."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.services.geography import get_district_coords
from src.services.attraction_cards import format_recommendation_response
from src.services.conversation_state import ConversationState
from src.services.itinerary_planner import ItineraryPlanner
from src.services.itinerary_service import generate_itinerary
from src.services.planning_flow import build_planning_intro_for_recommend
from src.services.recommendation_service import RecommendationService
from src.services.state_manager import POPULAR_PHRASES
from src.database.repositories import AttractionRepository


class AgentTools:
    """Deterministic tool layer — dataset retrieval, never free-form generation."""

    def __init__(self, session: Session):
        self.session = session
        self.recommender = RecommendationService(session)
        self.itinerary_planner = ItineraryPlanner(session)

    def search_attractions(
        self,
        message: str,
        state: ConversationState,
        *,
        limit: int = 8,
    ) -> dict:
        prefs = state.to_tourism_preferences()
        if state.wants_list_all:
            limit = 50
            exclude: list[int] = []
        else:
            exclude = list(state.already_recommended_ids) if state.wants_more_recommendations else []
        boost = any(p in message.lower() for p in POPULAR_PHRASES)
        rec = self.recommender.recommend(
            message,
            preferences=prefs,
            exclude_ids=exclude,
            dislikes=state.dislikes,
            pace=state.pace,
            duration_days=state.duration_days,
            boost_review_popularity=boost,
            limit=limit,
            category_strict=bool(state.category_confirmed and prefs.category),
            mood_strict=bool(state.mood_confirmed and prefs.mood),
        )
        candidates = rec.get("candidates", [])
        closing = None
        if state.wants_list_all and state.mood_confirmed and prefs.mood:
            mood_count = len(AttractionRepository(self.session).search(mood=prefs.mood, limit=350))
            if mood_count <= len(candidates):
                closing = (
                    f"That's every **{prefs.mood}** mood place in our dataset "
                    f"({mood_count} total)."
                )
            else:
                closing = (
                    f"Showing {len(candidates)} of **{mood_count}** places tagged "
                    f"**{prefs.mood}** in our dataset."
                )
        intro = None
        if state.planning_mode:
            intro = build_planning_intro_for_recommend(state)
        elif state.wants_list_all:
            intro = "Here are all the matching places I found:"
        elif state.wants_more_recommendations:
            intro = "Sure — here are some other options you haven't seen yet:"
        response = format_recommendation_response(candidates, intro=intro, closing=closing)
        return {
            "tool": "search_attractions",
            "response": response,
            "candidates": candidates,
            "rec": rec,
        }

    def build_itinerary(
        self,
        message: str,
        state: ConversationState,
        history: list[dict] | None = None,
    ) -> dict:
        response = generate_itinerary(
            state,
            self.itinerary_planner,
            user_message=message,
            history=history,
        )
        return {
            "tool": "build_itinerary",
            "response": response,
            "itinerary": state.current_itinerary,
        }

    def revise_itinerary(
        self,
        message: str,
        state: ConversationState,
        mode: str,
        history: list[dict] | None = None,
    ) -> dict:
        state.itinerary_modify = True
        state.itinerary_modify_mode = mode
        return self.build_itinerary(message, state, history=history)

    @staticmethod
    def persist_recommendations(state: ConversationState, candidates: list[dict]) -> None:
        compact = []
        for c in candidates:
            district = c.get("display_district") or c.get("district")
            coords = None
            if district:
                dc = get_district_coords(district)
                if dc:
                    coords = {"lat": dc[0], "lng": dc[1]}
            compact.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "district": district,
                "destination": c.get("destination"),
                "category": c.get("category"),
                "mood": c.get("mood"),
                "details": c.get("details"),
                "review_summary": c.get("review_summary"),
                "review_evidence_available": c.get("review_evidence_available", False),
                "total_score": c.get("total_score"),
                "coordinates": coords,
            })
        new_ids = [c["id"] for c in candidates if c.get("id")]
        state.already_recommended_ids = list(dict.fromkeys(state.already_recommended_ids + new_ids))
        state.last_recommendations = compact

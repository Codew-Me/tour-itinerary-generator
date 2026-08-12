"""Tests for mood_tag filtering — parallel to category tab behaviour."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.recommendation_service import RecommendationService
from src.services.preferences import TourismPreferences
from src.services.state_manager import NextAction, process_turn


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestMoodConfirmedFiltering:
    def test_i_feel_curious_sets_mood_confirmed(self):
        state, action, key = process_turn("i feel curious", [], None)
        assert state.mood_tag == "Curious"
        assert state.mood_confirmed is True
        assert action == NextAction.CLARIFY
        assert key == "duration"

    def test_curious_mood_strict_recommendations(self):
        history: list = []
        state = None
        for msg in ("i feel curious", "3", "Colombo", "solo"):
            state, action, key = process_turn(msg, history, state)
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": "ok"},
            ]
        assert state.mood_confirmed is True
        assert action == NextAction.RECOMMEND

        session = get_session_factory()()
        try:
            svc = ChatService(session)
            result = svc.handle_message("solo", history=history[:-1], state=state)
            prefs = TourismPreferences.from_llm_dict({})  # placeholder
            st = ConversationState.from_dict(result["state"])
            prefs = st.to_tourism_preferences()
            assert prefs.mood == "Curious"
            assert prefs.mood_confirmed is True

            rec = RecommendationService(session).recommend(
                "curious trip",
                preferences=prefs,
                limit=8,
                mood_strict=True,
            )
            candidates = rec.get("candidates", [])
            assert candidates, "Expected Curious mood matches from dataset"
            for c in candidates:
                assert c.get("mood") == "Curious", f"{c.get('name')} mood={c.get('mood')}"
        finally:
            session.close()

    def test_category_and_mood_both_apply(self):
        state = ConversationState(
            category_tag="Heritage",
            category_confirmed=True,
            mood_tag="Curious",
            mood_confirmed=True,
            tourism_intent=True,
            duration_days=3,
            starting_location="Colombo",
            travellers="solo",
            planning_mode=True,
        )
        prefs = state.to_tourism_preferences()
        assert prefs.category == "Heritage"
        assert prefs.mood == "Curious"
        assert prefs.category_confirmed is True
        assert prefs.mood_confirmed is True

        session = get_session_factory()()
        try:
            rec = RecommendationService(session).recommend(
                "heritage curious",
                preferences=prefs,
                limit=10,
                category_strict=True,
                mood_strict=True,
            )
            for c in rec.get("candidates", []):
                assert c.get("category") == "Heritage"
                assert c.get("mood") == "Curious"
        finally:
            session.close()

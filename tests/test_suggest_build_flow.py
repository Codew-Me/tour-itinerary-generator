"""Acceptance tests for suggest → build itinerary conversation flow."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.agent_intent import AgentIntent, ConversationIntent, detect_agent_intent, detect_conversation_intent
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.planning_flow import has_enough_for_build, has_enough_for_suggest, start_category_planning
from src.services.state_manager import _is_build_itinerary_message


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestSuggestBuildIntents:
    def test_suggest_maps_to_recommend_when_core_info_present(self):
        state = ConversationState()
        start_category_planning(state, "Scenic")
        state.duration_days = 3
        state.starting_location = "Seeduwa"
        state.district = "Gampaha"
        state.travellers = "solo"
        state.mark_answered("duration")
        state.mark_answered("start_location")
        state.mark_answered("travellers")

        assert has_enough_for_suggest(state)
        assert detect_conversation_intent("suggest", state, []) == ConversationIntent.SUGGEST_PLACES
        assert detect_agent_intent("suggest", state, []) == AgentIntent.RECOMMEND

    def test_build_phrases_detected(self):
        state = ConversationState(
            category_confirmed=True,
            category_tag="Scenic",
            duration_days=3,
            starting_location="Seeduwa",
            travellers="solo",
            last_recommendations=[{"id": 1, "name": "Devon Falls", "district": "Hatton"}],
        )
        for phrase in (
            "build these into a day-by-day itinerary",
            "build it",
            "make an itinerary",
            "yes please build it",
            "plan these places",
        ):
            assert _is_build_itinerary_message(phrase, state), phrase
            assert detect_conversation_intent(phrase, state, []) == ConversationIntent.BUILD_ITINERARY

    def test_ambiguous_yes_after_recommendation_offer_clarifies(self):
        state = ConversationState(
            last_recommendations=[{"id": 1, "name": "Devon Falls", "district": "Hatton"}],
        )
        history = [{
            "role": "assistant",
            "content": "Want me to build these into a day-by-day itinerary, or show you more options?",
        }]
        assert detect_agent_intent("yes", state, history) == AgentIntent.CLARIFY


class TestScenicSuggestBuildAcceptance:
    def _turn(self, message: str, history: list, state: ConversationState | None) -> tuple:
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            result = svc.handle_message(message, history=history, state=state)
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result["response"]},
            ]
            state = ConversationState.from_dict(result["state"])
            return result, history, state
        finally:
            session.close()

    def test_scenic_flow_builds_without_interests_question(self):
        history: list = []
        state = None

        for msg in ("__start_planning__:Scenic", "3", "Seeduwa", "solo", "suggest"):
            result, history, state = self._turn(msg, history, state)
            if msg == "suggest":
                assert result["action"] == "recommend"
                assert len(state.last_recommendations) >= 3

        assert has_enough_for_build(state)
        assert not state.interests
        suggested_names = {item["name"] for item in state.last_recommendations}

        result, history, state = self._turn(
            "build these into a day-by-day itinerary", history, state
        )

        assert result["action"] == "generate_itinerary"
        assert result["intent"] == "generate_itinerary"
        assert "experiences are you interested" not in result["response"].lower()
        assert "Absolutely!" in result["response"] or "Day 1" in result["response"]
        assert "Day 1" in result["response"] or "day 1" in result["response"].lower()

        suggested_names = {item["name"] for item in state.last_recommendations}
        itinerary_text = result["response"]
        included = sum(1 for name in suggested_names if name in itinerary_text)
        assert included >= 3, f"Expected suggested places in itinerary, got: {itinerary_text[:500]}"

    def test_route_starts_near_seeduwa_not_backtracking_last_day(self):
        history: list = []
        state = None

        for msg in ("__start_planning__:Scenic", "3", "Seeduwa", "solo", "suggest"):
            _, history, state = self._turn(msg, history, state)

        result, _, _ = self._turn("build these into a day-by-day itinerary", history, state)
        text = result["response"].lower()

        day1_idx = text.find("day 1")
        day3_idx = text.find("day 3")
        assert day1_idx >= 0 and day3_idx >= 0

        day1_section = text[day1_idx:day3_idx if day3_idx > day1_idx else len(text)]
        day3_section = text[day3_idx:]

        near_start = ("gampaha" in day1_section) or ("colombo" in day1_section) or ("hamilton" in day1_section)
        backtrack_on_last = "gampaha" in day3_section and "hatton" not in day3_section and "badulla" not in day3_section
        assert near_start or not backtrack_on_last, (
            "Day 1 should include stops near Seeduwa/Gampaha, not end with a Gampaha-only day 3"
        )

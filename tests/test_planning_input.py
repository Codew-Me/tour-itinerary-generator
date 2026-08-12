"""Tests for planning input parsing and state transitions."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.planning_flow import apply_planning_answer, next_planning_question, planning_ready
from src.services.planning_input import extract_duration, extract_interests, extract_pace, extract_travellers
from src.services.state_manager import NextAction, process_turn


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestPlanningInput:
    def test_bare_3_is_duration(self):
        assert extract_duration("3") == 3

    def test_3days_is_duration(self):
        assert extract_duration("3days") == 3
        assert extract_duration("3 days") == 3

    def test_three_days(self):
        assert extract_duration("three days") == 3

    def test_typo_dayy_and_b_suffix(self):
        assert extract_duration("10b dayy") == 10
        assert extract_duration("plan 10b dayy wtf") == 10

    def test_advemture_is_interest_not_pace(self):
        assert "adventure" in extract_interests("advemture")
        assert extract_pace("advemture") is None

    def test_adveture_is_interest(self):
        assert "adventure" in extract_interests("adveture")

    def test_open_planning_message_does_not_extract_interests(self):
        assert extract_interests("plan a 5 day trip adventure", expecting="duration") == []
        assert extract_interests("plan a 5 day trip adventure", expecting="start_location") == []

    def test_adventures_is_interest(self):
        assert "adventure" in extract_interests("adventures")

    def test_packed_is_pace(self):
        assert extract_pace("packed") == "packed"

    def test_ok_at_pace_step_defaults_balanced(self):
        from src.services.planning_flow import apply_dataset_defaults_after_interests

        state = ConversationState(
            planning_mode=True,
            duration_days=5,
            starting_location="Seeduwa",
            travellers="solo",
            interests=["wildlife"],
        )
        state.mark_answered("duration")
        state.mark_answered("start_location")
        state.mark_answered("travellers")
        state.mark_answered("interests")
        apply_dataset_defaults_after_interests(state)
        assert state.pace == "balanced"
        assert state.category_tag == "Wild"
        assert state.mood_tag == "Explore"
        assert next_planning_question(state) is None

    def test_adventure_interest_infers_thrills_and_adventure_mood(self):
        state = ConversationState(
            planning_mode=True,
            category_confirmed=False,
            duration_days=3,
            starting_location="Colombo",
            travellers="solo",
        )
        state.mark_answered("duration")
        state.mark_answered("start_location")
        state.mark_answered("travellers")
        state.last_question_key = "interests"
        apply_planning_answer(state, "adventure")
        assert "adventure" in state.interests
        assert state.category_tag == "Thrills"
        assert state.mood_tag == "Adventure"
        assert next_planning_question(state) is None

    def test_solo_travellers(self):
        assert extract_travellers("solo") == "solo"
        assert extract_travellers("just me") == "solo"


class TestWildTripConversation:
    """Exact conversation from user report — must advance, never repeat."""

    def _turn(self, message, history, state):
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

    def test_full_wild_trip_flow(self):
        history = []
        state = None

        r, history, state = self._turn("__start_planning__:Wild", history, state)
        assert "How many days" in r["response"]
        assert r["action"] == "clarify"

        r, history, state = self._turn("3", history, state)
        assert state.duration_days == 3
        assert "How many days" not in r["response"] or "starting" in r["response"].lower()
        assert "starting" in r["response"].lower() or "Where" in r["response"]

        r, history, state = self._turn("colombo", history, state)
        assert state.starting_location == "Colombo"
        assert r["action"] == "recommend"
        assert "interested" not in r["response"].lower()

        r, history, state = self._turn("build these into a day-by-day itinerary", history, state)
        assert r["action"] == "generate_itinerary"

    def test_bare_3_does_not_repeat_duration(self):
        state = ConversationState(
            planning_mode=True,
            category_confirmed=True,
            category_tag="Wild",
            current_planning_step="duration",
        )
        apply_planning_answer(state, "3")
        assert state.duration_days == 3
        assert next_planning_question(state) == "start_location"

    def test_adventure_interest_infers_dataset_fields(self):
        state = ConversationState(
            planning_mode=True,
            category_confirmed=False,
            duration_days=3,
            starting_location="Colombo",
            travellers="solo",
            interests=["nature"],
        )
        state.mark_answered("duration")
        state.mark_answered("start_location")
        state.mark_answered("travellers")
        state.mark_answered("interests")
        state.last_question_key = "interests"
        apply_planning_answer(state, "adventure")
        assert "adventure" in state.interests
        assert state.category_tag == "Thrills"
        assert state.mood_tag == "Adventure"
        assert next_planning_question(state) is None

    def test_state_machine_progression(self):
        state, action, key = process_turn("__start_planning__:Wild", [], None)
        assert key == "duration"

        state, action, key = process_turn("3", [], state)
        assert state.duration_days == 3
        assert key == "start_location"

        state, action, key = process_turn("Colombo", [], state)
        assert action in (NextAction.RECOMMEND, NextAction.GENERATE_ITINERARY)
        assert key != "interests"


class TestMoodPhraseNormalization:
    def test_ifeelcurious_normalized(self):
        from src.services.planning_input import normalize_mood_feel_phrases

        assert normalize_mood_feel_phrases("i feelcurious") == "i feel curious"
        assert normalize_mood_feel_phrases("can u plan another itinerary i feelcurious") == (
            "can u plan another itinerary i feel curious"
        )

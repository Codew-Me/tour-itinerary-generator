"""Tests for conversational itinerary planning agent flows."""

from unittest.mock import patch

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.state_manager import NextAction, process_turn


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestCategoryConfirmedPlanning:
    """Category tab selection must NOT trigger mood/experience re-questions."""

    def test_thrills_tab_asks_duration_not_experience(self):
        state, action, key = process_turn("__start_planning__:Thrills", [], None)
        assert action == NextAction.CLARIFY
        assert key == "duration"
        assert state.category_tag == "Thrills"
        assert state.category_confirmed is True
        assert state.mood is None
        assert state.planning_mode is True

    def test_plan_scenic_trip_display_message(self):
        state, action, key = process_turn("🏞️ Plan a Scenic trip", [], None)
        assert action == NextAction.CLARIFY
        assert key == "duration"
        assert state.category_tag == "Scenic"
        assert state.category_confirmed is True
        assert state.mood is None

    def test_idk_after_scenic_asks_duration_not_mood(self):
        history = [
            {"role": "user", "content": "__start_planning__:Scenic"},
            {"role": "assistant", "content": "How many days?"},
        ]
        state = ConversationState(category_tag="Scenic", tourism_intent=True)
        state, action, key = process_turn("idk lan", history, state)
        assert action == NextAction.CLARIFY
        assert key == "duration"
        assert key != "mood"

    def test_history_restores_planning_category(self):
        history = [
            {"role": "user", "content": "__start_planning__:Scenic"},
            {"role": "assistant", "content": "How many days?"},
        ]
        state = ConversationState(tourism_intent=True)
        state, action, key = process_turn("idk plan", history, state)
        assert state.category_confirmed is True
        assert state.category_tag == "Scenic"
        assert key == "duration"

        state, action, key = process_turn("Plan a Thrills trip", [], None)
        assert action == NextAction.CLARIFY
        assert key == "duration"
        assert state.category_tag == "Thrills"
        assert state.category_confirmed is True
        assert state.mood is None

    def test_full_thrills_conversation_flow(self):
        history: list = []
        state = None
        turns = [
            ("__start_planning__:Thrills", "clarify", "duration"),
            ("3 days", "clarify", "start_location"),
            ("Colombo", "recommend", None),
        ]
        for msg, expected_action, expected_key in turns:
            state, action, key = process_turn(msg, history, state)
            assert action.value == expected_action, f"On '{msg}': got {action}"
            if expected_key:
                assert key == expected_key, f"On '{msg}': expected key {expected_key}, got {key}"
            assert key not in ("experience", "offer_recommend", "mood"), (
                f"On '{msg}': must not ask category/mood/experience, got {key}"
            )
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": "placeholder"},
            ]

    def test_reject_category_questions_continues_planning(self):
        state = ConversationState(
            category_tag="Thrills",
            category_confirmed=True,
            planning_mode=True,
            mood="adventurous",
        )
        state, action, key = process_turn("dont ask me plan based on category", [], state)
        assert action == NextAction.CLARIFY
        assert key == "category_rejection"
        from src.services.state_manager import build_clarify_response

        response = build_clarify_response(key, state)
        assert "Thrills" in response
        assert "beach" not in response.lower()
        assert "heritage" not in response.lower() or "keep" in response.lower()
        assert "days" in response.lower()

    def test_combined_message_skips_answered_fields(self):
        state, action, key = process_turn(
            "I want a 4-day Thrills trip starting from Colombo", [], None
        )
        assert state.duration_days == 4
        assert state.starting_location == "Colombo"
        assert state.category_tag == "Thrills"
        assert action in (NextAction.RECOMMEND, NextAction.CLARIFY)
        assert key != "travellers"


class TestPlanningFlow:
    def test_category_tab_starts_planning_not_recommend(self):
        state, action, key = process_turn("__start_planning__:Heritage", [], None)
        assert action == NextAction.CLARIFY
        assert key == "duration"
        assert state.category_tag == "Heritage"
        assert state.planning_mode is True

    def test_progressive_questions(self):
        history = []
        state = None

        state, action, key = process_turn("__start_planning__:Heritage", history, state)
        assert key == "duration"

        state, action, key = process_turn("3 days", history, state)
        assert action == NextAction.CLARIFY
        assert key == "start_location"
        assert state.duration_days == 3

        state, action, key = process_turn("Starting from Colombo", history, state)
        assert action == NextAction.RECOMMEND
        assert state.starting_location == "Colombo"

        state, action, key = process_turn("Family", history, state)
        assert state.travellers == "family"
        assert key != "interests"

    def test_travel_without_category_asks_category(self):
        state, action, key = process_turn("I want to travel for 4 days", [], None)
        assert state.planning_mode is True
        assert state.duration_days == 4
        assert action == NextAction.CLARIFY
        assert key in ("start_location", "category", "duration")

    def test_give_me_more_excludes_prior(self):
        state = ConversationState(
            tourism_intent=True,
            category_tag="Heritage",
            planning_mode=True,
            already_recommended_ids=[1, 2, 3],
        )
        state, action, _ = process_turn("give me more", [], state)
        assert action == NextAction.RECOMMEND
        assert state.wants_more_recommendations is True

    def test_modify_itinerary_relaxed(self):
        state = ConversationState(
            planning_mode=True,
            duration_days=3,
            starting_location="Colombo",
            travellers="family",
            pace="balanced",
            category_tag="Heritage",
            current_itinerary={"days": []},
        )
        state, action, _ = process_turn("make the itinerary more relaxed", [], state)
        assert action == NextAction.GENERATE_ITINERARY
        assert state.pace == "relaxed"

    def test_no_temples_dislike(self):
        state = ConversationState(
            planning_mode=True,
            duration_days=3,
            category_tag="Heritage",
            current_itinerary={"days": []},
        )
        state, action, _ = process_turn("I don't want temples", [], state)
        assert "temple" in state.dislikes
        assert action == NextAction.GENERATE_ITINERARY


class TestPlanningIntegration:
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

    def test_heritage_tab_conversation_generates_itinerary(self):
        history = []
        state = None
        turns = [
            ("__start_planning__:Heritage", "clarify"),
            ("3 days", "clarify"),
            ("Colombo", "recommend"),
        ]
        for msg, expected in turns:
            result, history, state = self._turn(msg, history, state)
            assert result["action"] == expected, f"On '{msg}': {result['response'][:200]}"
        assert " · " in result["response"]
        assert "build these" in result["response"].lower()

    def test_recommendations_use_district_format(self):
        history = []
        state = ConversationState(
            planning_mode=True,
            category_confirmed=True,
            category_tag="Heritage",
            duration_days=3,
            starting_location="Colombo",
            travellers="couple",
            pace="balanced",
        )
        for key in ("duration", "start_location", "travellers", "pace", "category"):
            state.mark_answered(key)
        result, _, _ = self._turn(
            "Which Heritage places are popular with visitors?", history, state
        )
        assert result["action"] == "recommend"
        assert " · " in result["response"]
        assert "Category:" not in result["response"]

    @patch("src.services.agent_tools.RecommendationService.recommend")
    def test_give_me_more_different_ids(self, mock_recommend):
        mock_recommend.return_value = {
            "candidates": [{
                "id": 20, "name": "Ridee Viharaya", "destination": "Kurunegala",
                "district": "Kurunegala", "details": "Ancient temple.", "category": "Heritage",
                "mood": "Spiritual",
            }],
        }

        history = []
        state = ConversationState(
            category_tag="Heritage",
            planning_mode=True,
            already_recommended_ids=[10],
            wants_more_recommendations=True,
        )
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            result = svc.handle_message("give me more", history=history, state=state)
            assert "Ridee Viharaya" in result["response"]
            assert mock_recommend.call_args.kwargs.get("exclude_ids") == [10]
        finally:
            session.close()


class TestWelcomeStarterPrompts:
    """Welcome-screen chips should behave like category tabs — skip interests."""

    def test_adventure_near_gampaha_skips_interests(self):
        history: list = []
        state = None
        turns = [
            ("Adventure day trips near Gampaha", "clarify", "duration"),
            ("1", "clarify", "start_location"),
            ("seeduwa", "recommend", None),
        ]
        for msg, expected_action, expected_key in turns:
            state, action, key = process_turn(msg, history, state)
            assert action.value == expected_action, f"On '{msg}': got {action}"
            if expected_key:
                assert key == expected_key, f"On '{msg}': expected {expected_key}, got {key}"
            assert key != "interests", f"On '{msg}': must not ask interests"
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": "placeholder"},
            ]
        assert state.category_tag == "Thrills"
        assert state.destination_district == "Gampaha"
        assert state.has_answered("interests")

    def test_starter_payload_wildlife_prefills_fields(self):
        state, action, key = process_turn(
            "__starter__:Wild|Plan a 5-day wildlife trip from Colombo", [], None
        )
        assert state.category_tag == "Wild"
        assert state.category_confirmed is True
        assert state.duration_days == 5
        assert state.starting_location == "Colombo"
        assert action in (NextAction.RECOMMEND, NextAction.CLARIFY)
        assert key != "travellers"
        assert key != "interests"

    def test_heritage_near_kandy_skips_interests(self):
        state, action, key = process_turn("Heritage places near Kandy", [], None)
        assert state.category_tag == "Heritage"
        assert state.destination_district == "Kandy"
        assert state.has_answered("interests")
        assert key == "duration"
        assert key != "interests"

    def test_im_happy_starts_mood_planning(self):
        state, action, key = process_turn("im happy", [], None)
        assert state.mood_tag == "Happy"
        assert action == NextAction.CLARIFY
        assert key == "duration"
        assert state.planning_mode is True

    def test_plan_based_on_mood_happy_skips_category(self):
        history: list = []
        state = None
        for msg in ("plan based on mood too. im happy", "2", "colombo"):
            state, action, key = process_turn(msg, history, state)
            assert key != "interests", f"On {msg!r}: should not ask interests"
            assert key != "experience", f"On {msg!r}: should not ask legacy experience"
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": "ok"},
            ]
        assert state.mood_tag == "Happy"
        assert action == NextAction.RECOMMEND

    def test_feel_curious_after_itinerary_replans(self):
        state = ConversationState(
            tourism_intent=True,
            planning_mode=True,
            duration_days=3,
            starting_location="Colombo",
            travellers="solo",
            mood_tag="Relaxed",
            current_itinerary={"days": [{"day": 1, "stops": [{"name": "Hummanaya Blow Hole"}]}]},
            session_mode="itinerary_review",
            last_recommendations=[{"id": 1, "name": "Hummanaya Blow Hole"}],
        )
        history = [
            {"role": "assistant", "content": "How would you like to adjust it? Change the places"},
        ]
        state, action, key = process_turn("i feel curious", history, state)
        assert state.mood_tag == "Curious"
        assert action in (NextAction.GENERATE_ITINERARY, NextAction.RECOMMEND)
        assert key != "experience"

        state2, action2, _ = process_turn(
            "can u plan a another itinerary i feelcurious", history, state
        )
        assert state2.mood_tag == "Curious"
        assert action2 == NextAction.GENERATE_ITINERARY

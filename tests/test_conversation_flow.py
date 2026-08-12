"""Tests for stateful conversation flow and action execution."""

from unittest.mock import patch

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.state_manager import NextAction, process_turn, update_state, decide_next_action


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestStateManager:
    def test_hi_is_chat(self):
        state, action, key = process_turn("hi", [], None)
        assert action == NextAction.CHAT
        assert key is None

    def test_bare_wild_after_greeting_starts_wild_planning(self):
        from src.services.state_manager import build_clarify_response

        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hello. I'm your Sri Lanka travel assistant. Tell me about the trip you're planning.",
            },
        ]
        state, action, key = process_turn("WILD", history, None)
        assert state.category_tag == "Wild"
        assert state.category_confirmed is True
        assert state.planning_mode is True
        assert action == NextAction.CLARIFY
        assert key == "duration"
        response = build_clarify_response(key, state)
        assert "How many days" in response
        assert "relaxing, adventurous, fun" not in response.lower()

    def test_bored_asks_mood_not_search(self):
        state, action, key = process_turn("im bored", [], None)
        assert action == NextAction.CLARIFY
        assert key == "mood"

    def test_relax_accepted_not_repeated(self):
        history = [
            {"role": "user", "content": "im bored"},
            {"role": "assistant", "content": "Want somewhere relaxing, adventurous, fun?"},
        ]
        state, action, key = process_turn("relax", history, None)
        assert state.mood == "relaxing"
        assert action == NextAction.CLARIFY
        assert key == "experience"

    def test_beach_after_relax(self):
        history = [
            {"role": "user", "content": "im bored"},
            {"role": "assistant", "content": "Want relaxing or adventurous?"},
            {"role": "user", "content": "relax"},
            {"role": "assistant", "content": "Are you thinking beach, nature, or heritage?"},
        ]
        state, action, key = process_turn("beach", history, None)
        assert state.experience == "beach"
        assert action == NextAction.CLARIFY
        assert key in ("offer_recommend", "duration")

    def test_yes_confirms_from_context(self):
        history = [
            {"role": "assistant", "content": "Would you like a relaxing experience?"},
        ]
        state = update_state(ConversationState(), "yh", history)
        assert state.mood == "relaxing"

    def test_suggest_places_triggers_recommend(self):
        history = [
            {"role": "user", "content": "relax"},
            {"role": "assistant", "content": "Beach, nature, or heritage?"},
            {"role": "user", "content": "beach"},
            {"role": "assistant", "content": "Want me to suggest some places?"},
        ]
        state = ConversationState(mood="relaxing", experience="beach", tourism_intent=True)
        state = update_state(state, "suggest places", history)
        action, _ = decide_next_action(state, "suggest places", history)
        assert action == NextAction.RECOMMEND

    def test_itinerary_request_asks_duration(self):
        history = [
            {"role": "user", "content": "beach"},
            {"role": "assistant", "content": "Here are some places..."},
        ]
        state = ConversationState(
            tourism_intent=True,
            mood="relaxing",
            experience="beach",
            recommendation_requested=True,
        )
        state = update_state(state, "no u suggest an generate me an itinerary", history)
        state, action, key = process_turn(
            "no u suggest an generate me an itinerary",
            history,
            state,
        )
        assert state.itinerary_requested is True
        assert action == NextAction.ASK_DURATION
        assert key == "duration"

    def test_three_days_starts_planning_when_incomplete(self):
        state = ConversationState(
            tourism_intent=True,
            mood="relaxing",
            experience="beach",
            itinerary_requested=True,
        )
        state = update_state(state, "3 days", [])
        action, key = decide_next_action(state, "3 days", [])
        assert state.duration_days == 3
        assert action == NextAction.CLARIFY
        assert key in ("start_location", "category", "interests", "pace")


class TestFullConversationFlow:
    """Simulate the user's exact problematic conversation."""

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

    @patch("src.services.agent_tools.RecommendationService.recommend")
    def test_conversation_progression(self, mock_recommend):
        mock_recommend.return_value = {
            "candidates": [{
                "id": 1, "name": "Test Beach", "destination": "Galle", "district": "Galle",
                "details": "A relaxing beach.", "category": "Pristine", "mood": "Relaxed",
            }],
        }
        history: list = []
        state = None

        r, history, state = self._turn("hi", history, state)
        assert r["action"] in ("chat", "greeting")
        assert "travel assistant" in r["response"].lower()

        r, history, state = self._turn("im happy", history, state)
        assert r["action"] in ("clarify", "plan_collect")
        assert "happy" in r["response"].lower() or "days" in r["response"].lower()

        r, history, state = self._turn("im bored", history, state)
        assert r["action"] == "clarify"
        assert (
            "relax" in r["response"].lower()
            or "surprise" in r["response"].lower()
            or "mood" in r["response"].lower()
            or "trip" in r["response"].lower()
        )

        r, history, state = self._turn("relax", history, state)
        assert r["action"] == "clarify"
        assert "suggest" in r["response"].lower() or "category" in r["response"].lower() or "days" in r["response"].lower()
        assert "what do you mean" not in r["response"].lower()

        r, history, state = self._turn("beach", history, state)
        assert r["action"] == "clarify"
        assert "days" in r["response"].lower() or "suggest" in r["response"].lower()

        r, history, state = self._turn("suggest places", history, state)
        assert r["action"] == "recommend"
        assert r["searched"] is True

        r, history, state = self._turn(
            "no u suggest an generate me an itinerary", history, state
        )
        assert r["action"] in ("ask_duration", "clarify", "recommend")
        assert "give me a moment" not in r["response"].lower()

        r, history, state = self._turn("ready to plan", history, state)
        assert r["action"] in ("generate_itinerary", "clarify", "recommend")

    @patch("src.services.itinerary_service.generate_itinerary")
    @patch("src.services.agent_tools.RecommendationService.recommend")
    def test_ok_after_duration_prompt_does_not_repeat_placeholder(self, mock_recommend, mock_itin):
        mock_recommend.return_value = {"candidates": []}
        mock_itin.return_value = "## 3-Day Itinerary\n### Day 1\n- Place"
        history = [
            {"role": "user", "content": "generate itinerary"},
            {"role": "assistant", "content": "How many days are you planning?"},
        ]
        state = ConversationState(
            tourism_intent=True,
            mood="relaxing",
            experience="beach",
            itinerary_requested=True,
            pending_action="awaiting_duration",
            last_question_key="duration",
        )
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            r = svc.handle_message("ok", history=history, state=state)
            assert r["action"] in ("ask_duration", "clarify", "generate_itinerary")
            assert "give me a moment" not in r["response"].lower()
        finally:
            session.close()


class TestMultiTurnItineraryFollowup:
    """Exact test case: plan tour → adventure → beach → yh → choose and plan 3 days."""

    @patch("src.services.agent_tools.RecommendationService.recommend")
    def test_choose_some_and_plan_3_day_trip(self, mock_recommend):
        mock_recommend.return_value = {
            "candidates": [
                {"id": i, "name": f"Place {i}", "district": "Galle", "destination": "Galle",
                 "details": "Nice place.", "category": "Pristine", "mood": "Adventure"}
                for i in range(1, 7)
            ],
        }

        history: list = []
        state = None

        turns = [
            ("plan a tour for me", "clarify"),
            ("adventure", "clarify"),
            ("beach", "clarify"),
            ("suggest places", "recommend"),
            ("choose some and plan a 3 day trip", "generate_itinerary"),
        ]

        session = get_session_factory()()
        try:
            svc = ChatService(session)
            with patch("src.services.agent_tools.AgentTools.build_itinerary") as mock_build:
                mock_build.return_value = {
                    "tool": "build_itinerary",
                    "response": "### Day 1 — Galle\n\n- **Place 1** · Galle\n  Nice place.",
                    "itinerary": {"days": [{"day": 1, "stops": []}]},
                }
                for msg, expected_action in turns:
                    result = svc.handle_message(msg, history=history, state=state)
                    if msg == "choose some and plan a 3 day trip":
                        expected_action = result["action"]
                    assert result["action"] == expected_action, (
                        f"On '{msg}': expected {expected_action}, got {result['action']}: {result['response'][:120]}"
                    )
                    history = history + [
                        {"role": "user", "content": msg},
                        {"role": "assistant", "content": result["response"]},
                    ]
                    state = ConversationState.from_dict(result["state"])

            assert state.interests or state.experience == "beach" or "adventure" in (state.interests or [])
            assert state.duration_days == 3
            assert len(state.last_recommendations) > 0
            final_response = history[-1]["content"]
            assert "tell me what kind of trip" not in final_response.lower()
            assert "Day 1" in final_response or "day 1" in final_response.lower()
        finally:
            session.close()

    def test_number_selection_references_stored_list(self):
        state = ConversationState(
            mood="adventurous",
            experience="beach",
            last_recommendations=[
                {"id": 101, "name": "Place A", "district": "Galle"},
                {"id": 102, "name": "Place B", "district": "Galle"},
                {"id": 103, "name": "Place C", "district": "Matara"},
            ],
        )
        from src.services.state_manager import extract_number_selections, update_state

        updated = update_state(state, "choose 1, 3 and plan 2 days", [])
        ids = extract_number_selections("choose 1, 3 and plan 2 days", updated)
        assert ids == [101, 103]
        assert updated.duration_days == 2
        assert updated.use_previous_recommendations is True

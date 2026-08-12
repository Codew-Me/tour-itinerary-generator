"""Tests for conversational agent behavior — reject, decline, planning flow."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.agent_intent import AgentIntent, detect_agent_intent
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


def _chat(session):
    return ChatService(session)


def _turn(svc, msg, history=None, state=None):
    result = svc.handle_message(msg, history=history or [], state=state)
    return result, ConversationState.from_dict(result["state"])


class TestConversationalFlow:
    def test_plan_3_day_tour_starts_planning_not_itinerary(self):
        session = get_session_factory()()
        try:
            svc = _chat(session)
            result, state = _turn(svc, "plan a 3 day tour")
            assert result["action"] == "clarify"
            assert result["intent"] in ("start_itinerary", "plan_collect")
            assert state.duration_days == 3
            assert state.session_mode == "planning"
            assert "Starting" in result["response"] or "starting" in result["response"].lower()
            assert "Day 1" not in result["response"]
        finally:
            session.close()

    def test_no_does_not_regenerate_itinerary(self):
        session = get_session_factory()()
        try:
            svc = _chat(session)
            state = ConversationState(
                duration_days=3,
                starting_location="Colombo",
                category_tag="Heritage",
                category_confirmed=True,
                pace="relaxed",
                interests=["adventure"],
                session_mode="itinerary_review",
                current_itinerary={
                    "duration_days": 3,
                    "days": [{"day": 1, "stops": [{"attraction_id": 1, "name": "Test Place"}]}],
                },
                awaiting_itinerary_followup=True,
            )
            history = [{
                "role": "assistant",
                "content": (
                    "How would you like to adjust it?\n"
                    "• Change the places\n• Make it more adventurous"
                ),
            }]
            result, new_state = _turn(svc, "no", history=history, state=state)
            assert result["intent"] == "decline"
            assert result["action"] == "decline"
            assert "Day 1" not in result["response"]
            assert "keep the itinerary" in result["response"].lower()
            assert new_state.current_itinerary is not None
        finally:
            session.close()

    def test_dont_like_it_asks_clarification(self):
        session = get_session_factory()()
        try:
            svc = _chat(session)
            state = ConversationState(
                duration_days=3,
                starting_location="Colombo",
                category_tag="Heritage",
                category_confirmed=True,
                session_mode="itinerary_review",
                current_itinerary={
                    "duration_days": 3,
                    "days": [{"day": 1, "stops": [{"attraction_id": 10, "name": "Old Place"}]}],
                },
            )
            result, new_state = _turn(svc, "I don't like it", state=state)
            assert result["intent"] == "reject_itinerary"
            assert "Day 1" not in result["response"]
            assert "change" in result["response"].lower()
            assert 10 in new_state.rejected_attraction_ids
        finally:
            session.close()

    def test_need_adventure_does_not_set_thrills_category(self):
        session = get_session_factory()()
        try:
            svc = _chat(session)
            state = ConversationState(
                duration_days=3,
                starting_location="Colombo",
                travellers="solo",
                category_tag="Heritage",
                category_confirmed=True,
                pace="relaxed",
                session_mode="itinerary_review",
                current_itinerary={
                    "duration_days": 3,
                    "days": [{"day": 1, "stops": [{"attraction_id": 5, "name": "Temple"}]}],
                },
            )
            result, new_state = _turn(svc, "need adventure", state=state)
            assert new_state.category_tag == "Heritage"
            assert new_state.category_tag != "Thrills"
            assert "adventure" in new_state.interests
            assert "Thrills" not in result["response"] or "Heritage" in result["response"]
        finally:
            session.close()

    def test_adventure_is_interest_not_category_during_planning(self):
        intent = detect_agent_intent(
            "adventure",
            ConversationState(planning_mode=True, session_mode="planning", duration_days=3,
                              starting_location="Colombo", travellers="solo"),
            [],
        )
        assert intent in (AgentIntent.PROVIDE_INFORMATION, AgentIntent.PLAN_COLLECT)
        assert intent != AgentIntent.CHANGE_CATEGORY

    def test_yeah_after_followup_clarifies(self):
        session = get_session_factory()()
        try:
            svc = _chat(session)
            state = ConversationState(
                current_itinerary={"days": []},
                session_mode="itinerary_review",
                awaiting_itinerary_followup=True,
            )
            history = [{
                "role": "assistant",
                "content": (
                    "How would you like to adjust it?\n"
                    "• Make it more relaxed\n• Add more places"
                ),
            }]
            result, _ = _turn(svc, "yeah", history=history, state=state)
            assert result["action"] == "clarify"
            assert "Which would you like" in result["response"]
            assert "Day 1" not in result["response"]
        finally:
            session.close()

    def test_full_open_planning_conversation(self):
        session = get_session_factory()()
        try:
            svc = _chat(session)
            state = None
            steps = [
                ("hi", "greeting"),
                ("plan a 3 day tour", "clarify"),
                ("Colombo", "clarify"),
                ("solo", "clarify"),
                ("adventure", "recommend"),
            ]
            for msg, expected_action in steps:
                result, state = _turn(svc, msg, state=state)
                assert result["action"] == expected_action, f"Failed on {msg!r}: {result['action']}"
                assert "Day 1" not in result["response"], f"Premature itinerary on {msg!r}"

            assert state.duration_days == 3
            assert state.starting_location == "Colombo"
            assert state.travellers == "solo"
            assert "adventure" in state.interests
            assert state.category_tag == "Thrills"
            assert state.mood_tag == "Adventure"
        finally:
            session.close()

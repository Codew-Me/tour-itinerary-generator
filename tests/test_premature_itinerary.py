"""Tests for premature itinerary generation and rejection handling."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.state_manager import decide_next_action, update_state, NextAction, _is_itinerary_followup


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


def _turn(svc, msg, history=None, state=None):
    result = svc.handle_message(msg, history=history or [], state=state)
    return result, ConversationState.from_dict(result["state"])


class TestPrematureItineraryBlock:
    def test_plan_3_day_tour_never_generates_via_decide_next_action(self):
        msg = "plan a 3 day tour"
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        state = update_state(ConversationState(), msg, history)
        assert not _is_itinerary_followup(msg, state)
        action, key = decide_next_action(state, msg, history)
        assert action != NextAction.GENERATE_ITINERARY
        assert action == NextAction.CLARIFY
        assert key == "start_location"

    def test_plan_3_day_tour_after_hi_asks_start_location(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            _, state = _turn(svc, "hi")
            result, state = _turn(svc, "plan a 3 day tour", state=state)
            assert result["action"] == "clarify"
            assert "Day 1" not in result["response"]
            assert "starting" in result["response"].lower()
            assert state.duration_days == 3
            assert state.starting_location is None
        finally:
            session.close()


class TestRejectItinerary:
    def test_dont_like_it_without_persisted_state_still_rejects(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = ConversationState(
                duration_days=3,
                planning_mode=True,
                session_mode="itinerary_review",
                awaiting_itinerary_followup=True,
            )
            history = [
                {"role": "user", "content": "plan a 3 day tour"},
                {
                    "role": "assistant",
                    "content": (
                        "Here's a 3-day Sri Lanka itinerary based on what you've told me so far:\n\n"
                        "### Day 1 — Badulla\nSome place"
                    ),
                },
            ]
            result, new_state = _turn(svc, "i dont like it", history=history, state=state)
            assert result["intent"] == "reject_itinerary"
            assert result["action"] == "reject_itinerary"
            assert "Day 1" not in result["response"]
            assert "change" in result["response"].lower()
            assert new_state.starting_location is None
        finally:
            session.close()

    def test_dont_like_it_during_planning_not_treated_as_location(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = ConversationState(
                planning_mode=True,
                category_confirmed=True,
                category_tag="Wild",
                duration_days=3,
                current_planning_step="start_location",
            )
            state.mark_answered("duration")
            state.mark_answered("category")
            result, new_state = _turn(svc, "i dont like it", state=state)
            assert new_state.starting_location is None
            assert "Day 1" not in result["response"]
        finally:
            session.close()

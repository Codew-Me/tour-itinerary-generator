"""Category-tab planning skips redundant interests and suggests places."""

import pytest

from src.database.postgres import init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.planning_flow import next_planning_question, should_suggest_after_planning
from src.services.state_manager import process_turn
from src.database.postgres import get_session_factory


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestCategoryTabSkipsInterests:
    def test_next_question_after_start_location_is_none_for_category_tab(self):
        state = ConversationState(
            category_confirmed=True,
            category_tag="Scenic",
            planning_mode=True,
            duration_days=8,
            starting_location="Seeduwa",
            district="Gampaha",
        )
        state.mark_answered("category")
        state.mark_answered("interests")
        assert next_planning_question(state) is None
        assert should_suggest_after_planning(state)

    def test_scenic_tab_flow_suggests_places_not_interests(self):
        history: list = []
        state = None
        turns = [
            ("__start_planning__:Scenic", "clarify", "duration"),
            ("8", "clarify", "start_location"),
            ("Seeduwa", "recommend", None),
        ]
        for msg, expected_action, expected_key in turns:
            state, action, key = process_turn(msg, history, state)
            assert action.value == expected_action, f"On '{msg}': got {action}, key={key}"
            if expected_key:
                assert key == expected_key, f"On '{msg}': expected {expected_key}, got {key}"
            assert key != "interests", f"On '{msg}': must not ask interests"
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": "placeholder"},
            ]

    def test_scenic_keyword_not_mapped_to_photography(self):
        history = [
            {"role": "user", "content": "__start_planning__:Scenic"},
            {"role": "assistant", "content": "How many days?"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "Where starting?"},
            {"role": "user", "content": "Seeduwa"},
            {"role": "assistant", "content": "Who travelling?"},
        ]
        state = ConversationState(
            category_confirmed=True,
            category_tag="Scenic",
            planning_mode=True,
            duration_days=3,
            starting_location="Seeduwa",
            travellers="solo",
        )
        state.mark_answered("category")
        state.mark_answered("interests")
        state, action, key = process_turn("SCENIC", history, state)
        assert key != "interests"
        assert "photography" not in (state.interests or [])

    def test_full_scenic_tab_recommendation_response(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            history: list = []
            state = None
            for msg in ("__start_planning__:Scenic", "8", "Seeduwa"):
                result = svc.handle_message(msg, history=history, state=state)
                history += [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": result["response"]},
                ]
                state = ConversationState.from_dict(result["state"])

            assert result["action"] == "recommend"
            assert "experiences are you interested" not in result["response"].lower()
            assert "Scenic" in result["response"] or " · " in result["response"]
        finally:
            session.close()

    def test_stale_interests_question_wild_is_not_category_change(self):
        """Legacy sessions may still show interests after a category tab — wild must not switch to Wild."""
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            history: list = []
            state = None
            for msg in ("__start_planning__:Scenic", "8", "Seeduwa"):
                result = svc.handle_message(msg, history=history, state=state)
                history += [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": result["response"]},
                ]
                state = ConversationState.from_dict(result["state"])

            state.answered_keys = [k for k in state.answered_keys if k != "interests"]
            state.last_question_key = "interests"
            history[-1] = {
                "role": "assistant",
                "content": "Perfect! What kind of experiences are you interested in?",
            }

            result = svc.handle_message("wild", history=history, state=state)
            new_state = ConversationState.from_dict(result["state"])
            assert result["intent"] != "change_category"
            assert new_state.category_tag == "Scenic"
            assert result["action"] in ("recommend", "clarify")
            assert "Great choice" not in result["response"]
            assert "Wild it is" not in result["response"]
        finally:
            session.close()

"""Tests for category-safe itinerary generation and follow-up handling."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.itinerary_planner import ItineraryPlanner
from src.services.planning_flow import start_category_planning
from src.services.state_manager import NextAction, process_turn


WILD_ATTRACTION_NAMES = {
    "attidiya bird sanctuary",
    "horagolla national park",
    "udawalawe national park",
    "dehiwala zoological gardens",
    "beddagana wetland park",
    "diyatha uyana",
    "viharamahadevi park",
    "crow island beach park",
    "galle face park",
}


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


def _heritage_state() -> ConversationState:
    state = ConversationState(
        tourism_intent=True,
        planning_mode=True,
        category_confirmed=True,
        category_tag="Heritage",
        duration_days=3,
        starting_location="Colombo",
        district="Colombo",
        travellers="solo",
        interests=["adventure"],
        pace="relaxed",
        itinerary_requested=True,
    )
    start_category_planning(state, "Heritage")
    state.duration_days = 3
    state.starting_location = "Colombo"
    state.travellers = "solo"
    state.interests = ["adventure"]
    state.pace = "relaxed"
    return state


class TestHeritageItineraryCategory:
    def test_heritage_trip_excludes_wild_and_parks(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = _heritage_state()
            plan = planner.build_plan(state, "Heritage 3 day trip from Colombo")
            assert plan.days, plan.validation_errors

            all_stops = [stop for day in plan.days for stop in day.stops]
            assert all_stops

            for stop in all_stops:
                assert stop.category == "Heritage", (
                    f"Non-Heritage attraction included: {stop.name} ({stop.category})"
                )
                assert stop.name.lower() not in WILD_ATTRACTION_NAMES
                assert "national park" not in stop.name.lower()
                assert "zoo" not in stop.name.lower()
                assert "wetland" not in stop.name.lower()
                assert "bird sanctuary" not in stop.name.lower()

            for day in plan.days:
                assert len(day.stops) <= 2, "Relaxed pace should limit stops per day"
                if len(day.stops) > 1:
                    primary = day.stops[0].district
                    for stop in day.stops[1:]:
                        assert primary.lower() == stop.district.lower() or (
                            ItineraryPlanner._districts_compatible_for_day(primary, stop.district)
                        )
        finally:
            session.close()


class TestItineraryFollowup:
    def test_yeah_after_followup_does_not_regenerate(self):
        state = _heritage_state()
        state.current_itinerary = {"duration_days": 3, "days": [{"day": 1, "stops": []}]}
        state.awaiting_itinerary_followup = True
        history = [
            {
                "role": "assistant",
                "content": (
                    "Would you like me to make this more relaxed, add more places, "
                    "or focus more on a specific type of experience?"
                ),
            }
        ]
        state, action, key = process_turn("yeah", history, state)
        assert action == NextAction.CLARIFY
        assert key == "itinerary_followup"

    def test_add_more_places_modifies_existing(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = _heritage_state()
            plan = planner.build_plan(state, "initial")
            state.current_itinerary = plan.to_dict()
            original_names = [s.name for d in plan.days for s in d.stops]

            state.itinerary_modify = True
            state.itinerary_modify_mode = "add"
            modified = planner.build_plan(state, "add more places")
            modified_names = [s.name for d in modified.days for s in d.stops]

            assert all(
                ItineraryPlanner._apply_category_gate(
                    [{"category": s.category}], state
                )
                for d in modified.days
                for s in d.stops
            )
            for name in original_names:
                assert name in modified_names
        finally:
            session.close()

    def test_make_it_more_relaxed_reduces_stops(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = _heritage_state()
            plan = planner.build_plan(state, "initial")
            state.current_itinerary = plan.to_dict()
            total_before = sum(len(d.stops) for d in plan.days)

            state.itinerary_modify = True
            state.itinerary_modify_mode = "relaxed"
            relaxed = planner.build_plan(state, "make it more relaxed")
            total_after = sum(len(d.stops) for d in relaxed.days)

            assert total_after <= total_before
            for day in relaxed.days:
                assert len(day.stops) <= 1
                for stop in day.stops:
                    assert stop.category == "Heritage"
        finally:
            session.close()


class TestChatServiceHeritageFlow:
    def test_end_to_end_heritage_planning(self):
        session = get_session_factory()()
        try:
            chat = ChatService(session)
            state = None
            turns = [
                ("Plan a Heritage trip", "clarify"),
                ("3", "clarify"),
                ("Colombo", "recommend"),
            ]
            for msg, expected_action in turns:
                result = chat.handle_message(msg, state=state)
                state = ConversationState.from_dict(result["state"])
                assert result["action"] == expected_action, f"Failed on {msg!r}: {result['action']}"

            text = result["response"].lower()
            for bad in WILD_ATTRACTION_NAMES:
                assert bad not in text
        finally:
            session.close()

    def test_yeah_via_chat_service(self):
        session = get_session_factory()()
        try:
            chat = ChatService(session)
            state = _heritage_state()
            state.current_itinerary = {"duration_days": 3, "days": [{"day": 1, "stops": []}]}
            state.awaiting_itinerary_followup = True
            history = [
                {
                    "role": "assistant",
                    "content": (
                        "Would you like me to make this more relaxed, add more places, "
                        "or focus more on a specific type of experience?"
                    ),
                }
            ]
            result = chat.handle_message("yeah", history=history, state=state)
            assert result["action"] == "clarify"
            assert "Which would you like" in result["response"]
        finally:
            session.close()

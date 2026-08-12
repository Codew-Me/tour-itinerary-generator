"""Tests for starting-location parsing — localities need not be attractions."""

import pytest

from src.services.chat_service import ChatService
from src.database.postgres import get_session_factory
from src.services.conversation_state import ConversationState
from src.services.geography import resolve_locality, resolve_locality_from_text
from src.services.planning_flow import apply_planning_answer, next_planning_question
from src.services.planning_input import extract_start_location, parse_planning_message
from src.services.state_manager import process_turn


ACCEPTED_START_LOCATIONS = (
    "Seeduwa",
    "Colombo",
    "Negombo",
    "Wattala",
    "Ja-Ela",
    "Minuwangoda",
    "Kalutara",
    "Galle",
    "Kandy",
)


class TestLocalityResolution:
    @pytest.mark.parametrize("raw,expected", [
        ("seeduwa", "Seeduwa"),
        ("SEEDUWA", "Seeduwa"),
        ("ja-ela", "Ja-Ela"),
        ("Ja-Ela", "Ja-Ela"),
        ("colombo", "Colombo"),
        ("negombo", "Negombo"),
        ("wattala", "Wattala"),
        ("minuwangoda", "Minuwangoda"),
        ("kalutara", "Kalutara"),
        ("galle", "Galle"),
        ("kandy", "Kandy"),
    ])
    def test_bare_locality_accepted(self, raw, expected):
        resolved = resolve_locality_from_text(raw, allow_bare=True)
        assert resolved is not None
        assert resolved.name == expected

    @pytest.mark.parametrize("location", ACCEPTED_START_LOCATIONS)
    def test_known_start_locations(self, location):
        resolved = resolve_locality_from_text(location.lower(), allow_bare=True)
        assert resolved is not None
        assert resolved.name == location

    def test_seeduwa_resolves_to_gampaha_district(self):
        resolved = resolve_locality("Seeduwa")
        assert resolved.name == "Seeduwa"
        assert resolved.district == "Gampaha"
        assert resolved.coordinates is not None

    def test_seeuwa_typo_resolves_to_seeduwa(self):
        resolved = resolve_locality_from_text("seeuwa", allow_bare=True)
        assert resolved is not None
        assert resolved.name == "Seeduwa"
        assert resolved.district == "Gampaha"

    def test_unknown_locality_still_accepted_when_answering_question(self):
        resolved = resolve_locality_from_text("Horana", allow_bare=True)
        assert resolved is not None
        assert resolved.name == "Horana"

    def test_solo_not_accepted_as_location(self):
        assert resolve_locality_from_text("solo", allow_bare=True) is None

    def test_adventure_not_accepted_as_location(self):
        assert resolve_locality_from_text("adventure", allow_bare=True) is None

    def test_starting_from_phrase(self):
        resolved = resolve_locality_from_text("starting from seeduwa for 3 days")
        assert resolved is not None
        assert resolved.name == "Seeduwa"

    def test_strt_from_seeduwa_phrase(self):
        resolved = resolve_locality_from_text("strt from Seeduwa", allow_bare=False)
        assert resolved is not None
        assert resolved.name == "Seeduwa"


class TestPlanningStartLocationStep:
    def _planning_state(self, **kwargs) -> ConversationState:
        state = ConversationState(
            planning_mode=True,
            category_confirmed=True,
            category_tag="Essence",
            duration_days=3,
            current_planning_step="start_location",
            **kwargs,
        )
        state.mark_answered("duration")
        state.mark_answered("category")
        return state

    def test_seeduwa_completes_core_planning(self):
        state = self._planning_state()
        apply_planning_answer(state, "seeduwa")
        assert state.starting_location == "Seeduwa"
        assert state.district == "Gampaha"
        assert next_planning_question(state) is None

    def test_parse_planning_message_uses_expecting_step(self):
        parsed = parse_planning_message("seeduwa", expecting="start_location")
        assert parsed["starting_location"] == "Seeduwa"

        parsed_idle = parse_planning_message("Horana", expecting="travellers")
        assert parsed_idle["starting_location"] is None

    def test_extract_start_location_without_bare_flag(self):
        assert extract_start_location("seeduwa", allow_bare=True) == "Seeduwa"
        assert extract_start_location("seeduwa", allow_bare=False) == "Seeduwa"

    def test_planning_phrase_is_not_start_location(self):
        assert extract_start_location("Plan A Tour For Me", allow_bare=True) is None
        parsed = parse_planning_message("Plan A Tour For Me", expecting="start_location")
        assert parsed["starting_location"] is None

    def test_essence_trip_conversation(self):
        state, action, key = process_turn("__start_planning__:Essence", [], None)
        assert key == "duration"

        state, action, key = process_turn("3 days", [], state)
        assert state.duration_days == 3
        assert key == "start_location"

        state, action, key = process_turn("seeduwa", [], state)
        assert state.starting_location == "Seeduwa"
        assert state.category_tag == "Essence"
        assert state.duration_days == 3
        assert key != "travellers"


def _turn(svc, msg, history=None, state=None):
    result = svc.handle_message(msg, history=history or [], state=state)
    return result, ConversationState.from_dict(result["state"])


class TestHappyMoodFlow:
    def test_seeuwa_typo_builds_itinerary(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            steps = [
                ("im happy", None),
                ("12", None),
                ("seeuwa", "generate_itinerary"),
            ]
            for msg, expected_action in steps:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
                if expected_action:
                    assert result["action"] == expected_action, (
                        f"On {msg!r}: expected {expected_action}, got {result['action']}"
                    )
            assert state.starting_location == "Seeduwa"
            assert "Gregory Lake" in result["response"]
            assert "No attractions found" not in result["response"]
        finally:
            session.close()

    def test_unknown_start_corrected_to_seeduwa_retries_itinerary(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            for msg in ["im happy", "12", "Horana"]:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
            assert "couldn't build" in result["response"].lower() or "No attractions" in result["response"]
            result, state = _turn(svc, "seeduwa", history=history, state=state)
            assert result["action"] == "generate_itinerary"
            assert state.starting_location == "Seeduwa"
            assert "Gregory Lake" in result["response"]
        finally:
            session.close()

    def test_list_all_shows_happy_dataset_note(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            for msg in ["im happy", "12", "seeduwa"]:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
            result, state = _turn(svc, "list all", history=history, state=state)
            assert result["action"] == "recommend"
            assert "Gregory Lake" in result["response"]
            assert "every **Happy** mood place" in result["response"] or "1 total" in result["response"]
        finally:
            session.close()


class TestWildTripSeeduwaFlow:
    def test_wild_8_seeduwa_with_stale_planning_step(self):
        state = ConversationState(
            planning_mode=True,
            category_confirmed=True,
            category_tag="Wild",
            duration_days=8,
            current_planning_step="duration",
        )
        state.mark_answered("duration")
        state.mark_answered("category")
        apply_planning_answer(state, "seeduwa")
        assert state.starting_location == "Seeduwa"
        assert next_planning_question(state) is None

    def test_wild_trip_multi_turn(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            for msg in ["__start_planning__:Wild", "8", "seeduwa"]:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
                assert "Where will you be starting" not in result["response"] or msg == "8"
            assert state.starting_location == "Seeduwa"
            assert result["action"] in ("recommend", "generate_itinerary")
        finally:
            session.close()

    def test_wild_tab_suggests_places_after_start_location(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            for msg in ["__start_planning__:Wild", "3", "Seeduwa"]:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
            assert result["action"] == "recommend"
            assert "interested" not in result["response"].lower()
            assert state.category_tag == "Wild"
        finally:
            session.close()

    def test_wild_keyword_at_interests_step_is_wildlife_not_category(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            for msg in ["plan a 5 day trip", "seeuwa", "solo", "wild"]:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
            assert "wildlife" in state.interests
            assert result["intent"] != "change_category"
            assert result["action"] == "recommend"
            assert "relaxed, balanced, or packed" not in result["response"].lower()
        finally:
            session.close()

    def test_open_planning_wild_skips_pace_and_recommends(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            msgs = ["HI PLAN A TRIP", "5", "SEEDUWA", "SOLO", "WILD"]
            for msg in msgs:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
            assert state.mood_tag == "Explore"
            assert state.category_tag == "Wild"
            assert result["action"] == "recommend"
            assert "Which category" not in result["response"]
            assert "relaxed, balanced, or packed" not in result["response"].lower()
            assert "shortlist" in result["response"].lower() or " · " in result["response"]
        finally:
            session.close()

    def test_plan_5_day_adventure_flow(self):
        session = get_session_factory()()
        try:
            svc = ChatService(session)
            state = None
            history = []
            steps = [
                ("hi", "greeting"),
                ("plan a 5 day trip adventure", "clarify"),
                ("seeuwa", "recommend"),
            ]
            for msg, expected_action in steps:
                result, state = _turn(svc, msg, history=history, state=state)
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": result["response"]})
                assert result["action"] == expected_action, f"Failed on {msg!r}: {result['action']}"
            assert state.duration_days == 5
            assert state.starting_location is not None
            assert "adventure" in state.interests
            assert state.category_tag == "Thrills"
            assert result["action"] == "recommend"
            assert "relaxed, balanced, or packed" not in result["response"].lower()
        finally:
            session.close()

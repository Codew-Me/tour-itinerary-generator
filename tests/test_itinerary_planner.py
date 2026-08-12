"""Tests for structured itinerary planning pipeline."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.conversation_state import ConversationState
from src.services.geography import is_geographically_reasonable
from src.services.itinerary_planner import ItineraryPlanner
from src.services.itinerary_service import format_plan_deterministic, generate_itinerary


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestGeography:
    def test_colombo_not_reasonable_from_hambantota(self):
        assert is_geographically_reasonable("Hambantota", "Colombo") is False

    def test_hambantota_local_is_reasonable(self):
        assert is_geographically_reasonable("Hambantota", "Hambantota") is True

    def test_matara_neighbor_reasonable(self):
        assert is_geographically_reasonable("Hambantota", "Matara") is True


class TestItineraryPlanner:
    def _planner(self) -> ItineraryPlanner:
        session = get_session_factory()()
        return ItineraryPlanner(session)

    def test_hambantota_adventure_nature_no_colombo(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = ConversationState(
                tourism_intent=True,
                mood="adventurous",
                experience="nature",
                starting_location="Hambantota",
                district="Hambantota",
                duration_days=3,
                itinerary_requested=True,
            )
            plan = planner.build_plan(
                state,
                "Create a 3-day adventurous nature itinerary starting from Hambantota",
            )
            assert plan.validation_passed, plan.validation_errors
            assert len(plan.days) == 3

            all_stops = [stop for day in plan.days for stop in day.stops]
            assert len(all_stops) >= 3
            names = [s.name for s in all_stops]
            ids = [s.attraction_id for s in all_stops]

            # No duplicates
            assert len(ids) == len(set(ids))
            assert len(names) == len(set(n.lower() for n in names))

            # No Colombo detours
            for stop in all_stops:
                assert stop.district.lower() != "colombo", f"Colombo attraction included: {stop.name}"
                assert is_geographically_reasonable("Hambantota", stop.district)

            # Max 3 stops per day
            for day in plan.days:
                assert len(day.stops) <= 3

            text = format_plan_deterministic(plan)
            assert "Attidiya" not in text
            assert "Day 1" in text
            assert "Day 2" in text
            assert "Day 3" in text
        finally:
            session.close()

    def test_deterministic_format_includes_why_and_evidence(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = ConversationState(
                mood="adventurous",
                experience="nature",
                starting_location="Hambantota",
                duration_days=2,
            )
            plan = planner.build_plan(state, "2 day nature trip from Hambantota")
            text = format_plan_deterministic(plan, state)
            if plan.days:
                assert " · " in text
                assert "District:" not in text
                assert "Day 1" in text
                assert "Category:" not in text
                assert "Mood:" not in text
                assert "book in advance" not in text.lower()
        finally:
            session.close()

    def test_long_trip_fills_requested_days(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = ConversationState(
                tourism_intent=True,
                starting_location="Seeduwa",
                district="Gampaha",
                duration_days=20,
                travellers="solo",
                itinerary_requested=True,
            )
            plan = planner.build_plan(state, "20 day solo trip from Seeduwa")
            assert plan.duration_days == 20
            assert len(plan.days) >= 15, (
                f"Expected most of 20 days filled, got {len(plan.days)} days "
                f"with {sum(len(d.stops) for d in plan.days)} stops"
            )
            text = format_plan_deterministic(plan, state)
            assert "Day 1" in text
            assert "Day 10" in text or "Day 15" in text
        finally:
            session.close()

    def test_ten_day_trip_from_seeduwa_fills_all_days(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = ConversationState(
                tourism_intent=True,
                starting_location="Seeduwa",
                district="Gampaha",
                duration_days=10,
                travellers="solo",
                itinerary_requested=True,
            )
            plan = planner.build_plan(state, "10 day trip from Seeduwa")
            assert plan.duration_days == 10
            assert len(plan.days) == 10, (
                f"Expected 10 days, got {len(plan.days)} with "
                f"{sum(len(d.stops) for d in plan.days)} stops"
            )
            for day in plan.days:
                assert len(day.stops) >= 1, f"Day {day.day} has no stops"
            text = format_plan_deterministic(plan, state)
            assert "Day 10" in text
            assert "of your 10-day trip" not in text
        finally:
            session.close()

    def test_generate_itinerary_uses_planner(self):
        session = get_session_factory()()
        try:
            planner = ItineraryPlanner(session)
            state = ConversationState(
                mood="adventurous",
                experience="nature",
                starting_location="Hambantota",
                duration_days=3,
            )
            text = generate_itinerary(
                state,
                planner,
                user_message="3-day adventurous nature from Hambantota",
            )
            assert "Day 1" in text or "day 1" in text.lower()
            assert "Attidiya" not in text
        finally:
            session.close()

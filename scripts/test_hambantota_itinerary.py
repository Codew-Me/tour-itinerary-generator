"""Test Hambantota 3-day adventurous nature itinerary."""

from src.database.postgres import init_db, get_session_factory
from src.services.conversation_state import ConversationState
from src.services.itinerary_planner import ItineraryPlanner
from src.services.itinerary_service import format_plan_deterministic

init_db()
session = get_session_factory()()
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
    "Create a 3-day adventurous nature itinerary starting from Hambantota.",
)

print("VALIDATION:", "PASSED" if plan.validation_passed else "FAILED")
if plan.validation_errors:
    for e in plan.validation_errors:
        print(" ERROR:", e)

print("\n" + format_plan_deterministic(plan))

all_stops = [(d.day, s.name, s.district, s.attraction_id) for d in plan.days for s in d.stops]
print("\n--- STOPS ---")
for row in all_stops:
    print(row)

session.close()

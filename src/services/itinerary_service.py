"""Format validated itinerary plans into natural-language responses."""

from __future__ import annotations

from src.services.attraction_cards import format_attraction_card
from src.services.conversation_state import ConversationState
from src.services.itinerary_planner import ItineraryPlan, ItineraryPlanner


ITINERARY_FOLLOWUP = (
    "\n\n**How would you like to adjust it?**\n\n"
    "- Change the places\n"
    "- Make it more adventurous\n"
    "- Make it more relaxed\n"
    "- Add more places\n"
    "- Change the route"
)

ITINERARY_FOLLOWUP_CLARIFY = (
    "Sure 😊 Which would you like?\n\n"
    "• Make it more relaxed\n"
    "• Add more places\n"
    "• Focus on a specific experience"
)

RECOMMENDATION_FOLLOWUP_CLARIFY = (
    "Sure 😊 Would you like me to **build these into the itinerary** or **show you more options**?"
)


def format_plan_deterministic(plan: ItineraryPlan, state: ConversationState | None = None) -> str:
    """Deterministic formatter — user-facing cards without internal metadata."""
    cat = (state.category_tag if state else None) or (plan.experience or "").title()
    requested_days = plan.duration_days
    actual_days = len(plan.days)
    days_label = (
        f"{actual_days}-day (of your {requested_days}-day trip)"
        if actual_days and actual_days < requested_days
        else f"{requested_days}-day"
    )
    lines: list[str] = []

    built_from_suggestions = bool(
        state
        and state.last_recommendations
        and any("discussed in your previous recommendations" in a.lower() for a in plan.assumptions)
    )

    if built_from_suggestions and state and state.starting_location and cat:
        lines.append(
            f"Absolutely! I'll arrange those selected **{cat}** places into a **{days_label}** "
            f"itinerary from **{state.starting_location}**"
            f"{f', travelling as **{state.travellers}**' if state.travellers else ''}."
        )
    elif state and state.starting_location and cat:
        lines.append(
            f"Perfect! Based on your **{days_label}** trip from **{state.starting_location}**, "
            f"travelling as **{state.travellers or 'a group'}** "
            f"with a **{cat}** focus"
            f"{f' and **{state.mood_tag}** mood' if state.mood_tag else ''}, "
            f"here's a route I'd suggest:"
        )
    else:
        theme = cat or "Sri Lanka"
        lines.append(f"Here's a **{days_label} {theme}** itinerary based on what you've told me so far:")

    if plan.starting_location:
        lines.append(f"\n**Starting from:** {plan.starting_location}")

    if not plan.days:
        lines.append(
            "\nI couldn't build a valid itinerary from our dataset for this request. "
            "Try adjusting your starting location, category, or trip length."
        )
        if plan.validation_errors:
            for err in plan.validation_errors:
                lines.append(f"- {err}")
        return "\n".join(lines)

    trim_notes = [a for a in plan.assumptions if "optional stop" in a.lower() or "quite rushed" in a.lower()]
    route_notes = [a for a in plan.assumptions if "centroid" in a.lower() or "estimate" in a.lower()]

    if trim_notes:
        lines.append(f"\n{trim_notes[0]}")
    elif plan.starting_location and route_notes:
        lines.append(
            "\nI've ordered these stops from your starting point using district distance estimates."
        )
    elif plan.assumptions and any("discussed" in a.lower() for a in plan.assumptions):
        lines.append("\nThese are the places we shortlisted, arranged into day-by-day stops.")

    partial_notes = [
        a for a in plan.assumptions
        if "of" in a.lower() and "requested days" in a.lower()
    ]
    if partial_notes:
        lines.append(f"\n_{partial_notes[0]}_")

    for day in plan.days:
        primary = day.stops[0].district if day.stops else "—"
        lines.append("")
        lines.append(f"### Day {day.day} — {primary}")
        lines.append("")
        for stop in day.stops:
            card = format_attraction_card(
                {
                    "name": stop.name,
                    "destination": stop.destination,
                    "district": stop.district,
                    "details": stop.details_excerpt or stop.why,
                },
                include_review=False,
            )
            lines.append(card)
            lines.append("")

    lines.append(ITINERARY_FOLLOWUP.strip())
    return "\n".join(lines).strip()


def generate_itinerary(
    state: ConversationState,
    planner: ItineraryPlanner,
    user_message: str = "",
    history: list[dict] | None = None,
) -> str:
    """Build validated plan then format deterministically (no LLM hallucination risk)."""
    plan = planner.build_plan(state, user_message)
    if plan.days:
        state.current_itinerary = plan.to_dict()
        state.awaiting_itinerary_followup = True
        state.session_mode = "itinerary_review"
    else:
        state.current_itinerary = plan.to_dict()
        state.awaiting_itinerary_followup = False
        state.session_mode = "planning"
    return format_plan_deterministic(plan, state)

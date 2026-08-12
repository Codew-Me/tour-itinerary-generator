"""Conversation handlers for decline, rejection, and preference changes."""

from __future__ import annotations

from src.services.conversation_state import ConversationState
from src.services.planning_flow import apply_planning_answer, build_planning_question, next_planning_question
from src.services.planning_input import parse_planning_message


def mark_itinerary_rejected(state: ConversationState) -> None:
    """Record rejected attractions so the next generation avoids them."""
    if not state.current_itinerary:
        return
    rejected_ids: list[int] = []
    for day in state.current_itinerary.get("days", []):
        for stop in day.get("stops", []):
            aid = stop.get("attraction_id")
            if aid:
                rejected_ids.append(aid)
    state.rejected_attraction_ids = list(
        dict.fromkeys(state.rejected_attraction_ids + rejected_ids)
    )
    state.already_recommended_ids = list(
        dict.fromkeys(state.already_recommended_ids + rejected_ids)
    )
    state.rejected_itineraries.append(state.current_itinerary)
    state.rejected_itineraries = state.rejected_itineraries[-5:]


def handle_decline(state: ConversationState) -> str:
    state.awaiting_itinerary_followup = False
    state.itinerary_modify = False
    state.itinerary_requested = False
    if state.current_itinerary:
        state.session_mode = "itinerary_review"
        return (
            "No problem 😊 I'll keep the itinerary as it is. "
            "Is there anything else you'd like to change — the places, the route, or the overall experience?"
        )
    return "No problem 😊 What would you like to do next?"


def handle_reject_itinerary(state: ConversationState, message: str) -> tuple[str, bool]:
    """Return (response, should_revise_now)."""
    mark_itinerary_rejected(state)
    state.session_mode = "revision"
    state.awaiting_itinerary_followup = False
    state.itinerary_modify = False

    parsed = parse_planning_message(message)
    apply_planning_answer(state, message)

    if parsed.get("interests") or _contains_preference_change(message):
        state.itinerary_modify = True
        state.itinerary_modify_mode = "focus"
        return _build_preference_revision_ack(state), True

    state.awaiting_rejection_reason = True
    return (
        "No problem 😊 Let's change it. What isn't working for you — "
        "the **places**, the **route**, or the **type of experience**?\n\n"
        "You can also say things like:\n"
        "• More adventure\n"
        "• More relaxing\n"
        "• Different places\n"
        "• Change the route"
    ), False


def handle_change_preference(state: ConversationState, message: str) -> tuple[str, bool]:
    apply_planning_answer(state, message)
    state.awaiting_rejection_reason = False
    state.session_mode = "revision"
    state.itinerary_modify = True
    state.itinerary_modify_mode = "focus"
    return _build_preference_revision_ack(state), True


def handle_explicit_category_change(state: ConversationState, category: str) -> str:
    state.category_tag = category
    state.category_confirmed = True
    state.mark_answered("category")
    # Category selection defines the trip theme — skip redundant interests follow-up.
    state.mark_answered("interests")
    if state.current_itinerary:
        mark_itinerary_rejected(state)
        state.session_mode = "revision"
        state.itinerary_modify = True
        state.itinerary_modify_mode = "focus"
        return (
            f"Got it — I'll rebuild your trip with a **{category}** focus "
            f"while keeping your other preferences."
        )
    state.planning_mode = True
    state.session_mode = "planning"
    q = next_planning_question(state)
    if q:
        return f"Great choice! ⚡ **{category}** it is.\n\n{build_planning_question(q, state)}"
    return f"Perfect! ⚡ I'll plan a **{category}** trip for you."


def _build_preference_revision_ack(state: ConversationState) -> str:
    parts = []
    if state.mood_tag:
        parts.append(f"**{state.mood_tag}** mood")
    if state.interests:
        parts.append(f"**{', '.join(state.interests)}**-focused")
    duration = state.duration_days
    start = state.starting_location
    cat = state.category_tag
    if state.mood_tag and not state.interests:
        lines = [f"Got it — I'll rebuild your trip with a {parts[0]} from our attractions dataset."]
    else:
        lines = ["Got it — I'll revise the itinerary"]
        if parts:
            lines[0] += f" to match a {' · '.join(parts)} vibe"
        lines[0] += "."
    kept = []
    if duration:
        kept.append(f"**{duration} days**")
    if start:
        kept.append(f"starting from **{start}**")
    if cat:
        kept.append(f"**{cat}** category")
    if kept:
        lines.append(f"I'll keep {' · '.join(kept)} and find better-matching places from our dataset.")
    else:
        lines.append("I'll find better-matching places from our dataset.")
    return "\n\n".join(lines)


def _contains_preference_change(message: str) -> bool:
    lower = message.lower()
    markers = (
        "adventure", "adventurous", "relax", "wildlife", "nature", "beach", "heritage", "hiking",
        "curious", "happy", "excited", "peaceful", "spiritual", "i feel", "feeling",
    )
    return any(m in lower for m in markers)

"""Update conversation state and decide the next action deterministically."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from src.services.conversation_state import ConversationState
from src.services.geography import extract_starting_location
from src.services.planning_input import _is_open_planning_message
from src.services.planning_flow import (
    NEAR_DESTINATION_RE,
    acknowledge_category_rejection,
    apply_mood_from_message,
    apply_planning_answer,
    apply_starter_prompt_hints,
    build_planning_intro_for_recommend,
    build_planning_question,
    detect_explicit_category_selection,
    has_enough_for_build,
    is_planning_start_message,
    next_planning_question,
    planning_ready,
    resolve_planning_category,
    should_suggest_after_planning,
    start_category_planning,
    start_open_planning,
    wants_mood_planning,
    wants_replan_with_mood,
)
from src.services.preferences import (
    INTEREST_TO_DATASET,
    VALID_CATEGORIES,
    VALID_MOODS,
    normalize_category,
    normalize_mood,
)

AFFIRM = {"yes", "yeah", "yh", "yep", "sure", "ok", "okay", "fine", "y", "k", "yea", "correct", "right"}
NEG = {"no", "nope", "nah", "n"}
GREETINGS = {"hi", "hello", "hey", "hiya", "yo", "good morning", "good evening", "good afternoon"}
THANKS = {"thanks", "thank you", "thx", "ty"}
BYE = {"bye", "goodbye", "see you"}

DISTRICTS = (
    "galle", "colombo", "kandy", "matara", "negombo", "mirissa", "weligama",
    "ella", "badulla", "matale", "anuradhapura", "trincomalee", "jaffna", "sigiriya",
    "hambantota", "ampara", "ratnapura", "kalutara", "gampaha", "trincomalee",
)


class NextAction(str, Enum):
    CHAT = "chat"
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    ASK_DURATION = "ask_duration"
    GENERATE_ITINERARY = "generate_itinerary"


ClarifyKey = Literal[
    "mood", "experience", "offer_recommend", "duration",
    "start_location", "travellers", "interests", "pace", "category", "category_rejection",
    "itinerary_followup",
]

MORE_PHRASES = (
    "give me more", "show more", "more options", "more places",
    "another suggestion", "other options", "what else",
    "list all", "show all", "all places", "see all",
)
MODIFY_PHRASES = (
    "more relaxed", "make it relaxed", "make it more relaxed", "slow down", "less packed",
    "don't want temple", "dont want temple", "no temple", "no temples",
    "add more places", "add more", "more places",
    "focus more on", "focus on adventure", "focus on a specific",
    "add ", "remove ", "swap ",
)
ITINERARY_FOLLOWUP_OPTIONS = (
    "make this more relaxed",
    "add more places",
    "make it more adventurous",
    "make it more relaxed",
    "change the places",
    "change the route",
    "how would you like to adjust",
)
POPULAR_PHRASES = (
    "popular with visitors", "most popular", "highly rated", "best reviewed",
)
REJECT_CATEGORY_PHRASES = (
    "dont ask me plan based on category",
    "don't ask me plan based on category",
    "dont ask based on category",
    "don't ask based on category",
    "dont ask me based on category",
    "stop asking about category",
    "already chose the category",
    "already selected the category",
    "dont ask about category",
    "don't ask about category",
)
VAGUE_ANSWERS = (
    "idk", "i dk", "dunno", "don't know", "dont know", "not sure", "no idea",
    "whatever", "idk lan", "idk plan", "you tell me", "anything",
)

ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
}

BUILD_ITINERARY_PHRASES = (
    "build these",
    "build it",
    "build an itinerary",
    "build the itinerary",
    "build these into",
    "make an itinerary",
    "create an itinerary",
    "create the itinerary",
    "make the itinerary",
    "plan these",
    "plan these places",
    "turn these into",
    "use these places",
    "yes please build",
    "yes, build",
    "yes build",
)

ITINERARY_FOLLOWUP_PHRASES = BUILD_ITINERARY_PHRASES + (
    "choose some", "pick a few", "pick some", "select some", "use these",
    "use the", "plan with", "turn into", "choose and plan",
)

RECOMMENDATION_FOLLOWUP_OPTIONS = (
    "build these into a day-by-day itinerary",
    "show you more options",
)


def _is_plan_trip_message(text: str, state: ConversationState | None = None) -> bool:
    """True when user wants to finalize an itinerary from existing recommendations or legacy mood flow."""
    lower = text.lower()
    has_days = extract_days(text) is not None
    if not state:
        return False
    if state.last_recommendations:
        if has_days and re.search(r"\bplan\b", lower):
            return True
        if _contains_any(lower, ("itinerary", "day-by-day", "day plan", "make an itinerary", "create an itinerary")):
            return True
        if re.search(r"\bplan\b", lower) and re.search(r"\b(trip|tour|itinerary|days?)\b", lower):
            return True
    if state.current_itinerary and _contains_any(
        lower, ("itinerary", "day-by-day", "day plan", "make an itinerary", "regenerate")
    ):
        return True
    if state.mood and state.experience:
        if has_days and re.search(r"\bplan\b", lower):
            return True
        if _contains_any(lower, ("itinerary", "day-by-day", "day plan", "make an itinerary")):
            return True
    return False


def _assistant_showed_itinerary(last_assistant: str) -> bool:
    lower = last_assistant.lower()
    return (
        "### day" in lower
        or "day 1 —" in lower
        or "day 1 -" in lower
        or ("here's a" in lower and "itinerary" in lower)
        or ("here is a" in lower and "itinerary" in lower)
    )


def _is_itinerary_followup(message: str, state: ConversationState) -> bool:
    lower = _normalize(message)
    if state.last_recommendations and _contains_any(lower, ITINERARY_FOLLOWUP_PHRASES):
        return True
    if state.last_recommendations and extract_days(message) and re.search(r"\bplan\b", lower):
        return True
    if state.last_recommendations and _contains_any(lower, ("itinerary", "day trip")):
        return True
    if state.mood and state.experience and extract_days(message) and re.search(r"\bplan\b", lower):
        return True
    return _is_plan_trip_message(message, state)


def extract_number_selections(message: str, state: ConversationState) -> list[int] | None:
    """Map 'choose 1, 3 and 4' or 'first and third' to attraction IDs."""
    if not state.last_recommendations:
        return None
    lower = message.lower()

    choose_match = re.search(
        r"(?:choose|pick|select|use)\s+(.+?)(?:\s+and\s+plan|\s+to\s+plan|\s+for|\s*$)",
        lower,
    )
    segment = choose_match.group(1) if choose_match else lower
    segment = re.sub(r"\b\d+\s*(?:day|days|night|nights)\b", "", segment)

    selected: list[int] = []

    for match in re.finditer(r"\b(\d+)\b", segment):
        idx = int(match.group(1))
        if 1 <= idx <= len(state.last_recommendations):
            selected.append(state.last_recommendations[idx - 1]["id"])

    for word, idx in ORDINAL_MAP.items():
        if re.search(rf"\b{word}\b", segment) and idx <= len(state.last_recommendations):
            selected.append(state.last_recommendations[idx - 1]["id"])

    if not selected:
        return None
    return list(dict.fromkeys(selected))


def _last_assistant_message(history: list[dict]) -> str:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _is_itinerary_followup_prompt(last_assistant: str) -> bool:
    lower = last_assistant.lower()
    return any(opt in lower for opt in ITINERARY_FOLLOWUP_OPTIONS)


def _is_recommendation_followup_prompt(last_assistant: str) -> bool:
    lower = last_assistant.lower()
    return any(opt in lower for opt in RECOMMENDATION_FOLLOWUP_OPTIONS)


def _is_build_itinerary_message(message: str, state: ConversationState) -> bool:
    """True when user wants to turn the last recommendation set into an itinerary."""
    if not state.last_recommendations and not state.selected_attraction_ids:
        return False
    lower = _normalize(message)
    if _contains_any(lower, BUILD_ITINERARY_PHRASES):
        return True
    if _contains_any(lower, ("itinerary", "day-by-day", "day by day", "day trip")):
        if re.search(r"\b(build|make|create|plan|turn|use)\b", lower):
            return True
    return _is_itinerary_followup(message, state)


def _parse_itinerary_modify_mode(message: str) -> str | None:
    lower = _normalize(message)
    if _contains_any(lower, ("make it more relaxed", "more relaxed", "make it relaxed", "slow down", "less packed")):
        return "relaxed"
    if _contains_any(lower, ("add more places", "add more", "more places")):
        return "add"
    if _contains_any(lower, ("focus more on", "focus on adventure", "focus on a specific")):
        return "focus"
    return None


def extract_days(text: str) -> int | None:
    lower = text.lower()
    match = re.search(r"(\d+)\s*(?:day|days|night|nights)\b", lower)
    if match:
        days = int(match.group(1))
        return days if 1 <= days <= 30 else None
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "week": 7, "weekend": 2}
    for word, num in word_map.items():
        if re.search(rf"\b{word}\s+(?:day|days)\b", lower):
            return num
    return None


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip("!.?")


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(p in lower for p in phrases)


def _has_confirmed_category(state: ConversationState) -> bool:
    """True only when user explicitly confirmed a category (tab / plan message)."""
    return bool(state.category_confirmed)


def _has_category_preference(state: ConversationState) -> bool:
    return bool(state.category_confirmed or state.category_tag)


def _restore_planning_from_history(state: ConversationState, history: list[dict]) -> None:
    """Re-apply category planning from prior turns (metadata may be stale)."""
    if state.category_confirmed or state.planning_mode or state.duration_days is not None:
        return
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        cat = resolve_planning_category(content)
        if cat:
            start_category_planning(state, cat)
            return


def _apply_interest_tags(state: ConversationState, lower: str) -> None:
    """Map explicit category selection only — interests go through apply_planning_answer."""
    for cat in VALID_CATEGORIES:
        if re.search(rf"\b{re.escape(cat)}\b", lower, re.IGNORECASE):
            state.tourism_intent = True

    if not state.category_confirmed and not state.planning_mode:
        for mood in VALID_MOODS:
            if re.search(rf"\b{re.escape(mood)}\b", lower, re.IGNORECASE):
                state.mood_tag = mood
                state.mood = mood.lower()
                state.tourism_intent = True
                state.mark_answered("mood")


def _apply_message_signals(
    state: ConversationState,
    message: str,
    *,
    is_current: bool,
    last_assistant: str = "",
) -> ConversationState:
    """Extract mood, experience, requests, and duration from one message."""
    lower = _normalize(message)

    if _contains_any(lower, ("bored", "trip", "travel", "tour", "visit sri", "somewhere to go", "place to go", "plan a")):
        state.tourism_intent = True

    if _contains_any(lower, ("interested in", "i feel", "i'm feeling", "im feeling", "in the mood", "i love", "i like", "feeling")):
        state.tourism_intent = True

    _apply_interest_tags(state, lower)

    # Preference corrections
    if _contains_any(lower, ("actually", "instead", "rather", "change to", "prefer")):
        state.tourism_intent = True

    for loc in DISTRICTS:
        if re.search(rf"\b{loc}\b", lower):
            district_name = loc.title() if loc != "ella" else "Badulla"
            if NEAR_DESTINATION_RE.search(message):
                state.destination_district = district_name
            elif not state.destination_district:
                state.district = district_name
            state.tourism_intent = True

    start_loc = extract_starting_location(message)
    if start_loc:
        state.starting_location = start_loc
        if not state.destination_district:
            state.district = start_loc
        state.tourism_intent = True

    if re.search(r"\brelax(ing|ed)?\b", lower):
        if not state.category_confirmed:
            state.mood = "relaxing"
            state.tourism_intent = True
            state.mark_answered("mood")
    elif _contains_any(lower, ("adventur", "exciting")):
        if (
            not state.category_confirmed
            and not state.category_tag
            and not state.planning_mode
            and not _is_open_planning_message(message)
        ):
            state.tourism_intent = True
            if "adventure" not in state.interests:
                state.interests.append("adventure")
    elif _contains_any(lower, ("peaceful", "calm", "quiet")):
        if not state.category_confirmed:
            state.mood = "peaceful"
            state.tourism_intent = True
            state.mark_answered("mood")

    if re.search(r"\bbeach(es)?\b", lower):
        if not state.category_confirmed and not state.planning_mode:
            state.experience = "beach"
            state.tourism_intent = True
            state.mark_answered("experience")
    elif re.search(r"\bnature\b", lower) or "wildlife" in lower:
        if not state.category_confirmed and not state.planning_mode:
            state.experience = "nature"
            state.tourism_intent = True
            state.mark_answered("experience")
    elif re.search(r"\bscenic\b", lower) or "photography" in lower:
        if not state.category_confirmed and not state.planning_mode:
            state.experience = "scenic"
            state.tourism_intent = True
            state.mark_answered("experience")
    elif _contains_any(lower, ("heritage", "cultural", "culture", "history")):
        if not state.category_confirmed and not state.planning_mode:
            state.experience = "heritage"
            state.tourism_intent = True
            state.mark_answered("experience")

    if re.search(r"\bfamily\b", lower) and not state.travellers:
        state.travellers = "family"
        state.mark_answered("travellers")
        state.tourism_intent = True

    if is_current and lower in AFFIRM and last_assistant:
        if _contains_any(last_assistant, ("relax", "peaceful", "adventur", "vibe", "mood")):
            state.mood = "adventurous" if "adventur" in last_assistant else "relaxing"
            state.tourism_intent = True
            state.mark_answered("mood")
        if _contains_any(last_assistant, ("beach", "nature", "cultural", "heritage", "atmosphere")):
            if "beach" in last_assistant:
                state.experience = "beach"
                state.mark_answered("experience")
            elif "nature" in last_assistant:
                state.experience = "nature"
                state.mark_answered("experience")
        if _contains_any(last_assistant, ("suggest", "recommend", "places")):
            state.recommendation_requested = True
            state.sub_intent = "recommend"
        if _contains_any(last_assistant, ("how many days", "how long", "duration")):
            state.pending_action = "awaiting_duration"
        if _contains_any(last_assistant, ("itinerary", "generate", "create")) and state.duration_days:
            state.pending_action = "generate_itinerary"

    if is_current and lower in NEG and last_assistant:
        if _contains_any(last_assistant, ("mirissa", "negombo", "weligama", "specific")):
            state.recommendation_requested = True
            state.sub_intent = "recommend"
        elif "beach" in last_assistant and state.experience == "beach":
            state.experience = None
        state.mark_answered(state.last_question_key or "negation")

    if _contains_any(lower, ("suggest place", "suggest some", "recommend place", "recommendation")):
        state.recommendation_requested = True
        state.sub_intent = "recommend"
        state.tourism_intent = True
    if re.search(r"\bsuggest\b", lower) and re.search(r"\bplace", lower):
        state.recommendation_requested = True
        state.sub_intent = "recommend"
        state.tourism_intent = True
    if _contains_any(lower, MORE_PHRASES):
        state.wants_more_recommendations = True
        state.recommendation_requested = True
        state.sub_intent = "recommend"
        state.tourism_intent = True

    # Must-include place names (e.g. "add Sigiriya")
    add_match = re.search(r"\badd\s+([a-z][a-z\s'-]{2,40})", lower)
    if add_match:
        state.must_include.append(add_match.group(1).strip())
        state.itinerary_modify = True

    if _contains_any(lower, MODIFY_PHRASES):
        state.itinerary_modify = True
        mode = _parse_itinerary_modify_mode(message)
        if mode:
            state.itinerary_modify_mode = mode
        elif state.current_itinerary:
            state.itinerary_modify_mode = state.itinerary_modify_mode or "add"

    if is_current and lower in AFFIRM and last_assistant and _is_itinerary_followup_prompt(last_assistant):
        state.awaiting_itinerary_followup = True
        state.itinerary_modify = False
        state.itinerary_requested = False

    if _contains_any(lower, ("itinerary", "day plan", "day-by-day", "day by day")):
        state.itinerary_requested = True
        state.sub_intent = "itinerary"
        state.tourism_intent = True
    if extract_days(message) and re.search(r"\bplan\b", lower):
        state.tourism_intent = True
        days = extract_days(message)
        if not planning_ready(state):
            if not state.planning_mode:
                start_open_planning(state, duration_days=days)
            elif days is not None and state.duration_days is None:
                state.duration_days = days
                state.mark_answered("duration")
            state.itinerary_requested = False
        else:
            state.itinerary_requested = True
            state.sub_intent = "itinerary"
    if _contains_any(lower, ITINERARY_FOLLOWUP_PHRASES):
        state.itinerary_requested = True
        state.sub_intent = "itinerary"
        state.use_previous_recommendations = True
        state.tourism_intent = True
    if _is_build_itinerary_message(message, state):
        state.itinerary_requested = True
        state.sub_intent = "itinerary"
        state.use_previous_recommendations = True
        state.tourism_intent = True
    if "generate" in lower and _contains_any(lower, ("itinerary", "plan", "trip")):
        state.itinerary_requested = True
        state.sub_intent = "itinerary"
        state.tourism_intent = True
    if "make me" in lower and "itinerary" in lower:
        state.itinerary_requested = True
        state.sub_intent = "itinerary"
        state.tourism_intent = True

    if is_current:
        nums = extract_number_selections(message, state)
        if nums:
            state.selected_attraction_ids = nums
            state.use_previous_recommendations = True
            state.itinerary_requested = True
            state.sub_intent = "itinerary"
        days = extract_days(message)
        if days:
            state.duration_days = days
            state.mark_answered("duration")
            if state.itinerary_requested:
                state.pending_action = "generate_itinerary"

        if lower in {"ok", "okay"}:
            if state.itinerary_requested and state.duration_days:
                state.pending_action = "generate_itinerary"
            elif state.itinerary_requested and not state.duration_days:
                state.pending_action = "awaiting_duration"
            elif state.recommendation_requested or (state.mood and state.experience):
                state.pending_action = None

    return state


def update_state(state: ConversationState, message: str, history: list[dict] | None = None) -> ConversationState:
    """Merge the latest user message into structured state."""
    history = history or []
    last_assistant = _last_assistant_message(history).lower()

    if _contains_any(_normalize(message), REJECT_CATEGORY_PHRASES):
        state.reject_category_questions = True
        state.category_confirmed = True
        state.planning_mode = True
        state.mark_answered("category")
        state.mark_answered("experience")
        state.mark_answered("mood")
        state.mark_answered("offer_recommend")

    _restore_planning_from_history(state, history)

    for msg in history:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            state = _apply_message_signals(state, content, is_current=False)
            start_loc = extract_starting_location(content)
            if start_loc:
                state.starting_location = start_loc
                state.district = start_loc

    # Category tab / bare theme selection before parsing as a planning answer
    plan_cat = resolve_planning_category(message, state=state)
    if plan_cat:
        start_category_planning(state, plan_cat)

    state = _apply_message_signals(
        state,
        message,
        is_current=True,
        last_assistant=last_assistant,
    )

    apply_starter_prompt_hints(state, message)
    apply_mood_from_message(state, message)
    apply_planning_answer(state, message)

    # Category mention in natural language (not yet confirmed)
    if state.category_tag and not plan_cat and not state.category_confirmed:
        lower_msg = _normalize(message)
        if not _contains_any(lower_msg, ("suggest", "recommend", "show me places")):
            entering_planning = (
                state.planning_mode
                or _contains_any(lower_msg, ("plan", "travel", "itinerary", "trip"))
                or (not state.mood and not state.experience)
            )
            if entering_planning and not (
                state.last_recommendations and _is_itinerary_followup(message, state)
            ):
                state.planning_mode = True
                state.category_confirmed = True
                state.tourism_intent = True
                state.itinerary_requested = True
                state.mark_answered("category")
                state.mark_answered("experience")
                state.mark_answered("mood")
                state.mark_answered("offer_recommend")

    # Generic travel intent without category yet
    if _contains_any(_normalize(message), ("i want to travel", "plan my trip", "help me plan")):
        if not _has_confirmed_category(state):
            state.planning_mode = True
            state.tourism_intent = True
            state.itinerary_requested = True

    all_user_text = " ".join(
        m.get("content", "") for m in history if m.get("role") == "user"
    ).lower() + " " + _normalize(message)
    if _contains_any(all_user_text, ("relax", "beach", "bored", "suggest", "itinerary", "travel", "tour", "adventure", "plan")):
        state.tourism_intent = True

    return state


def decide_next_action(
    state: ConversationState,
    message: str,
    history: list[dict] | None = None,
) -> tuple[NextAction, ClarifyKey | None]:
    """Decide what to do after state update — deterministic, no repeat loops."""
    history = history or []
    lower = _normalize(message)
    last_assistant = _last_assistant_message(history).lower()

    # Category-tab / plan-category planning start
    if resolve_planning_category(message, state=state):
        q = next_planning_question(state)
        return NextAction.CLARIFY, q or "duration"

    # User rejected category re-questions — acknowledge and continue planning
    if _contains_any(lower, REJECT_CATEGORY_PHRASES):
        return NextAction.CLARIFY, "category_rejection"

    # Pure social
    if lower in GREETINGS and not state.tourism_intent:
        return NextAction.CHAT, None
    if lower in THANKS:
        return NextAction.CHAT, None
    if lower in BYE:
        return NextAction.CHAT, None

    if wants_mood_planning(lower) and not state.mood_tag and not state.has_answered("mood"):
        state.tourism_intent = True
        return NextAction.CLARIFY, "mood"

    # Mood-first planning — dataset mood_tag drives recommendations (skip legacy experience flow)
    if state.mood_tag and state.tourism_intent:
        if state.planning_mode and not planning_ready(state):
            q = next_planning_question(state)
            if q:
                return NextAction.CLARIFY, q
        if wants_replan_with_mood(_normalize(message)) and planning_ready(state):
            state.itinerary_requested = True
            state.use_previous_recommendations = False
            return NextAction.GENERATE_ITINERARY, None
        if planning_ready(state) and not state.last_recommendations:
            return NextAction.RECOMMEND, None

    # More recommendations (exclude already shown)
    if state.wants_more_recommendations or _contains_any(lower, MORE_PHRASES):
        state.wants_more_recommendations = True
        if _contains_any(lower, ("list all", "show all", "all places", "see all")):
            state.wants_list_all = True
        state.recommendation_requested = True
        return NextAction.RECOMMEND, None

    # Itinerary modification on existing plan
    if state.itinerary_modify and state.current_itinerary:
        state.itinerary_requested = True
        state.sub_intent = "itinerary"
        if not state.itinerary_modify_mode:
            state.itinerary_modify_mode = _parse_itinerary_modify_mode(message) or "add"
        return NextAction.GENERATE_ITINERARY, None

    # Ambiguous affirmation after itinerary follow-up offer
    if (
        lower in AFFIRM
        and state.current_itinerary
        and (_is_itinerary_followup_prompt(last_assistant) or state.awaiting_itinerary_followup)
    ):
        state.awaiting_itinerary_followup = True
        return NextAction.CLARIFY, "itinerary_followup"

    # Follow-up itinerary from saved recommendations (before progressive planning)
    if _is_build_itinerary_message(message, state) or _is_itinerary_followup(message, state):
        days = extract_days(message)
        if days:
            state.duration_days = days
        if state.duration_days and (planning_ready(state) or has_enough_for_build(state)):
            state.itinerary_requested = True
            state.sub_intent = "itinerary"
            if state.last_recommendations:
                state.use_previous_recommendations = True
            if planning_ready(state) or state.use_previous_recommendations or has_enough_for_build(state):
                return NextAction.GENERATE_ITINERARY, None
        if state.last_recommendations or (state.mood and state.experience):
            return NextAction.ASK_DURATION, "duration"

    # Progressive itinerary planning mode — category is a confirmed preference
    if state.planning_mode or state.category_confirmed:
        if _contains_any(lower, VAGUE_ANSWERS):
            q = next_planning_question(state) or state.last_question_key or "duration"
            return NextAction.CLARIFY, q  # type: ignore[return-value]
        if _contains_any(lower, POPULAR_PHRASES) or (
            re.search(r"\b(suggest|recommend|popular|which)\b", lower)
            and state.category_tag
            and planning_ready(state)
        ):
            return NextAction.RECOMMEND, None
        if _contains_any(lower, ("suggest", "recommend", "show me")) and planning_ready(state):
            return NextAction.RECOMMEND, None
        q = next_planning_question(state)
        if q:
            return NextAction.CLARIFY, q
        if should_suggest_after_planning(state):
            if not state.pace:
                state.pace = "balanced"
            return NextAction.RECOMMEND, None
        if planning_ready(state) and not state.current_itinerary:
            state.itinerary_requested = True
            state.sub_intent = "itinerary"
            return NextAction.GENERATE_ITINERARY, None
        if planning_ready(state) and state.current_itinerary and not state.itinerary_modify:
            if lower in AFFIRM and state.awaiting_itinerary_followup:
                return NextAction.CLARIFY, "itinerary_followup"
            return NextAction.CHAT, None

    # Itinerary path — require planning_ready before generating
    if (state.itinerary_requested or state.sub_intent == "itinerary") and not (
        state.category_confirmed and not planning_ready(state)
    ):
        if not planning_ready(state) and not state.use_previous_recommendations:
            state.planning_mode = True
            q = next_planning_question(state)
            if q:
                return NextAction.CLARIFY, q
            return NextAction.ASK_DURATION, "duration"
        if state.duration_days and (planning_ready(state) or state.use_previous_recommendations):
            return NextAction.GENERATE_ITINERARY, None
        if lower in AFFIRM and state.pending_action == "awaiting_duration":
            return NextAction.ASK_DURATION, "duration"
        if lower in {"ok", "okay"} and state.pending_action in ("awaiting_duration", "generate_itinerary"):
            if state.duration_days and (planning_ready(state) or state.use_previous_recommendations):
                return NextAction.GENERATE_ITINERARY, None
            return NextAction.ASK_DURATION, "duration"
        if extract_days(message) and (planning_ready(state) or state.use_previous_recommendations):
            return NextAction.GENERATE_ITINERARY, None
        return NextAction.ASK_DURATION, "duration"

    # Explicit recommendation request
    if state.recommendation_requested or (
        re.search(r"\bsuggest\b", lower) and re.search(r"\bplace", lower)
    ):
        return NextAction.RECOMMEND, None

    # Generic trip planning without category — legacy mood entry only
    if (
        (_contains_any(lower, ("plan a tour",)) or re.search(r"\bplan\s+a\s+trip\b", lower))
        and not state.mood
        and not _has_confirmed_category(state)
        and not resolve_planning_category(message, state=state)
    ):
        state.tourism_intent = True
        return NextAction.CLARIFY, "mood"

    # Legacy mood/experience flow — never when category or dataset mood is selected
    if (state.tourism_intent or lower in {"im bored", "i'm bored", "bored"}) and not (
        state.planning_mode or state.category_confirmed or state.reject_category_questions
        or _has_confirmed_category(state) or state.mood_tag
    ):
        if not state.mood and not state.has_answered("mood"):
            if lower in {"im bored", "i'm bored", "bored"}:
                return NextAction.CLARIFY, "mood"
            if lower in AFFIRM and _contains_any(last_assistant, ("relax", "vibe", "mood")):
                state.mood = state.mood or "relaxing"
                state.mark_answered("mood")
            elif re.search(r"\brelax", lower):
                state.mood = "relaxing"
                state.mark_answered("mood")
            else:
                return NextAction.CLARIFY, "mood"

        if state.mood and not state.experience and not state.has_answered("experience"):
            if re.search(r"\bbeach\b", lower):
                state.experience = "beach"
                state.mark_answered("experience")
            elif re.search(r"\bnature\b", lower):
                state.experience = "nature"
                state.mark_answered("experience")
            elif lower in AFFIRM and "beach" in last_assistant:
                state.experience = "beach"
                state.mark_answered("experience")
            else:
                return NextAction.CLARIFY, "experience"

        if state.mood and state.experience and not state.recommendation_requested:
            if _contains_any(lower, ("suggest", "recommend", "show me", "places")):
                return NextAction.RECOMMEND, None
            if not state.has_answered("offer_recommend"):
                return NextAction.CLARIFY, "offer_recommend"

        # Direct interest (Category / mood_tag) — skip when category confirmed
        if (state.category_tag or state.mood_tag) and not state.recommendation_requested:
            if state.category_confirmed or state.planning_mode:
                pass
            elif _contains_any(lower, ("suggest", "recommend", "show me", "places", "yes", "yh", "yeah", "sure")):
                return NextAction.RECOMMEND, None
            elif not state.has_answered("offer_recommend"):
                return NextAction.CLARIFY, "offer_recommend"

    # Default: friendly chat — but not if we have active tourism context
    if state.tourism_intent and (state.mood or state.last_recommendations):
        if state.last_recommendations and _is_plan_trip_message(message, state):
            state.duration_days = extract_days(message) or state.duration_days or 3
            state.use_previous_recommendations = True
            return NextAction.GENERATE_ITINERARY, None

    return NextAction.CHAT, None


def build_clarify_response(key: ClarifyKey, state: ConversationState) -> str:
    """One useful question — never ask to explain obvious terms."""
    if key == "category_rejection":
        return acknowledge_category_rejection(state)
    if key == "recommendation_followup":
        from src.services.itinerary_service import RECOMMENDATION_FOLLOWUP_CLARIFY
        return RECOMMENDATION_FOLLOWUP_CLARIFY
    if key == "itinerary_followup":
        from src.services.itinerary_service import ITINERARY_FOLLOWUP_CLARIFY
        return ITINERARY_FOLLOWUP_CLARIFY
    if state.reject_category_questions and key in ("experience", "offer_recommend", "mood", "category"):
        return acknowledge_category_rejection(state)
    if key in ("duration", "start_location", "travellers", "interests", "pace", "category"):
        return build_planning_question(key, state)
    if key == "mood":
        if state.category_confirmed and state.category_tag:
            q = next_planning_question(state) or "duration"
            return build_planning_question(q, state)
        if state.mood_tag:
            q = next_planning_question(state) or "duration"
            return build_planning_question(q, state)
        mood_list = ", ".join(VALID_MOODS)
        return (
            "Let's plan from your mood. Our attractions are tagged with moods like: "
            f"{mood_list}. Which one fits you today?"
        )
    if key == "experience":
        if state.mood_tag:
            q = next_planning_question(state) or "duration"
            return build_planning_question(q, state)
        if state.category_confirmed and state.category_tag:
            q = next_planning_question(state)
            if q:
                return build_planning_question(q, state)
        cats = ", ".join(VALID_CATEGORIES)
        mood_hint = state.mood_tag or state.mood or "great"
        return (
            f"Nice — a {mood_hint} vibe. Which category interests you? "
            f"{cats}"
        )
    if key == "offer_recommend":
        exp = state.experience or (state.category_tag or "travel").lower()
        mood = state.mood or state.mood_tag or "great"
        if state.category_tag and not state.experience:
            exp = state.category_tag.lower()
        return f"Sounds like a {mood} {exp} kind of trip. Want me to suggest some places?"
    if key == "duration":
        return "I can turn that into an itinerary. How many days are you planning?"
    return "How can I help with your Sri Lanka trip?"


def build_chat_response(message: str, state: ConversationState) -> str:
    lower = _normalize(message)
    if lower in GREETINGS:
        return "Hello. I'm your Sri Lanka travel assistant. How can I help you plan your trip today?"
    if lower in THANKS:
        return "You're welcome. Happy to help with your Sri Lanka travel plans anytime."
    if lower in BYE:
        return "Safe travels. Come back anytime you need destination ideas."
    if lower in {"im happy", "i'm happy", "happy"}:
        return (
            "Wonderful. I can suggest places tagged **Happy** in our dataset. "
            "How many days are you planning to travel?"
        )
    if state.tourism_intent and state.last_recommendations:
        return (
            "I still have your previous place suggestions in mind. "
            "Would you like me to turn them into an itinerary? Just say how many days."
        )
    if state.tourism_intent and state.mood:
        return f"Got it — a {state.mood} trip! What would you like to do next?"
    return "I'm here to help you explore Sri Lanka! Tell me what kind of trip you're dreaming of."


def process_turn(
    message: str,
    history: list[dict] | None,
    state: ConversationState | None,
) -> tuple[ConversationState, NextAction, ClarifyKey | None]:
    """Full turn: update state then decide action."""
    state = state or ConversationState()
    state = update_state(state, message, history)
    action, clarify_key = decide_next_action(state, message, history)
    return state, action, clarify_key

"""Intent detection for the conversational itinerary planning agent."""



from __future__ import annotations



import re

from enum import Enum



from src.services.conversation_state import ConversationState

from src.services.planning_flow import (

    detect_explicit_category_selection,

    has_enough_for_build,

    has_enough_for_suggest,

    has_valid_itinerary,

    next_planning_question,

    planning_ready,

    resolve_planning_category,

    should_suggest_after_planning,

    start_mood_planning,

    wants_mood_planning,

    wants_replan_with_mood,

)

from src.services.planning_input import parse_planning_message

from src.services.geography import is_start_location_correction

from src.services.state_manager import (

    AFFIRM,

    BYE,

    GREETINGS,

    MORE_PHRASES,

    NEG,

    THANKS,

    _assistant_showed_itinerary,

    _contains_any,

    _is_build_itinerary_message,

    _is_itinerary_followup,

    _is_itinerary_followup_prompt,

    _is_recommendation_followup_prompt,

    _last_assistant_message,

    _normalize,

    _parse_itinerary_modify_mode,

    extract_days,

)



OFF_TOPIC_PATTERNS = (

    "what is the capital of",

    "write code",

    "python script",

    "tell me a joke",

    "who is the president",

    "solve this equation",

    "weather in new york",

    "translate this",

    "how do i fix my",

)



REJECT_ITINERARY_PHRASES = (

    "don't like it",

    "dont like it",

    "don't like this",

    "dont like this",

    "don't like these",

    "dont like these",

    "don't like the",

    "dont like the",

    "i don't like",

    "i dont like",

    "not what i want",

    "not what i wanted",

    "this isn't for me",

    "this isnt for me",

    "isn't for me",

    "isnt for me",

    "something different",

    "want something different",

    "not for me",

    "hate it",

    "hate this",

    "don't want this",

    "dont want this",

)



CHANGE_PREFERENCE_PHRASES = (

    "need adventure",

    "want adventure",

    "more adventure",

    "more adventurous",

    "make it adventurous",

    "make it more adventurous",

    "more relaxing",

    "more relaxed",

    "make it relaxed",

    "want wildlife",

    "want nature",

    "want hiking",

    "want beaches",

    "focus on beach",

    "focus on heritage",

    "focus on wildlife",

)



SUGGEST_PHRASES = (

    "suggest",

    "recommend",

    "show me places",

    "popular with visitors",

)





class ConversationIntent(str, Enum):

    """Fine-grained conversational goals mapped onto agent actions."""



    SUGGEST_PLACES = "suggest_places"

    BUILD_ITINERARY = "build_itinerary"

    SHOW_MORE = "show_more"

    MODIFY_ITINERARY = "modify_itinerary"

    CHANGE_PLACES = "change_places"

    CHANGE_ROUTE = "change_route"

    MAKE_RELAXED = "make_relaxed"

    MAKE_ADVENTUROUS = "make_adventurous"

    ADD_PLACES = "add_places"

    CHANGE_PREFERENCE = "change_preference"

    GENERAL_QUESTION = "general_question"





class AgentIntent(str, Enum):

    GREETING = "greeting"

    ACKNOWLEDGE = "acknowledge"

    OFF_TOPIC = "off_topic"

    IDLE = "idle"

    START_ITINERARY = "start_itinerary"

    PROVIDE_INFORMATION = "provide_information"

    PLAN_COLLECT = "plan_collect"

    RECOMMEND = "recommend"

    GENERATE_ITINERARY = "generate_itinerary"

    MODIFY_ITINERARY = "modify_itinerary"

    REVISE_ITINERARY = "revise_itinerary"

    REJECT_ITINERARY = "reject_itinerary"

    DECLINE = "decline"

    CHANGE_PREFERENCE = "change_preference"

    CHANGE_CATEGORY = "change_category"

    REQUEST_MORE_OPTIONS = "request_more_options"

    CLARIFY = "clarify"

    ACCEPT = "accept"





def detect_conversation_intent(

    message: str,

    state: ConversationState,

    history: list[dict] | None = None,

) -> ConversationIntent | None:

    """Detect the user's conversational goal within an active trip-planning session."""

    history = history or []

    lower = _normalize(message)



    if state.wants_more_recommendations or _contains_any(lower, MORE_PHRASES):

        return ConversationIntent.SHOW_MORE



    if _is_suggest_places_message(message, state) and not _requests_itinerary_without_duration(message, state):

        return ConversationIntent.SUGGEST_PLACES



    if _is_build_itinerary_message(message, state) and has_enough_for_build(state):

        return ConversationIntent.BUILD_ITINERARY



    if _parse_itinerary_modify_mode(message) == "relaxed" or state.itinerary_modify_mode == "relaxed":

        return ConversationIntent.MAKE_RELAXED

    if _parse_itinerary_modify_mode(message) == "add":

        return ConversationIntent.ADD_PLACES

    if _parse_itinerary_modify_mode(message) == "focus":

        return ConversationIntent.CHANGE_PREFERENCE



    if state.current_itinerary and _contains_any(lower, ("change the places", "different places", "swap")):

        return ConversationIntent.CHANGE_PLACES

    if state.current_itinerary and _contains_any(lower, ("change the route", "different route", "reorder")):

        return ConversationIntent.CHANGE_ROUTE

    if state.current_itinerary and _contains_any(lower, ("more adventurous", "make it adventurous")):

        return ConversationIntent.MAKE_ADVENTUROUS



    if _parse_itinerary_modify_mode(message) or state.itinerary_modify:

        return ConversationIntent.MODIFY_ITINERARY



    if lower in AFFIRM and _is_recommendation_followup_prompt(_last_assistant_message(history).lower()) and state.last_recommendations:

        return None



    if _is_change_preference(message, state, parse_planning_message(message)):

        return ConversationIntent.CHANGE_PREFERENCE



    return None





def detect_agent_intent(

    message: str,

    state: ConversationState,

    history: list[dict] | None = None,

) -> AgentIntent:

    """Classify user goal before choosing an action."""

    history = history or []

    lower = _normalize(message)

    last_assistant = _last_assistant_message(history).lower()

    parsed = parse_planning_message(message)

    conversation_intent = detect_conversation_intent(message, state, history)



    if lower in GREETINGS and not _has_active_session(state):

        return AgentIntent.GREETING



    if lower in THANKS or lower in BYE:

        return AgentIntent.ACKNOWLEDGE



    if _is_clearly_off_topic(lower):

        return AgentIntent.OFF_TOPIC



    if _is_decline(lower, state, last_assistant):

        return AgentIntent.DECLINE



    if _is_reject_itinerary(lower) and (

        state.current_itinerary or _assistant_showed_itinerary(last_assistant)

    ):

        return AgentIntent.REJECT_ITINERARY



    explicit_cat = detect_explicit_category_selection(message, state)

    if explicit_cat and (state.current_itinerary or state.planning_mode or state.session_mode != "idle"):
        from src.services.planning_flow import _expecting_planning_answer
        from src.services.planning_input import extract_interests

        expecting = _expecting_planning_answer(state)
        if state.last_question_key == "interests" and extract_interests(message, expecting="interests"):
            pass
        elif not (expecting == "interests" and parse_planning_message(message, expecting=expecting).get("interests")):
            return AgentIntent.CHANGE_CATEGORY



    if resolve_planning_category(message, state=state):

        return AgentIntent.START_ITINERARY



    if conversation_intent in (

        ConversationIntent.MODIFY_ITINERARY,

        ConversationIntent.CHANGE_PLACES,

        ConversationIntent.CHANGE_ROUTE,

        ConversationIntent.MAKE_RELAXED,

        ConversationIntent.MAKE_ADVENTUROUS,

        ConversationIntent.ADD_PLACES,

    ) or _parse_itinerary_modify_mode(message) or state.itinerary_modify:

        return AgentIntent.MODIFY_ITINERARY



    if conversation_intent == ConversationIntent.CHANGE_PREFERENCE or _is_change_preference(message, state, parsed):

        return AgentIntent.CHANGE_PREFERENCE



    if wants_replan_with_mood(message) and state.mood_tag:

        if planning_ready(state):

            return AgentIntent.GENERATE_ITINERARY

        return AgentIntent.PLAN_COLLECT



    if (
        is_start_location_correction(message, state.starting_location)
        and planning_ready(state)
        and not has_valid_itinerary(state)
    ):

        return AgentIntent.GENERATE_ITINERARY



    if (

        lower in AFFIRM

        and state.current_itinerary

        and (_is_itinerary_followup_prompt(last_assistant) or state.awaiting_itinerary_followup)

    ):

        return AgentIntent.CLARIFY



    if (

        lower in AFFIRM

        and state.last_recommendations

        and _is_recommendation_followup_prompt(last_assistant)

    ):

        return AgentIntent.CLARIFY



    if conversation_intent == ConversationIntent.SHOW_MORE:

        return AgentIntent.REQUEST_MORE_OPTIONS



    if conversation_intent == ConversationIntent.BUILD_ITINERARY:

        return AgentIntent.GENERATE_ITINERARY



    if conversation_intent == ConversationIntent.SUGGEST_PLACES:

        return AgentIntent.RECOMMEND



    if _requests_itinerary_without_duration(message, state):

        return AgentIntent.START_ITINERARY



    if _contains_any(lower, SUGGEST_PHRASES):

        if has_enough_for_suggest(state) or state.tourism_intent or state.planning_mode or state.last_recommendations:

            return AgentIntent.RECOMMEND



    if _is_build_itinerary_message(message, state) and has_enough_for_build(state):

        return AgentIntent.GENERATE_ITINERARY



    if _is_itinerary_followup(message, state):

        if has_enough_for_build(state):

            return AgentIntent.GENERATE_ITINERARY

        if planning_ready(state):

            return AgentIntent.GENERATE_ITINERARY

        if _is_start_itinerary_message(message, state):

            return AgentIntent.START_ITINERARY



    if _is_start_itinerary_message(message, state) and not _is_itinerary_followup(message, state):

        return AgentIntent.START_ITINERARY



    if state.planning_mode or state.session_mode == "planning":

        if should_suggest_after_planning(state):

            return AgentIntent.RECOMMEND

        if planning_ready(state) and not has_valid_itinerary(state):

            return AgentIntent.GENERATE_ITINERARY

        if _message_provides_planning_info(parsed, state):

            return AgentIntent.PROVIDE_INFORMATION

        if next_planning_question(state):

            return AgentIntent.PLAN_COLLECT



    if state.session_mode in ("itinerary_review", "revision") and state.current_itinerary:

        if _contains_any(lower, ("change", "update", "modify", "swap", "remove", "add", "different")):

            return AgentIntent.MODIFY_ITINERARY

        if _message_provides_planning_info(parsed, state):

            return AgentIntent.CHANGE_PREFERENCE



    if _looks_off_topic(message, state):

        return AgentIntent.OFF_TOPIC



    if wants_mood_planning(lower) and not state.mood_tag:

        return AgentIntent.PLAN_COLLECT



    if state.mood_tag and state.tourism_intent:

        if not state.planning_mode:

            start_mood_planning(state)

        if next_planning_question(state):

            return AgentIntent.PLAN_COLLECT

        if has_enough_for_suggest(state) or planning_ready(state):

            return AgentIntent.RECOMMEND



    return AgentIntent.IDLE





def resolve_agent_phase(intent: AgentIntent, state: ConversationState) -> str:

    if intent in (AgentIntent.START_ITINERARY, AgentIntent.PLAN_COLLECT, AgentIntent.PROVIDE_INFORMATION):

        return "planning"

    if intent in (AgentIntent.RECOMMEND, AgentIntent.REQUEST_MORE_OPTIONS):

        return "recommending"

    if intent == AgentIntent.GENERATE_ITINERARY:

        return "itinerary"

    if intent in (

        AgentIntent.MODIFY_ITINERARY,

        AgentIntent.REVISE_ITINERARY,

        AgentIntent.REJECT_ITINERARY,

        AgentIntent.CHANGE_PREFERENCE,

        AgentIntent.CHANGE_CATEGORY,

        AgentIntent.CLARIFY,

        AgentIntent.DECLINE,

    ):

        if state.current_itinerary or state.session_mode in ("itinerary_review", "revision"):

            return "revision"

    if state.session_mode == "itinerary_review":

        return "itinerary"

    if state.planning_mode:

        return "planning"

    return state.session_mode if state.session_mode != "idle" else "idle"





def _requests_itinerary_without_duration(message: str, state: ConversationState) -> bool:

    lower = _normalize(message)

    wants_itinerary = _contains_any(lower, ("itinerary", "generate")) and re.search(

        r"\b(plan|generate|itinerary|ready to plan)\b", lower

    )

    if not wants_itinerary:

        return False

    if state.duration_days:

        return False

    return bool(state.itinerary_requested or state.last_recommendations or state.tourism_intent)





def _is_suggest_places_message(message: str, state: ConversationState) -> bool:

    lower = _normalize(message)

    if not _contains_any(lower, SUGGEST_PHRASES):

        return False

    if re.search(r"\bplace", lower):

        return True

    if lower.strip() in {"suggest", "recommend"}:

        return has_enough_for_suggest(state)

    return has_enough_for_suggest(state) or bool(state.last_recommendations)





def _is_decline(lower: str, state: ConversationState, last_assistant: str) -> bool:

    if lower not in NEG:

        return False

    if state.current_itinerary:

        return True

    if _is_itinerary_followup_prompt(last_assistant) or state.awaiting_itinerary_followup:

        return True

    if _is_recommendation_followup_prompt(last_assistant):

        return True

    if state.awaiting_rejection_reason:

        return True

    return False





def _is_reject_itinerary(lower: str) -> bool:

    return _contains_any(lower, REJECT_ITINERARY_PHRASES)





def _is_change_preference(message: str, state: ConversationState, parsed: dict) -> bool:

    if not (

        state.current_itinerary

        or state.session_mode in ("itinerary_review", "revision")

        or state.awaiting_rejection_reason

    ):

        return False

    lower = _normalize(message)

    if wants_replan_with_mood(message):

        return True

    if _contains_any(lower, CHANGE_PREFERENCE_PHRASES):

        return True

    if parsed.get("interests"):

        return True

    return False





def _is_start_itinerary_message(message: str, state: ConversationState) -> bool:

    lower = message.lower()

    has_plan = _contains_any(lower, ("plan", "itinerary", "trip", "tour", "ready to plan"))

    has_days = extract_days(message) is not None

    if has_plan and (has_days or "tour" in lower or "trip" in lower or "ready to plan" in lower):

        return not planning_ready(state)

    return False





def _message_provides_planning_info(parsed: dict, state: ConversationState) -> bool:

    if parsed.get("duration_days") is not None and state.duration_days is None:

        return True

    if parsed.get("starting_location") and not state.starting_location:

        return True

    if parsed.get("travellers") and not state.travellers:

        return True

    if parsed.get("interests") and not state.interests:

        return True

    if parsed.get("pace") and not state.pace:

        return True

    return bool(

        parsed.get("duration_days")

        or parsed.get("starting_location")

        or parsed.get("travellers")

        or parsed.get("interests")

        or parsed.get("pace")

    )





def _has_active_session(state: ConversationState) -> bool:

    return bool(

        state.planning_mode

        or state.category_confirmed

        or state.current_itinerary

        or state.last_recommendations

        or state.tourism_intent

        or state.session_mode != "idle"

    )





def _is_clearly_off_topic(lower: str) -> bool:

    if _contains_any(lower, OFF_TOPIC_PATTERNS):

        return True

    if re.search(r"\b\d+\s*[+\-*/]\s*\d+\b", lower):

        return True

    return False





def _looks_off_topic(message: str, state: ConversationState) -> bool:

    if _has_active_session(state):

        return False

    lower = message.lower()

    return _is_clearly_off_topic(lower)



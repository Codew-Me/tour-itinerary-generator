"""Progressive itinerary-planning conversation flow."""

from __future__ import annotations

import re

from src.services.conversation_state import ConversationState
from src.services.geography import resolve_locality
from src.services.planning_input import parse_planning_message
from src.services.preferences import VALID_CATEGORIES, VALID_MOODS

PLANNING_START_PREFIX = "__start_planning__:"
STARTER_PREFIX = "__starter__:"
PLANNING_STEPS = ("duration", "start_location", "travellers", "interests", "pace", "category")

STARTER_THEME_KEYWORDS: tuple[tuple[re.Pattern[str], str, str | None], ...] = (
    (re.compile(r"\badventur", re.I), "adventure", "Thrills"),
    (re.compile(r"\bwildlife\b", re.I), "wildlife", "Wild"),
    (re.compile(r"\bheritage\b|\bcultural\b|\bhistory\b", re.I), "heritage", "Heritage"),
    (re.compile(r"\bbeach\b|\bcoast\b|\bgetaway\b", re.I), "beach", "Pristine"),
    (re.compile(r"\bscenic\b|\bwaterfall", re.I), "photography", "Scenic"),
    (re.compile(r"\bnature\b", re.I), "nature", "Wild"),
    (re.compile(r"\bfamily[\s-]friendly\b", re.I), "family", "Essence"),
)

NEAR_DESTINATION_RE = re.compile(
    r"\bnear\s+([a-zA-Z\s\-']+?)(?:\.|,|$|\s+for|\s+from|\s+with|\s+starting)",
    re.I,
)
FROM_START_RE = re.compile(
    r"\bfrom\s+([a-zA-Z\s\-']+?)(?:\.|,|$|\s+for|\s+near|\s+with|\s+starting)",
    re.I,
)

HAPPY_PHRASES = (
    "im happy",
    "i'm happy",
    "i am happy",
    "feeling happy",
    "im feeling happy",
    "i'm feeling happy",
    "i feel happy",
)

MOOD_PLANNING_PHRASES = (
    "plan based on mood",
    "based on mood",
    "plan on mood",
    "plan by mood",
    "mood based",
    "according to mood",
    "from my mood",
)

REPLAN_ITINERARY_RE = re.compile(
    r"\b(?:plan|make|build|create|give me|can u|can you)\b.{0,30}\b(?:another|new|different)\b.{0,20}\b(?:itinerary|plan|trip)\b"
    r"|\b(?:another|new|different)\s+(?:itinerary|plan|trip)\b",
    re.I,
)

EXPLICIT_CATEGORY_PHRASES = (
    "want {cat}",
    "change to {cat}",
    "change the category to {cat}",
    "make it a {cat}",
    "make it {cat}",
    "{cat} trip",
    "{cat} category",
    "focus on {cat}",
    "i want {cat}",
    "choose {cat}",
    "pick {cat}",
    "go with {cat}",
)


def is_planning_start_message(message: str) -> str | None:
    if message.startswith(PLANNING_START_PREFIX):
        return message.split(":", 1)[1].strip()
    return None


def detect_plan_category_message(message: str) -> str | None:
    lower = message.lower()
    if not re.search(r"\bplan\b", lower):
        return None
    for cat in VALID_CATEGORIES:
        if re.search(rf"\b{re.escape(cat.lower())}\b", lower):
            return cat
    return None


def resolve_planning_category(message: str, state: ConversationState | None = None) -> str | None:
    started = is_planning_start_message(message)
    if started:
        return started
    planned = detect_plan_category_message(message)
    if planned:
        return planned
    lower = message.strip().lower().rstrip("!.?")
    for cat in VALID_CATEGORIES:
        cat_lower = cat.lower()
        if lower in (cat_lower, f"{cat_lower} trip", f"{cat_lower} focus"):
            if state and _should_treat_bare_category_as_interest(message, state):
                return None
            return cat
    return None


def _should_treat_bare_category_as_interest(message: str, state: ConversationState) -> bool:
    """Bare 'wild' is a wildlife interest during Q&A — not a fresh Wild category selection."""
    lower = message.strip().lower().rstrip("!.?")
    if lower not in ("wild", "wild trip", "wild focus"):
        return False
    from src.services.planning_input import extract_interests

    if not extract_interests(message, expecting="interests"):
        return False
    if state.planning_mode or state.category_confirmed:
        return True
    if state.last_question_key == "interests":
        return True
    return False


def _planning_qa_blocks_bare_category(state: ConversationState) -> bool:
    """True when the user is likely answering a planning question, not picking a category."""
    collecting = next_planning_question(state)
    if collecting in ("duration", "start_location", "travellers", "interests", "pace"):
        return True
    # Use last_question_key even if apply_planning_answer already marked the field this turn.
    if state.last_question_key in ("interests", "pace"):
        return True
    expecting = _expecting_planning_answer(state)
    return expecting in ("interests", "pace")


def detect_explicit_category_selection(
    message: str,
    state: ConversationState | None = None,
) -> str | None:
    """Only match when user explicitly chooses a dataset category."""
    if is_planning_start_message(message):
        return is_planning_start_message(message)
    lower = message.lower().strip()

    # During progressive Q&A, bare category words are experience interests — not re-selection.
    if state and _planning_qa_blocks_bare_category(state):
        return _explicit_category_from_phrases(lower)

    for cat in VALID_CATEGORIES:
        cat_lower = cat.lower()
        if lower.rstrip("!.?") == cat_lower:
            if state and _should_treat_bare_category_as_interest(message, state):
                return None
            return cat
        for template in EXPLICIT_CATEGORY_PHRASES:
            phrase = template.format(cat=cat_lower)
            if phrase in lower:
                return cat
        if re.search(rf"\b{re.escape(cat_lower)}\b", lower) and re.search(
            r"\b(want|choose|pick|change|category|make it|go with|focus on)\b", lower
        ):
            return cat
    return None


def _explicit_category_from_phrases(lower: str) -> str | None:
    """Category only when user explicitly names it — not a bare keyword answer."""
    for cat in VALID_CATEGORIES:
        cat_lower = cat.lower()
        for template in EXPLICIT_CATEGORY_PHRASES:
            phrase = template.format(cat=cat_lower)
            if phrase in lower:
                return cat
        if re.search(rf"\b{re.escape(cat_lower)}\b", lower) and re.search(
            r"\b(want|choose|pick|change|category|make it|go with|focus on)\b", lower
        ):
            return cat
    return None


def start_open_planning(state: ConversationState, duration_days: int | None = None) -> None:
    """Begin planning without a pre-selected category tab."""
    state.planning_mode = True
    state.session_mode = "planning"
    state.tourism_intent = True
    state.sub_intent = "itinerary"
    state.current_planning_step = "duration"
    if duration_days is not None:
        state.duration_days = duration_days
        state.mark_answered("duration")
        state.current_planning_step = next_planning_question(state)


def start_category_planning(state: ConversationState, category: str) -> None:
    state.planning_mode = True
    state.session_mode = "planning"
    state.category_confirmed = True
    state.tourism_intent = True
    state.category_tag = category
    state.sub_intent = "itinerary"
    state.itinerary_requested = True
    state.current_planning_step = "duration"
    state.mark_answered("category")
    state.mark_answered("experience")
    state.mark_answered("mood")
    state.mark_answered("offer_recommend")
    # Category tab already defines the trip theme — interests question is redundant.
    state.mark_answered("interests")


def wants_mood_planning(message: str) -> bool:
    lower = message.lower().strip()
    if any(p in lower for p in MOOD_PLANNING_PHRASES):
        return True
    return bool(re.search(r"\bplan\b.{0,40}\bmood\b|\bmood\b.{0,40}\bplan\b", lower))


def wants_replan_with_mood(message: str) -> bool:
    """User wants a fresh plan driven by mood — often during itinerary review."""
    from src.services.planning_input import normalize_mood_feel_phrases

    text = normalize_mood_feel_phrases(message)
    lower = text.lower().strip()
    if wants_mood_planning(text):
        return True
    if REPLAN_ITINERARY_RE.search(lower):
        return True
    if re.search(r"\b(i feel|i'm feeling|im feeling|feeling)\b", lower):
        return True
    return False


def prepare_mood_replan(state: ConversationState) -> None:
    """Reset itinerary artifacts so the next plan uses the updated mood_tag."""
    from src.services.agent_handlers import mark_itinerary_rejected

    if state.current_itinerary:
        mark_itinerary_rejected(state)
    state.current_itinerary = None
    state.last_recommendations = []
    state.itinerary_requested = True
    state.itinerary_modify = False
    state.itinerary_modify_mode = None
    state.awaiting_itinerary_followup = False
    state.wants_more_recommendations = False
    state.session_mode = "revision"
    state.tourism_intent = True
    state.sub_intent = "itinerary"


def start_mood_planning(state: ConversationState) -> None:
    """Begin trip planning filtered by dataset mood_tag (attractions.xlsx)."""
    state.planning_mode = True
    state.session_mode = "planning"
    state.mood_confirmed = True
    state.tourism_intent = True
    state.sub_intent = "itinerary"
    state.itinerary_requested = True
    state.mark_answered("mood")
    state.mark_answered("interests")
    state.current_planning_step = next_planning_question(state)


def apply_mood_from_message(state: ConversationState, message: str) -> bool:
    """Map conversational mood to dataset mood_tag and enter or refresh mood-first planning."""
    from src.services.planning_input import normalize_mood_feel_phrases
    from src.services.preferences import MOOD_ALIASES, normalize_mood

    message = normalize_mood_feel_phrases(message)
    lower = message.lower().strip()
    trimmed = lower.rstrip("!.?")

    if wants_mood_planning(message):
        state.tourism_intent = True

    happy_match = trimmed == "happy" or any(p in lower for p in HAPPY_PHRASES)
    mood_context = happy_match or bool(
        re.search(r"\b(i feel|i'm feeling|im feeling|in the mood|feeling)\b", lower)
        or re.search(r"\bmood\b", lower)
        or wants_mood_planning(message)
    )

    replan_context = wants_replan_with_mood(message) or bool(
        mood_context
        and (state.current_itinerary or state.session_mode in ("itinerary_review", "revision"))
    )

    if (state.current_itinerary or state.itinerary_modify) and not replan_context:
        return False
    if state.category_confirmed and state.planning_mode and not replan_context:
        return False

    previous_mood = state.mood_tag
    mood_set = False
    if happy_match:
        state.mood_tag = "Happy"
        state.mood = "happy"
        mood_set = True
    elif mood_context:
        for alias, mood in MOOD_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                tag = normalize_mood(mood)
                if tag and tag in VALID_MOODS:
                    state.mood_tag = tag
                    state.mood = alias
                    mood_set = True
                    break
        if not mood_set:
            for mood in VALID_MOODS:
                if re.search(rf"\b{re.escape(mood)}\b", message, re.IGNORECASE):
                    state.mood_tag = mood
                    state.mood = mood.lower()
                    mood_set = True
                    break

    if mood_set:
        state.tourism_intent = True
        state.mood_confirmed = True
        state.mark_answered("mood")
        state.mark_answered("interests")
        if replan_context and (previous_mood != state.mood_tag or state.current_itinerary):
            prepare_mood_replan(state)
        elif not state.planning_mode:
            start_mood_planning(state)
        return True
    return False


def _looks_like_starter_prompt(message: str) -> bool:
    lower = message.lower().strip()
    if lower.startswith(PLANNING_START_PREFIX) or lower.startswith(STARTER_PREFIX):
        return True
    if len(lower) < 12:
        return False
    if re.search(r"\b(suggest|show|recommend)\s+(me\s+)?(places|spots|destinations)\b", lower):
        return False
    theme_signals = (
        "adventure", "adventur", "wildlife", "heritage", "beach", "scenic", "nature",
        "family-friendly", "getaway", "cultural", "history", "wild ", " wild",
    )
    trip_signals = (
        "plan a", "plan my", "day trip", "day trips", "weekend", " trip", " tour",
        " travel", "near ", "from ", "places near",
    )
    has_theme = any(s in lower for s in theme_signals)
    has_trip = (
        any(s in lower for s in trip_signals)
        or bool(re.search(r"\b\d[\s-]*(day|days)\b", lower))
    )
    return has_theme and has_trip


def apply_starter_prompt_hints(state: ConversationState, message: str) -> None:
    """Pre-fill theme, district, and duration from welcome chips and rich openers."""
    raw = message.strip()
    lower = raw.lower()

    if lower.startswith(STARTER_PREFIX):
        payload = raw.split(":", 1)[1].strip()
        category_hint, _, display = payload.partition("|")
        if category_hint in VALID_CATEGORIES:
            start_category_planning(state, category_hint)
        message = display or message

    if not _looks_like_starter_prompt(message):
        return

    from src.services.planning_input import extract_duration, extract_interests, extract_travellers

    interests: list[str] = []
    forced_category: str | None = None
    for pattern, interest, category in STARTER_THEME_KEYWORDS:
        if pattern.search(message):
            interests.append(interest)
            if category:
                forced_category = category
            break

    for interest in extract_interests(message, expecting="interests"):
        if interest not in interests:
            interests.append(interest)

    if interests and not state.has_answered("interests"):
        state.interests = list(dict.fromkeys(state.interests + interests))
        state.mark_answered("interests")
        apply_dataset_defaults_after_interests(state)

    if forced_category and not state.category_confirmed:
        state.category_tag = forced_category
        state.category_confirmed = True
        state.mark_answered("category")

    near_match = NEAR_DESTINATION_RE.search(message)
    if near_match:
        resolved = resolve_locality(near_match.group(1).strip())
        if resolved.district or resolved.name:
            state.destination_district = resolved.district or resolved.name

    from_match = FROM_START_RE.search(message)
    if from_match and not state.starting_location:
        resolved = resolve_locality(from_match.group(1).strip())
        if resolved.name:
            state.starting_location = resolved.name
            state.mark_answered("start_location")
            if not state.destination_district:
                state.district = resolved.district or resolved.name

    duration = extract_duration(message)
    if duration is not None and state.duration_days is None:
        state.duration_days = duration
        state.mark_answered("duration")

    travellers = extract_travellers(message)
    if travellers and not state.travellers:
        state.travellers = travellers
        state.mark_answered("travellers")

    state.planning_mode = True
    state.tourism_intent = True
    state.itinerary_requested = True
    state.sub_intent = "itinerary"
    state.current_planning_step = next_planning_question(state)


def acknowledge_category_rejection(state: ConversationState) -> str:
    state.reject_category_questions = True
    state.category_confirmed = True
    state.planning_mode = True
    state.mark_answered("category")
    state.mark_answered("experience")
    state.mark_answered("mood")
    state.mark_answered("offer_recommend")
    return build_planning_question(next_planning_question(state) or "duration", state)


def _expecting_planning_answer(state: ConversationState) -> str | None:
    """Which planning field the user is most likely answering right now."""
    missing = next_planning_question(state)
    asked = state.last_question_key or state.current_planning_step

    if asked == "duration" and state.duration_days is not None:
        asked = None
    if asked == "start_location" and state.starting_location:
        asked = None
    if asked == "travellers" and state.travellers:
        asked = None
    if asked == "pace" and state.pace:
        asked = None

    if asked in PLANNING_STEPS:
        if asked == missing:
            return asked
        if asked == "pace" and missing == "interests":
            return asked
        if asked == "duration" and missing == "start_location":
            return "start_location"
        # Stale session: category tab skips interests in next_q but UI may still show it.
        if asked == "interests" and not state.has_answered("interests"):
            return "interests"
        if asked == "pace" and not state.pace and not state.has_answered("pace"):
            return "pace"

    return missing


def infer_category_from_interests(state: ConversationState) -> bool:
    """Map stated interests to a dataset category so open planning skips a redundant category step."""
    if state.category_confirmed and state.category_tag:
        return False
    if not state.interests or not state.has_answered("interests"):
        return False

    from src.services.preferences import CATEGORY_ALIASES, INTEREST_TO_DATASET, VALID_CATEGORIES

    for interest in reversed(state.interests):
        cat = None
        mapping = INTEREST_TO_DATASET.get(interest)
        if mapping:
            cat = mapping.get("category")
        if not cat:
            cat = CATEGORY_ALIASES.get(interest)
        if cat and cat in VALID_CATEGORIES:
            state.category_tag = cat
            state.category_confirmed = True
            state.mark_answered("category")
            if "category" not in state.last_parsed_fields:
                state.last_parsed_fields.append("category")
            return True
    return False


def infer_mood_from_interests(state: ConversationState) -> bool:
    """Map stated interests to a dataset mood_tag — never ask users for non-dataset 'pace'."""
    if state.mood_tag:
        return False
    if not state.interests:
        return False

    from src.services.preferences import INTEREST_TO_DATASET, VALID_MOODS, normalize_mood

    for interest in reversed(state.interests):
        mapping = INTEREST_TO_DATASET.get(interest) or {}
        mood = normalize_mood(mapping.get("mood"))
        if mood and mood in VALID_MOODS:
            state.mood_tag = mood
            state.mood = mood.lower()
            state.mark_answered("mood")
            if "mood" not in state.last_parsed_fields:
                state.last_parsed_fields.append("mood")
            return True
    return False


def apply_dataset_defaults_after_interests(state: ConversationState) -> None:
    """After interests: set internal schedule default + infer category & mood_tag from dataset."""
    if not state.has_answered("interests"):
        return
    if not state.pace:
        state.pace = "balanced"
        state.mark_answered("pace")
    state.pending_pace_clarify = False
    infer_category_from_interests(state)
    infer_mood_from_interests(state)


def apply_planning_answer(state: ConversationState, message: str) -> None:
    """Parse message and update structured state — never lose prior answers."""
    expecting = _expecting_planning_answer(state)
    parsed = parse_planning_message(message, expecting=expecting)
    state.last_parsed_fields = []

    if parsed["duration_days"] is not None and state.duration_days is None:
        state.duration_days = parsed["duration_days"]
        state.mark_answered("duration")
        state.last_parsed_fields.append("duration")

    if parsed["starting_location"] and not state.starting_location:
        from src.services.geography import _is_rejection_phrase

        if not _is_rejection_phrase(message):
            resolved = resolve_locality(parsed["starting_location"])
            state.starting_location = resolved.name
            if not state.destination_district:
                state.district = resolved.district or resolved.name
            state.mark_answered("start_location")
            state.last_parsed_fields.append("start_location")

    if parsed["travellers"] and not state.travellers:
        state.travellers = parsed["travellers"]
        state.mark_answered("travellers")
        state.last_parsed_fields.append("travellers")

    if state.category_confirmed and state.category_tag:
        from src.services.planning_input import normalize_user_input

        if normalize_user_input(message) == state.category_tag.lower():
            state.mark_answered("interests")
        elif expecting == "interests":
            state.mark_answered("interests")

    new_interests = [i for i in parsed["interests"] if i not in state.interests]
    allow_interest_update = (
        not state.category_confirmed
        or expecting == "interests"
        or state.current_itinerary
        or state.itinerary_modify
        or state.session_mode in ("itinerary_review", "revision")
    )
    if parsed["interests"] and not state.has_answered("interests") and allow_interest_update:
        state.interests = list(dict.fromkeys(state.interests + parsed["interests"]))
        state.mark_answered("interests")
        state.last_parsed_fields.append("interests")
        apply_dataset_defaults_after_interests(state)
    elif new_interests and allow_interest_update:
        state.interests.extend(new_interests)
        state.interests = list(dict.fromkeys(state.interests))
        state.mark_answered("interests")
        state.last_parsed_fields.append("interests")
        apply_dataset_defaults_after_interests(state)

    if parsed["pace"] and not state.pace:
        state.pace = parsed["pace"]
        state.pending_pace_clarify = False
        state.mark_answered("pace")
        state.last_parsed_fields.append("pace")
    elif parsed["pace_default"] and not state.pace:
        state.pace = "balanced"
        state.pending_pace_clarify = False
        state.mark_answered("pace")
        state.last_parsed_fields.append("pace")

    # Extra interests after initial answer — refresh dataset mappings
    if expecting == "interests" and parsed["interests"] and not new_interests:
        apply_dataset_defaults_after_interests(state)

    for cat in VALID_CATEGORIES:
        if re.search(rf"\b{re.escape(cat.lower())}\b", message.lower()) and re.search(
            r"\b(day|days|trip|travel|plan)\b", message.lower()
        ):
            if detect_explicit_category_selection(message, state) == cat:
                state.category_tag = cat
                state.category_confirmed = True
                state.planning_mode = True
                state.mark_answered("category")
            break

    # Explicit category selection during planning
    explicit_cat = detect_explicit_category_selection(message, state)
    if explicit_cat and next_planning_question(state) == "category":
        state.category_tag = explicit_cat
        state.category_confirmed = True
        state.mark_answered("category")
        state.last_parsed_fields.append("category")

    if "more relaxed" in message.lower() or "make it relaxed" in message.lower():
        state.pace = "relaxed"
    if "don't want temple" in message.lower() or "no temple" in message.lower():
        if "temple" not in state.dislikes:
            state.dislikes.append("temple")
    if "give me more" in message.lower() or "show more" in message.lower():
        state.wants_more_recommendations = True
    lower = message.lower()
    if any(p in lower for p in ("list all", "show all", "all places", "see all")):
        state.wants_list_all = True
        state.wants_more_recommendations = True

    _accept_start_location_if_missing(state, message)

    if state.has_answered("interests") and not state.category_confirmed:
        apply_dataset_defaults_after_interests(state)

    state.current_planning_step = next_planning_question(state)


def _accept_start_location_if_missing(state: ConversationState, message: str) -> None:
    """Accept a locality answer whenever start location is the next missing field."""
    from src.services.geography import _is_rejection_phrase, resolve_locality_from_text

    if state.starting_location or _is_rejection_phrase(message):
        return
    if next_planning_question(state) != "start_location":
        return

    resolved = resolve_locality_from_text(message, allow_bare=True)
    if not resolved:
        return

    state.starting_location = resolved.name
    if not state.destination_district:
        state.district = resolved.district or resolved.name
    state.mark_answered("start_location")
    if "start_location" not in state.last_parsed_fields:
        state.last_parsed_fields.append("start_location")


def has_core_trip_info(state: ConversationState) -> bool:
    """Minimum trip context for suggestions or building from a shortlist."""
    return bool(
        state.category_confirmed
        and state.category_tag
        and state.duration_days
        and state.starting_location
    )


def has_enough_for_suggest(state: ConversationState) -> bool:
    """User can request place suggestions without interests/pace."""
    return has_core_trip_info(state)


def has_enough_for_build(state: ConversationState) -> bool:
    """User can build an itinerary from a stored recommendation set."""
    if not state.duration_days:
        return False
    if not (state.last_recommendations or state.selected_attraction_ids):
        return False
    if has_core_trip_info(state):
        return True
    if state.mood and state.experience:
        return True
    if state.interests and state.tourism_intent:
        return True
    return bool(state.tourism_intent and not state.category_confirmed)


def has_valid_itinerary(state: ConversationState) -> bool:
    """True when the stored itinerary has at least one scheduled day."""
    itinerary = state.current_itinerary
    if not itinerary:
        return False
    return bool(itinerary.get("days"))


def next_planning_question(state: ConversationState) -> str | None:
    """Determine next missing field from structured state only."""
    if state.duration_days is None:
        return "duration"
    if not state.starting_location:
        return "start_location"
    # Mood or category tab already defines theme — skip redundant follow-ups.
    if not state.category_confirmed and not state.mood_confirmed:
        if not state.has_answered("interests"):
            return "interests"
        if not state.category_tag and not state.has_answered("category"):
            return "category"
    return None


def should_suggest_after_planning(state: ConversationState) -> bool:
    """After core Q&A with a confirmed category, show place suggestions next."""
    return bool(
        has_enough_for_suggest(state)
        and not state.last_recommendations
        and not state.current_itinerary
    )


def planning_ready(state: ConversationState) -> bool:
    return next_planning_question(state) is None


def build_planning_question(key: str, state: ConversationState) -> str:
    cat = state.category_tag or "your"

    if key == "duration":
        if state.mood_tag and state.mood_confirmed and not state.duration_days:
            return (
                f"Let's plan around a **{state.mood_tag}** mood — matched to our attractions dataset.\n\n"
                f"How many days are you planning to travel?"
            )
        if state.category_tag and not state.duration_days:
            return (
                f"Let's plan your **{state.category_tag}**-focused trip.\n\n"
                f"How many days are you planning to travel?"
            )
        return (
            "How many days are you planning to travel?\n"
            "• 1–2 days\n• 3–4 days\n• 5–7 days\n• More than a week"
        )

    if key == "start_location":
        if "duration" in state.last_parsed_fields:
            return "Great! Where will you be starting your journey?"
        return "Where will you be starting your journey?"

    if key == "travellers":
        if "start_location" in state.last_parsed_fields:
            return "Who are you travelling with — solo, couple, friends, or family?"
        return "Who are you travelling with — solo, couple, friends, or family?"

    if key == "interests":
        if "start_location" in state.last_parsed_fields:
            return "Perfect! What kind of experiences are you interested in?"
        return "What kind of experiences are you interested in — adventure, wildlife, nature, beaches, heritage, or something else?"

    if key == "pace":
        return "What kind of pace do you prefer — relaxed, balanced, or packed?"

    if key == "category":
        if state.interests:
            interest = state.interests[-1]
            return (
                f"{interest.title()} sounds great. Which category would you like to focus on?\n\n"
                "**Wild** · **Heritage** · **Scenic** · **Pristine** · **Essence** · **Thrills**"
            )
        return (
            "What kind of experience would you like?\n\n"
            "**Wild** · **Heritage** · **Scenic** · **Pristine** · **Essence** · **Thrills**"
        )

    return "Tell me a bit more about the trip you're planning."


def build_planning_transition(state: ConversationState) -> str | None:
    """Build acknowledgment when fields were just parsed — avoids repeating prior question."""
    if not state.last_parsed_fields:
        return None

    if planning_ready(state):
        cat = state.category_tag or "Sri Lanka"
        interest_txt = ""
        if state.interests:
            interest_txt = f", **{state.interests[0]}**-focused"
        mood_txt = f", **{state.mood_tag}** mood" if state.mood_tag else ""
        return (
            f"A **{cat}**{interest_txt}{mood_txt} trip from "
            f"**{state.starting_location}** for **{state.duration_days} days** — "
            f"I'm putting together your itinerary now."
        )

    nxt = next_planning_question(state)
    if nxt:
        return build_planning_question(nxt, state)
    return None


def build_planning_intro_for_recommend(state: ConversationState) -> str:
    parts = []
    if state.category_tag and state.category_confirmed:
        parts.append(f"a **{state.category_tag}** focus")
    elif state.category_tag:
        parts.append(f"a **{state.category_tag}** lean")
    if state.mood_tag and state.mood_confirmed:
        parts.append(f"**{state.mood_tag}** mood")
    elif state.mood_tag:
        parts.append(f"a **{state.mood_tag}** feel")
    if state.duration_days:
        parts.append(f"**{state.duration_days} days**")
    if state.starting_location:
        parts.append(f"starting from **{state.starting_location}**")
    if state.travellers:
        parts.append(f"travelling as **{state.travellers}**")
    if state.interests and not state.mood_confirmed:
        parts.append(f"interested in **{', '.join(state.interests)}**")
    ctx = ", ".join(parts) if parts else "your preferences"
    return f"Absolutely. Since you're planning {ctx}, here are some places I'd shortlist:"

"""Parse and normalize natural-language planning answers."""

from __future__ import annotations

import re

PACE_VALUES = {"relaxed", "balanced", "packed"}

PACE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "relaxed": ("relaxed", "relax", "slow", "easy", "chill", "leisurely"),
    "balanced": ("balanced", "balance", "moderate", "medium", "normal"),
    "packed": ("packed", "pack", "busy", "full", "intense", "lots", "max"),
}

INTEREST_SYNONYMS: dict[str, tuple[str, ...]] = {
    "adventure": (
        "adventure", "adventures", "adventurous", "advemture", "adventur", "adveture",
        "thrilling", "exciting", "active", "adrenaline",
    ),
    "wildlife": ("wildlife", "safari", "animals", "birdwatching", "birds", "wild"),
    "nature": ("nature", "natural", "forest", "jungle", "hiking", "trek", "trekking"),
    "beach": ("beach", "beaches", "coastal", "sea", "ocean", "snorkel"),
    "heritage": ("heritage", "history", "historic", "cultural", "culture", "temple", "temples"),
    "photography": ("photography", "photos", "views", "landscape"),
    "waterfalls": ("waterfall", "waterfalls"),
}

TRAVELLER_SYNONYMS: dict[str, tuple[str, ...]] = {
    "solo": ("solo", "alone", "just me", "by myself", "myself"),
    "couple": ("couple", "partner", "girlfriend", "boyfriend", "wife", "husband", "two of us"),
    "friends": ("friends", "friend", "mates", "buddies"),
    "family": ("family", "parents", "kids", "children", "child"),
}

from src.services.geography import resolve_locality_from_text

DURATION_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "weekend": 2, "week": 7,
}


def normalize_user_input(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def normalize_mood_feel_phrases(text: str) -> str:
    """Fix glued mood phrases like 'ifeelcurious' → 'i feel curious'."""
    if not text:
        return text
    normalized = text
    normalized = re.sub(
        r"\bifeel\s*(curious|happy|relaxed|excited|peaceful|adventurous|spiritual)\b",
        r"i feel \1",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bi\s*feel\s*(curious|happy|relaxed|excited|peaceful|adventurous|spiritual)\b",
        r"i feel \1",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(i'm|im)\s*feeling\s*(curious|happy|relaxed|excited|peaceful|adventurous|spiritual)\b",
        r"i feel \2",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized


def _contains_token(text: str, token: str) -> bool:
    if len(token) <= 4:
        return bool(re.search(rf"\b{re.escape(token)}\b", text))
    return token in text


def _fuzzy_interest_match(text: str) -> list[str]:
    """Match interests including common typos via substring / edit distance."""
    found: list[str] = []
    compact = text.replace(" ", "")
    for interest, synonyms in INTEREST_SYNONYMS.items():
        for syn in synonyms:
            if _contains_token(text, syn):
                found.append(interest)
                break
            # typo tolerance: allow 1-char edit on words >= 6 chars
            if len(syn) >= 6 and syn in compact:
                found.append(interest)
                break
            if len(syn) >= 7:
                for word in re.findall(r"[a-z']+", text):
                    if _levenshtein(word, syn) <= 2:
                        found.append(interest)
                        break
        if interest in found:
            continue
    return list(dict.fromkeys(found))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def extract_duration(text: str) -> int | None:
    lower = normalize_user_input(text)
    if not lower:
        return None

    m = re.search(r"(\d+)\s*b?\s*[-]?\s*(?:dayy|days|day|night|nights)\b", lower)
    if m:
        days = int(m.group(1))
        return days if 1 <= days <= 30 else None

    compact = lower.replace(" ", "")
    m = re.search(r"^(\d{1,2})b?days?$", compact)
    if m:
        return int(m.group(1))

    if re.fullmatch(r"\d{1,2}", lower):
        days = int(lower)
        return days if 1 <= days <= 30 else None

    for word, num in DURATION_WORDS.items():
        if re.search(rf"\b{word}\b", lower):
            if "week" in lower and word not in ("week", "weekend"):
                continue
            return num

    m = re.search(r"\b(\d{1,2})\b", lower)
    if m and re.search(r"\b(day|days|trip|travel|have|for)\b", lower):
        days = int(m.group(1))
        return days if 1 <= days <= 30 else None

    return None


def extract_start_location(text: str, *, allow_bare: bool = False) -> str | None:
    resolved = resolve_locality_from_text(text, allow_bare=allow_bare)
    return resolved.name if resolved else None


def extract_travellers(text: str) -> str | None:
    lower = normalize_user_input(text)
    for traveller, synonyms in TRAVELLER_SYNONYMS.items():
        for syn in synonyms:
            if syn in lower:
                return traveller
    return None


def extract_pace(text: str) -> str | None:
    lower = normalize_user_input(text)
    if _fuzzy_interest_match(lower) and not any(
        _contains_token(lower, s) for syns in PACE_SYNONYMS.values() for s in syns
    ):
        return None
    for pace, synonyms in PACE_SYNONYMS.items():
        for syn in synonyms:
            if _contains_token(lower, syn):
                return pace
    return None


def _is_open_planning_message(text: str) -> bool:
    """True when the user is opening a trip plan, not answering an interests question."""
    lower = normalize_user_input(text)
    if not re.search(r"\b(plan|trips?|tours?|travel(?:ing)?|getaway)\b", lower):
        return False
    return extract_duration(text) is not None or bool(
        re.search(r"\b(day|days|trips?|tours?)\b", lower)
    )


def extract_interests(text: str, *, expecting: str | None = None) -> list[str]:
    lower = normalize_user_input(text)
    if lower.startswith("__start_planning__:"):
        return []
    from src.services.preferences import VALID_CATEGORIES

    if lower in {c.lower() for c in VALID_CATEGORIES}:
        if expecting == "interests" and lower == "wild":
            return ["wildlife"]
        return []
    if _is_open_planning_message(text) and expecting not in ("interests", "pace"):
        return []
    if extract_pace(lower):
        pace_words = {s for syns in PACE_SYNONYMS.values() for s in syns}
        if any(_contains_token(lower, w) for w in pace_words):
            return _fuzzy_interest_match(lower)
        return _fuzzy_interest_match(lower)
    return _fuzzy_interest_match(lower)


def is_pace_refusal(text: str) -> bool:
    lower = normalize_user_input(text)
    if lower in {"ok", "okay", "k", "sure", "yes", "fine", "yep", "yeah", "y", "yea"}:
        return True
    return any(p in lower for p in (
        "don't care", "dont care", "doesn't matter", "doesnt matter",
        "no preference", "whatever", "up to you", "not bothered",
    ))


def parse_planning_message(text: str, *, expecting: str | None = None) -> dict:
    """Extract all planning fields present in one message."""
    allow_bare_location = expecting == "start_location"
    return {
        "duration_days": extract_duration(text),
        "starting_location": extract_start_location(text, allow_bare=allow_bare_location),
        "travellers": extract_travellers(text),
        "pace": extract_pace(text),
        "interests": extract_interests(text, expecting=expecting),
        "pace_default": is_pace_refusal(text),
    }

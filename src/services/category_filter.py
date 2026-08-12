"""Hard category and mood_tag eligibility checks for recommendations and itineraries."""

from __future__ import annotations

from src.data.normalizer import normalize_text
from src.services.preferences import normalize_category, normalize_mood


def normalize_category_value(value: str | None) -> str | None:
    return normalize_category(value)


def is_category_compatible(attraction: dict, selected_category: str | None) -> bool:
    """Return True when attraction matches the selected dataset category."""
    if not selected_category:
        return True
    expected = normalize_category_value(selected_category)
    actual = normalize_category_value(attraction.get("category"))
    if not expected or not actual:
        return False
    return normalize_text(expected) == normalize_text(actual)


def filter_by_category(
    candidates: list[dict],
    selected_category: str | None,
    *,
    strict: bool = True,
) -> list[dict]:
    """Keep only category-compatible attractions when strict and category is set."""
    if not strict or not selected_category:
        return list(candidates)
    return [c for c in candidates if is_category_compatible(c, selected_category)]


def is_mood_compatible(attraction: dict, selected_mood: str | None) -> bool:
    """Return True when attraction matches the dataset mood_tag (attractions.xlsx)."""
    if not selected_mood:
        return True
    expected = normalize_mood(selected_mood)
    actual = normalize_mood(attraction.get("mood"))
    if not expected or not actual:
        return False
    return normalize_text(expected) == normalize_text(actual)


def filter_by_mood(
    candidates: list[dict],
    selected_mood: str | None,
    *,
    strict: bool = True,
) -> list[dict]:
    """Keep only mood_tag-compatible attractions when strict and mood is set."""
    if not strict or not selected_mood:
        return list(candidates)
    return [c for c in candidates if is_mood_compatible(c, selected_mood)]

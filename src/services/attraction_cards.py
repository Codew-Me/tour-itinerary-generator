"""User-facing attraction card formatting (no internal category/mood labels)."""

from __future__ import annotations

import re

from src.services.review_stats import enrich_attraction

_GENERIC_DESCRIPTIONS = frozenset({
    "a notable sri lanka destination worth exploring.",
    "very nice place to visit.",
    "nice place to visit.",
    "good place to visit.",
    "great place to visit.",
    "must visit place.",
})


def _strip_leading_name(text: str, name: str | None) -> str:
    if not name:
        return text
    name_norm = re.sub(r"\s+", " ", name.strip())
    if not name_norm:
        return text
    pattern = re.compile(rf"^{re.escape(name_norm)}\s*[-–—:,]?\s*", re.IGNORECASE)
    return pattern.sub("", text).strip()


def format_details_for_card(details: str | None, *, name: str | None = None) -> str:
    """Return the full cleaned description for display (not truncated)."""
    if not details or not str(details).strip():
        return ""
    text = re.sub(r"\s+", " ", str(details).strip())
    text = _strip_leading_name(text, name)
    text = re.sub(r"^(?:is|are|was|were)\s+", "", text, flags=re.IGNORECASE)
    if not text:
        return ""
    if text.lower().strip() in _GENERIC_DESCRIPTIONS:
        return ""
    return text


def summarize_details(details: str | None, *, name: str | None = None, max_len: int | None = None) -> str:
    """Backward-compatible alias — returns full description unless max_len is set."""
    text = format_details_for_card(details, name=name)
    if not text or max_len is None:
        return text
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut.rstrip(".,;") + "..."


def _quote_review_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if not cleaned:
        return ""
    if not cleaned.startswith('"'):
        cleaned = f'"{cleaned}"'
    return cleaned


def _collect_review_quotes(candidate: dict, *, max_quotes: int = 2) -> list[str]:
    quotes: list[str] = []
    for item in candidate.get("sample_reviews") or []:
        if isinstance(item, dict):
            raw = item.get("review") or item.get("text") or ""
        else:
            raw = str(item)
        quote = _quote_review_text(raw)
        if quote and quote not in quotes:
            quotes.append(quote)
        if len(quotes) >= max_quotes:
            return quotes
    for raw in candidate.get("review_samples") or []:
        quote = _quote_review_text(str(raw))
        if quote and quote not in quotes:
            quotes.append(quote)
        if len(quotes) >= max_quotes:
            break
    return quotes


def format_visitor_feedback(candidate: dict) -> str | None:
    """Build a clear visitor feedback block from review stats and sample quotes."""
    parts: list[str] = []
    summary = candidate.get("review_summary")
    if summary:
        parts.append(summary.rstrip("."))
    parts.extend(_collect_review_quotes(candidate))
    if not parts:
        return None
    return "\n  ".join(parts)


def format_attraction_card(candidate: dict, *, include_review: bool = True) -> str:
    name = candidate.get("name", "Unknown")
    if include_review:
        c = enrich_attraction(candidate)
        district = c.get("display_district") or c.get("district") or "—"
        feedback = format_visitor_feedback(c)
    else:
        c = candidate
        district = c.get("district") or c.get("display_district") or "—"
        feedback = None

    desc = format_details_for_card(c.get("details"), name=name)
    lines = [f"- **{name}** · {district}"]
    if desc:
        lines.append(f"  {desc}")
    if include_review and feedback:
        for part in feedback.split("\n  "):
            part = part.strip()
            if part:
                lines.append(f"  _{part}_")
    return "\n".join(lines)


def format_recommendation_response(
    candidates: list[dict],
    *,
    intro: str | None = None,
    closing: str | None = None,
) -> str:
    if not candidates:
        return (
            "I couldn't find matching places in our dataset for that request. "
            "Try adjusting your category, starting location, or trip length."
        )
    parts: list[str] = []
    if intro:
        parts.append(intro)
    else:
        parts.append("Here are a few options that could work well:")
    parts.append("")
    for c in candidates:
        parts.append(format_attraction_card(c))
        parts.append("")
    if closing:
        parts.append(closing)
    elif len(candidates) >= 3:
        parts.append("Want me to build these into a day-by-day itinerary, or show you more options?")
    return "\n".join(parts).strip()

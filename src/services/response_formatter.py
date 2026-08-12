"""Format dataset-backed responses."""

from __future__ import annotations

import json
from typing import Any


def _evidence_badge(cand: dict) -> str:
    level = cand.get("evidence_level", "")
    if level == "Review-supported" or cand.get("sample_reviews"):
        return "✓ Review-supported"
    return "ℹ Metadata-supported"


def format_recommendations(data: dict) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return (
            "I couldn't find matching attractions in our dataset for that request. "
            "Try specifying a mood (peaceful, adventure), category (heritage, nature), or district."
        )

    lines = ["Here are attractions that match your request:\n"]
    for i, cand in enumerate(candidates[:6], 1):
        name = cand.get("name", "Unknown")
        district = cand.get("district") or cand.get("destination") or "—"
        category = cand.get("category") or "—"
        mood = cand.get("mood") or "—"
        review_count = cand.get("review_count", 0)
        badge = _evidence_badge(cand)

        lines.append(f"### {i}. {name}")
        lines.append(f"**District:** {district} · **Category:** {category} · **mood_tag:** {mood}")
        lines.append(f"**Linked reviews in dataset:** {review_count} · **Evidence:** {badge}")

        details = (cand.get("details") or "").strip()
        if details:
            excerpt = details[:400] + ("..." if len(details) > 400 else "")
            lines.append(f"**Details:** {excerpt}")

        if cand.get("sample_reviews"):
            snippet = cand["sample_reviews"][0].get("review", "")[:140]
            if snippet:
                lines.append(f"> *Visitor review:* \"{snippet}...\"")

        lines.append("")
    return "\n".join(lines)


def format_diverse_sample(data: dict) -> str:
    lines = []
    for i, cand in enumerate(data.get("candidates", [])[:4], 1):
        badge = _evidence_badge(cand)
        lines.append(
            f"- **{cand.get('name')}** ({cand.get('district', '—')}) — "
            f"{cand.get('category', '—')} · {cand.get('mood', '—')} · {badge}"
        )
    return "\n".join(lines)


def format_attractions(items: list[dict]) -> str:
    if not items:
        return "I couldn't find matching attractions in the dataset for that query."
    lines = ["Here is what I found in the Sri Lanka travel dataset:\n"]
    for i, item in enumerate(items[:8], 1):
        badge = _evidence_badge(item)
        lines.append(f"### {i}. {item.get('name', 'Unknown')}")
        lines.append(
            f"**District:** {item.get('district', '—')} · "
            f"**Category:** {item.get('category', '—')} · **Mood:** {item.get('mood', '—')}"
        )
        lines.append(f"**Linked reviews:** {item.get('review_count', 0)} · **Evidence:** {badge}")
        details = (item.get("details") or "")[:200]
        if details:
            lines.append(f"{details}...")
        lines.append("")
    return "\n".join(lines)


def format_from_tool_results(tool_results: list[Any]) -> str | None:
    for result in reversed(tool_results):
        if isinstance(result, list) and result and isinstance(result[0], dict):
            if "name" in result[0] and "category" in result[0]:
                return format_attractions(result)
        if isinstance(result, dict):
            if "candidates" in result:
                return format_recommendations(result)
            if "attractions" in result:
                return format_attractions(result.get("attractions", []))
    return None

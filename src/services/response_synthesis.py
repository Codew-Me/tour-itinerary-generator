"""LLM synthesizes natural responses from structured attraction data."""

from __future__ import annotations

from src.agent.graph import get_llm
from src.services.response_formatter import format_recommendations


SYNTHESIS_SYSTEM = """You are a friendly Sri Lanka travel assistant.

Write a warm, conversational response using ONLY the attraction data provided.
Each place was found by matching the user's preferences to our dataset:
- Category (attraction type)
- mood_tag (travel mood)
- Linked traveler reviews (when available)

CRITICAL RULES:
- Describe each place using the **Details** field from the data — paraphrase naturally, do NOT invent facts.
- Use the exact Category and mood_tag shown for each attraction.
- If a visitor review snippet is provided, use it as supporting evidence only — do not turn it into a separate activity.
- If reviews exist, note ✓ Review-supported; otherwise ℹ Metadata-supported.
- Do NOT invent prices, opening hours, or booking advice.
- Do NOT expose internal scores or system labels.

Format:

🌴 Here are a few places that match what you're looking for:

### 1. Attraction Name
📍 District · Category · mood_tag

About this place:
2–3 sentences drawn from the Details field — what makes it special and why it fits the user's mood/interest.

Evidence: ✓ Review-supported  OR  ℹ Metadata-supported
(Optional brief review quote if provided)

End with a short, natural follow-up (e.g. offer to plan an itinerary or narrow by district).
"""


def synthesize_response(
    user_message: str,
    history: list[dict],
    recommendation_data: dict,
) -> str:
    """Generate natural LLM response grounded in recommendation_data."""
    structured_block = format_recommendations(recommendation_data)

    history_text = "\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
        for m in (history or [])[-8:]
    )

    prefs = recommendation_data.get("inferred_category"), recommendation_data.get("inferred_mood")
    filter_note = ""
    if prefs[0] or prefs[1]:
        filter_note = f"Search filters applied — Category: {prefs[0] or 'any'}, mood_tag: {prefs[1] or 'any'}"

    prompt = f"""Conversation:
{history_text or '(start)'}

User's latest message: {user_message}
{filter_note}

Attraction data from our Sri Lanka database (use ONLY this — especially the Details field):
{structured_block}

Write the assistant's response:"""

    try:
        llm = get_llm()
        result = llm.invoke([
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        content = result.content if hasattr(result, "content") else str(result)
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        text = str(content).strip()
        if text and len(text) > 50:
            return text
    except Exception:
        pass

    return structured_block

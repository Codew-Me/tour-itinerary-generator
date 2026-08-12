"""LLM-driven conversation routing — decides when to chat vs search."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.agent.graph import get_llm
from src.services.preferences import VALID_CATEGORIES, VALID_MOODS, TourismPreferences


class ActionType(str, Enum):
    RESPOND = "respond"
    CLARIFY = "clarify"
    SEARCH = "search"


class SearchParams(BaseModel):
    district: str | None = None
    destination: str | None = None
    category: str | None = Field(None, description="One of: Heritage, Scenic, Essence, Wild, Pristine, Thrills")
    mood: str | None = Field(None, description="One of: Curious, Relaxed, Peaceful, Adventure, etc.")
    experience_description: str | None = Field(
        None,
        description="Natural language description of desired experience for semantic search",
    )
    diverse: bool = Field(False, description="True for 'surprise me' — pick varied categories")


class ConversationDecision(BaseModel):
    action: Literal["respond", "clarify", "search"]
    response: str | None = Field(None, description="Natural reply for respond/clarify actions")
    search_params: SearchParams | None = Field(None, description="Required when action is search")


ROUTER_SYSTEM = f"""You are the conversation brain for a Sri Lanka travel assistant.

Given the conversation history and latest user message, choose exactly ONE action:

1. **respond** — Normal conversation: greetings, thanks, bye, small talk, acknowledgments.
   Do NOT search any database. Write a warm, natural reply.

2. **clarify** — User may want travel help but has NOT explicitly asked for recommendations yet.
   Examples: "I'm bored", "I don't know what to do", "somewhere to go" (first time, no preferences yet).
   Ask a helpful follow-up question. Do NOT search yet.

3. **search** — User clearly wants destination/attraction recommendations NOW.
   Examples: "surprise me" (after wanting suggestions), "I want peaceful places in Galle",
   "show me heritage in Galle", "I want adventure", "recommend beaches in Matara".
   Extract search_params from the FULL conversation context (remember location from earlier turns).

CRITICAL RULES:
- "hi", "hello", "thanks", "bye" → respond (never search)
- "I'm bored" alone → clarify (do NOT search, do NOT recommend attractions yet)
- "somewhere to go" after bored → clarify (ask what kind of experience)
- "surprise me" after user wanted suggestions → search with diverse=true
- Use conversation history: if user said "going to Galle" earlier, remember location=Galle
- Do NOT map "bored" to Adventure automatically
- Do NOT expose internal labels to the user in the response field

Valid categories: {", ".join(VALID_CATEGORIES)}
Valid moods: {", ".join(VALID_MOODS)}

For semantic requests like "clear my head" or "something interesting", set experience_description
in natural language (e.g. "calm relaxing peaceful nature") — do not require exact mood_tag match.
"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior messages)"
    lines = []
    for msg in history[-12:]:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_json_decision(text: str) -> ConversationDecision | None:
    try:
        # Extract JSON object from response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        return ConversationDecision.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None


def route_conversation(message: str, history: list[dict] | None = None) -> tuple[ConversationDecision, TourismPreferences | None]:
    """
    LLM decides: respond | clarify | search.
    Returns decision and TourismPreferences if action is search.
    """
    history = history or []
    user_prompt = f"""Conversation so far:
{_format_history(history)}

Latest user message: {message}

Return JSON matching this schema:
{{
  "action": "respond" | "clarify" | "search",
  "response": "natural reply text (required for respond/clarify)",
  "search_params": {{
    "district": null or string,
    "destination": null or string,
    "category": null or one of valid categories,
    "mood": null or one of valid moods,
    "experience_description": null or string,
    "diverse": false or true
  }}
}}

search_params is required only when action is "search"."""

    try:
        llm = get_llm()
        try:
            structured = llm.with_structured_output(ConversationDecision)
            decision = structured.invoke([
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
        except Exception:
            raw = llm.invoke([
                {"role": "system", "content": ROUTER_SYSTEM + "\nRespond with valid JSON only."},
                {"role": "user", "content": user_prompt},
            ])
            content = raw.content if hasattr(raw, "content") else str(raw)
            decision = _parse_json_decision(content)
            if not decision:
                raise ValueError("Could not parse LLM decision")

        prefs = None
        if decision.action == "search" and decision.search_params:
            prefs = TourismPreferences.from_llm_dict(decision.search_params.model_dump())

        return decision, prefs

    except Exception:
        return _fallback_route(message, history)


def _fallback_route(message: str, history: list[dict] | None) -> tuple[ConversationDecision, TourismPreferences | None]:
    """Conservative fallback when LLM unavailable — never search on greetings or bored alone."""
    history = history or []
    lower = message.strip().lower()
    all_text = " ".join(m.get("content", "") for m in history).lower() + " " + lower

    greetings = {"hi", "hello", "hey", "hiya", "yo", "good morning", "good evening", "good afternoon"}
    if lower.rstrip("!.?") in greetings or lower in {"thanks", "thank you", "thx", "bye", "goodbye", "okay", "ok"}:
        if "thank" in lower:
            return ConversationDecision(action="respond", response="You're welcome! Happy to help with your Sri Lanka travel plans anytime."), None
        if lower in {"bye", "goodbye"}:
            return ConversationDecision(action="respond", response="Safe travels. Come back anytime you need destination ideas."), None
        return ConversationDecision(
            action="respond",
            response="Hello. I'm your Sri Lanka travel assistant. How can I help you plan your trip today?",
        ), None

    if lower in {"im bored", "i'm bored", "bored"}:
        return ConversationDecision(
            action="clarify",
            response="Let's find something worth doing. Want me to suggest a destination, an activity, or should I surprise you?",
        ), None

    # "surprise me" with prior tourism context
    if "surprise" in lower and any(w in all_text for w in ("go", "bored", "somewhere", "place", "suggest")):
        return ConversationDecision(action="search", response=None, search_params=SearchParams(diverse=True)), TourismPreferences(diverse=True)

    # Clear tourism requests
    search_signals = (
        "peaceful", "adventure", "heritage", "beach", "nature", "wildlife",
        "recommend", "show me", "places in", "attractions in", "clear my head",
    )
    if any(s in lower for s in search_signals) or ("want" in lower and any(w in lower for w in ("place", "somewhere", "destination"))):
        prefs = TourismPreferences(experience_description=message, search_query=message)
        for loc in ("galle", "matale", "colombo", "kandy", "matara", "badulla", "ella"):
            if loc in lower:
                prefs.district = loc.title() if loc != "ella" else "Badulla"
        if "heritage" in lower:
            prefs.category = "Heritage"
        if "adventure" in lower or "adventurous" in lower:
            prefs.mood = "Adventure"
        if "peaceful" in lower or "clear my head" in lower:
            prefs.mood = "Peaceful"
            prefs.experience_description = "peaceful calm relaxing nature scenic"
        return ConversationDecision(action="search", response=None, search_params=SearchParams(**{
            k: v for k, v in {
                "district": prefs.district,
                "category": prefs.category,
                "mood": prefs.mood,
                "experience_description": prefs.experience_description,
            }.items() if v
        })), prefs

    # "somewhere to go" — clarify unless we already clarified
    if "somewhere" in lower or "something to do" in lower:
        return ConversationDecision(
            action="clarify",
            response="What kind of experience are you looking for — nature, beach, adventure, heritage, relaxation, or should I surprise you?",
        ), None

    return ConversationDecision(
        action="clarify",
        response="I'd love to help you discover Sri Lanka! Are you looking for place recommendations, or just chatting?",
    ), None

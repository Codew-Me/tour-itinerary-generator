"""Structured conversation state for tourism chat."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.services.preferences import TourismPreferences, VALID_CATEGORIES, VALID_MOODS


@dataclass
class ConversationState:
    tourism_intent: bool = False
    sub_intent: str | None = None  # recommend | itinerary
    mood: str | None = None  # relaxing, adventurous, peaceful, etc.
    experience: str | None = None  # beach, nature, heritage, cultural
    district: str | None = None
    destination_district: str | None = None  # "near Kandy" preference — not the journey start
    starting_location: str | None = None
    category_tag: str | None = None  # direct dataset Category filter
    mood_tag: str | None = None  # direct dataset mood_tag filter
    duration_days: int | None = None
    recommendation_requested: bool = False
    itinerary_requested: bool = False
    use_previous_recommendations: bool = False
    last_recommendations: list[dict[str, Any]] = field(default_factory=list)
    selected_attraction_ids: list[int] = field(default_factory=list)
    pending_action: str | None = None  # awaiting_duration | generate_itinerary
    last_question_key: str | None = None
    answered_keys: list[str] = field(default_factory=list)
    # Itinerary planning agent state
    planning_mode: bool = False
    category_confirmed: bool = False  # set when user picks a tab / explicit category
    mood_confirmed: bool = False  # set when user states a dataset mood_tag (attractions.xlsx)
    reject_category_questions: bool = False  # user asked to stop category re-questions
    travellers: str | None = None  # solo | couple | friends | family
    pace: str | None = None  # relaxed | balanced | packed
    interests: list[str] = field(default_factory=list)
    current_planning_step: str | None = None
    pending_pace_clarify: bool = False  # user gave interest when pace was asked
    last_parsed_fields: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    already_recommended_ids: list[int] = field(default_factory=list)
    current_itinerary: dict | None = None
    wants_more_recommendations: bool = False
    wants_list_all: bool = False
    itinerary_modify: bool = False
    itinerary_modify_mode: str | None = None  # relaxed | add | focus
    awaiting_itinerary_followup: bool = False
    # Agent orchestration memory
    agent_phase: str = "idle"  # idle | planning | recommending | itinerary | revision
    session_mode: str = "idle"  # idle | planning | itinerary_review | revision
    last_intent: str | None = None
    intent_history: list[str] = field(default_factory=list)
    rejected_attraction_ids: list[int] = field(default_factory=list)
    rejected_itineraries: list[dict] = field(default_factory=list)
    awaiting_rejection_reason: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConversationState:
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        if "answered_keys" not in filtered:
            filtered["answered_keys"] = []
        if "last_recommendations" not in filtered:
            filtered["last_recommendations"] = []
        if "selected_attraction_ids" not in filtered:
            filtered["selected_attraction_ids"] = []
        if "dislikes" not in filtered:
            filtered["dislikes"] = []
        if "must_include" not in filtered:
            filtered["must_include"] = []
        if "already_recommended_ids" not in filtered:
            filtered["already_recommended_ids"] = []
        if "interests" not in filtered:
            filtered["interests"] = []
        if "last_parsed_fields" not in filtered:
            filtered["last_parsed_fields"] = []
        if "intent_history" not in filtered:
            filtered["intent_history"] = []
        if "agent_phase" not in filtered:
            filtered["agent_phase"] = "idle"
        if "session_mode" not in filtered:
            filtered["session_mode"] = "idle"
        if "rejected_attraction_ids" not in filtered:
            filtered["rejected_attraction_ids"] = []
        if "rejected_itineraries" not in filtered:
            filtered["rejected_itineraries"] = []
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str | None) -> ConversationState:
        if not raw:
            return cls()
        try:
            return cls.from_dict(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            return cls()

    def mark_answered(self, key: str) -> None:
        if key not in self.answered_keys:
            self.answered_keys.append(key)

    def has_answered(self, key: str) -> bool:
        return key in self.answered_keys

    def to_tourism_preferences(self) -> TourismPreferences:
        """Map conversational state to dataset filters (Category + mood_tag columns)."""
        from src.services.preferences import (
            INTEREST_TO_DATASET,
            normalize_category,
            normalize_mood,
        )

        category = normalize_category(self.category_tag) if self.category_confirmed else None
        mood = normalize_mood(self.mood_tag) if self.mood_confirmed else None

        if not self.category_confirmed:
            if not category:
                category = normalize_category(self.experience)
            for key in (self.experience,):
                if key and key.lower() in INTEREST_TO_DATASET:
                    mapping = INTEREST_TO_DATASET[key.lower()]
                    category = category or normalize_category(mapping.get("category"))

        if not self.mood_confirmed:
            if not mood:
                mood = normalize_mood(self.mood)
            if not self.category_confirmed:
                for key in (self.experience, self.mood):
                    if key and key.lower() in INTEREST_TO_DATASET:
                        mapping = INTEREST_TO_DATASET[key.lower()]
                        mood = mood or normalize_mood(mapping.get("mood"))
            for interest in reversed(self.interests):
                mapping = INTEREST_TO_DATASET.get(interest) or {}
                mood = mood or normalize_mood(mapping.get("mood"))

        exp_parts: list[str] = []
        if self.interests:
            exp_parts.extend(self.interests)
        if self.pace:
            exp_parts.append(self.pace)

        search_area = self.destination_district or self.district
        return TourismPreferences(
            district=search_area,
            destination=self.destination_district,
            category=category,
            mood=mood,
            experience_description=" ".join(exp_parts) if exp_parts else None,
            search_query=" ".join(exp_parts) if exp_parts else None,
            desired_experience=exp_parts,
            category_confirmed=self.category_confirmed,
            mood_confirmed=self.mood_confirmed,
        ).validated()

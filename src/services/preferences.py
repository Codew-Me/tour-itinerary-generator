"""Tourism search preferences — strict dataset Category + mood_tag values only."""

from __future__ import annotations

from dataclasses import dataclass, field

# Exact values from attractions dataset
VALID_CATEGORIES = ["Wild", "Heritage", "Scenic", "Pristine", "Essence", "Thrills"]

VALID_MOODS = [
    "Adventure",
    "Authentic",
    "Curious",
    "Excited",
    "Explore",
    "Happy",
    "Healing",
    "Peaceful",
    "Relaxed",
    "Spiritual",
]

# Conversational words → dataset mood_tag (must be in VALID_MOODS)
MOOD_ALIASES: dict[str, str] = {
    "adventure": "Adventure",
    "adventurous": "Adventure",
    "authentic": "Authentic",
    "curious": "Curious",
    "excited": "Excited",
    "explore": "Explore",
    "exploring": "Explore",
    "happy": "Happy",
    "fun": "Happy",
    "healing": "Healing",
    "peaceful": "Peaceful",
    "calm": "Peaceful",
    "relax": "Relaxed",
    "relaxing": "Relaxed",
    "relaxed": "Relaxed",
    "spiritual": "Spiritual",
}

# Conversational words → dataset Category (must be in VALID_CATEGORIES)
CATEGORY_ALIASES: dict[str, str] = {
    "wild": "Wild",
    "wildlife": "Wild",
    "nature": "Wild",
    "heritage": "Heritage",
    "cultural": "Heritage",
    "culture": "Heritage",
    "history": "Heritage",
    "scenic": "Scenic",
    "photography": "Scenic",
    "views": "Scenic",
    "pristine": "Pristine",
    "beach": "Pristine",
    "beaches": "Pristine",
    "coastal": "Pristine",
    "essence": "Essence",
    "family": "Essence",
    "local": "Essence",
    "thrills": "Thrills",
    "thrill": "Thrills",
    "adrenaline": "Thrills",
}


def normalize_mood(value: str | None) -> str | None:
    if not value:
        return None
    title = value.strip().title()
    if title in VALID_MOODS:
        return title
    return MOOD_ALIASES.get(value.strip().lower())


def normalize_category(value: str | None) -> str | None:
    if not value:
        return None
    title = value.strip().title()
    if title in VALID_CATEGORIES:
        return title
    lower = value.strip().lower()
    if lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[lower]
    return CATEGORY_ALIASES.get(lower)


# Legacy interest map (conversational → dataset columns)
INTEREST_TO_DATASET: dict[str, dict[str, str | None]] = {
    k: {"category": v, "mood": None}
    for k, v in CATEGORY_ALIASES.items()
    if k not in ("wild", "heritage", "scenic", "pristine", "essence", "thrills")
}
INTEREST_TO_DATASET.update({
    "adventure": {"category": "Thrills", "mood": "Adventure"},
    "adventurous": {"category": "Thrills", "mood": "Adventure"},
    "relaxation": {"category": None, "mood": "Peaceful"},
    "relaxing": {"category": None, "mood": "Relaxed"},
    "peaceful": {"category": None, "mood": "Peaceful"},
    "family": {"category": "Essence", "mood": "Relaxed"},
    "beach": {"category": "Pristine", "mood": "Relaxed"},
    "heritage": {"category": "Heritage", "mood": "Curious"},
    "cultural": {"category": "Heritage", "mood": "Curious"},
    "photography": {"category": "Scenic", "mood": "Curious"},
    "wildlife": {"category": "Wild", "mood": "Explore"},
    "nature": {"category": "Wild", "mood": "Peaceful"},
    "waterfalls": {"category": "Scenic", "mood": "Explore"},
})


@dataclass
class TourismPreferences:
    district: str | None = None
    destination: str | None = None
    category: str | None = None
    mood: str | None = None
    experience_description: str | None = None
    search_query: str | None = None
    desired_experience: list[str] = field(default_factory=list)
    diverse: bool = False
    category_confirmed: bool = False
    mood_confirmed: bool = False

    @classmethod
    def from_llm_dict(cls, data: dict | None) -> "TourismPreferences":
        if not data:
            return cls()
        exp = data.get("experience_description") or data.get("search_query") or ""
        experiences = data.get("desired_experience") or []
        if isinstance(experiences, str):
            experiences = [experiences]
        return cls(
            district=data.get("district"),
            destination=data.get("destination"),
            category=normalize_category(data.get("category")),
            mood=normalize_mood(data.get("mood")),
            experience_description=exp or None,
            search_query=exp or None,
            desired_experience=list(experiences),
            diverse=bool(data.get("diverse", False)),
            category_confirmed=bool(data.get("category_confirmed", False)),
            mood_confirmed=bool(data.get("mood_confirmed", False)),
        )

    def validated(self) -> "TourismPreferences":
        """Return copy with only valid dataset category/mood values."""
        return TourismPreferences(
            district=self.district,
            destination=self.destination,
            category=normalize_category(self.category),
            mood=normalize_mood(self.mood),
            experience_description=self.experience_description,
            search_query=self.search_query,
            desired_experience=self.desired_experience,
            diverse=self.diverse,
            category_confirmed=self.category_confirmed,
            mood_confirmed=self.mood_confirmed,
        )

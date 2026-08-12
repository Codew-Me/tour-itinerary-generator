"""Geographic helpers for itinerary planning (district centroids + distance estimates)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.data.normalizer import normalize_district, normalize_text

# Approximate district centroids (lat, lng) for Sri Lanka — used for distance *estimates*.
DISTRICT_CENTROIDS: dict[str, tuple[float, float]] = {
    "ampara": (7.291, 81.672),
    "anuradhapura": (8.311, 80.404),
    "badulla": (6.993, 81.055),
    "batticaloa": (7.731, 81.674),
    "colombo": (6.927, 79.861),
    "colombo (boralesgamuwa)": (6.841, 79.918),
    "galle": (6.053, 80.221),
    "gampaha": (7.091, 80.008),
    "hambantota": (6.124, 81.118),
    "hatton": (6.895, 80.595),
    "jaffna": (9.662, 80.025),
    "kalmunai": (7.409, 81.835),
    "kalutara": (6.585, 79.961),
    "kandy": (7.291, 80.636),
    "kegalle": (7.251, 80.346),
    "kegalle (aranayaka)": (7.120, 80.320),
    "kurunegala": (7.486, 80.362),
    "matale": (7.467, 80.623),
    "matara": (5.949, 80.535),
    "puttalam": (8.040, 79.839),
    "ratnapura": (6.682, 80.399),
    "ratnapura (kalthota)": (6.650, 80.450),
    "trincomalee": (8.587, 81.215),
}

# Districts reasonably reachable as day-trip bases from a starting district.
NEIGHBORING_DISTRICTS: dict[str, set[str]] = {
    "hambantota": {"matara", "monaragala", "ampara", "ratnapura", "galle"},
    "galle": {"hambantota", "matara", "kalutara", "colombo"},
    "matara": {"hambantota", "galle"},
    "colombo": {"gampaha", "kalutara", "galle"},
    "kandy": {"matale", "kegalle", "badulla", "kurunegala", "hatton"},
    "trincomalee": {"batticaloa", "ampara", "anuradhapura"},
    "jaffna": {"puttalam"},
    "anuradhapura": {"trincomalee", "kurunegala", "matale", "puttalam"},
    "badulla": {"ampara", "matara", "kandy", "ratnapura"},
    "ampara": {"hambantota", "batticaloa", "badulla", "monaragala"},
}

DEFAULT_MAX_RADIUS_KM = 90.0
ROAD_SPEED_KMH = 45.0  # estimate for rural Sri Lanka

# Localities users may start from — not limited to attraction destinations.
# Values: (display name, district for routing, optional coordinates)
LOCALITY_REGISTRY: dict[str, tuple[str, str | None, tuple[float, float] | None]] = {
    "seeduwa": ("Seeduwa", "Gampaha", (7.119, 79.885)),
    "wattala": ("Wattala", "Gampaha", (6.989, 79.893)),
    "ja ela": ("Ja-Ela", "Gampaha", (7.074, 79.892)),
    "ja-ela": ("Ja-Ela", "Gampaha", (7.074, 79.892)),
    "minuwangoda": ("Minuwangoda", "Gampaha", (7.173, 80.098)),
    "negombo": ("Negombo", "Gampaha", (7.208, 79.836)),
    "dehiwala": ("Dehiwala", "Colombo", (6.856, 79.861)),
    "moratuwa": ("Moratuwa", "Colombo", (6.773, 79.882)),
    "panadura": ("Panadura", "Kalutara", (6.713, 79.903)),
    "ella": ("Ella", "Badulla", (6.866, 81.046)),
    "sigiriya": ("Sigiriya", "Matale", (7.957, 80.760)),
    "nuwara eliya": ("Nuwara Eliya", "Hatton", (6.970, 80.782)),
    "mirissa": ("Mirissa", "Matara", (5.948, 80.471)),
    "weligama": ("Weligama", "Matara", (5.974, 80.430)),
}

_NON_LOCALITY_WORDS = frozenset({
    "solo", "alone", "couple", "friends", "family", "partner", "yes", "no", "yeah",
    "yep", "nope", "ok", "okay", "sure", "thanks", "thank", "hi", "hello", "hey",
    "relaxed", "balanced", "packed", "relax", "slow", "fast", "moderate",
    "adventure", "adventures", "wildlife", "heritage", "nature", "beach", "beaches",
    "wild", "scenic", "pristine", "essence", "thrills", "heritage", "plan", "trip",
    "travel", "days", "day", "night", "nights", "week", "weekend",
})


@dataclass(frozen=True)
class ResolvedLocality:
    name: str
    district: str | None = None
    coordinates: tuple[float, float] | None = None

    def to_coordinates_dict(self) -> dict[str, float] | None:
        if not self.coordinates:
            return None
        return {"lat": self.coordinates[0], "lng": self.coordinates[1]}


def normalize_district_key(name: str | None) -> str:
    if not name:
        return ""
    return normalize_text(normalize_district(name))


def get_district_coords(district: str | None) -> tuple[float, float] | None:
    key = normalize_district_key(district)
    return DISTRICT_CENTROIDS.get(key)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def estimate_travel_minutes(distance_km: float) -> int:
    if distance_km <= 0:
        return 0
    return max(15, int(round((distance_km / ROAD_SPEED_KMH) * 60)))


def distance_between_districts(d1: str | None, d2: str | None) -> float | None:
    c1 = get_district_coords(d1)
    c2 = get_district_coords(d2)
    if not c1 or not c2:
        return None
    return haversine_km(c1[0], c1[1], c2[0], c2[1])


def is_geographically_reasonable(
    starting_district: str | None,
    attraction_district: str | None,
    max_radius_km: float = DEFAULT_MAX_RADIUS_KM,
) -> bool:
    if not starting_district or not attraction_district:
        return True
    start_key = normalize_district_key(starting_district)
    attr_key = normalize_district_key(attraction_district)
    if start_key == attr_key:
        return True
    neighbors = NEIGHBORING_DISTRICTS.get(start_key, set())
    if attr_key in neighbors:
        return True
    dist = distance_between_districts(starting_district, attraction_district)
    if dist is None:
        return attr_key == start_key
    return dist <= max_radius_km


def normalize_locality_name(text: str) -> str:
    """Title-case a user-provided locality (handles hyphenated names like Ja-Ela)."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    if "-" in cleaned:
        return "-".join(part.strip().title() for part in cleaned.split("-") if part.strip())
    return cleaned.title()


def _district_display_name(key: str) -> str:
    if key == "colombo (boralesgamuwa)":
        return "Colombo (Boralesgamuwa)"
    return key.title()


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


def _fuzzy_match_locality(norm: str, *, max_distance: int = 2) -> ResolvedLocality | None:
    """Match minor typos like 'seeuwa' to registered localities such as Seeduwa."""
    if len(norm) < 4:
        max_distance = 1
    best: ResolvedLocality | None = None
    best_dist = max_distance + 1

    for key in LOCALITY_REGISTRY:
        dist = _levenshtein(norm, key)
        if dist < best_dist:
            name, district, coords = LOCALITY_REGISTRY[key]
            best = ResolvedLocality(name=name, district=district, coordinates=coords)
            best_dist = dist

    for key in DISTRICT_CENTROIDS:
        dist = _levenshtein(norm, key)
        if dist < best_dist:
            best = ResolvedLocality(
                name=_district_display_name(key),
                district=_district_display_name(key),
                coordinates=DISTRICT_CENTROIDS[key],
            )
            best_dist = dist

    return best if best_dist <= max_distance else None


def _lookup_locality(key: str) -> ResolvedLocality | None:
    norm = normalize_text(key)
    if not norm:
        return None

    if norm in LOCALITY_REGISTRY:
        name, district, coords = LOCALITY_REGISTRY[norm]
        return ResolvedLocality(name=name, district=district, coordinates=coords)

    if norm in DISTRICT_CENTROIDS:
        return ResolvedLocality(
            name=_district_display_name(norm),
            district=_district_display_name(norm),
            coordinates=DISTRICT_CENTROIDS[norm],
        )

    for reg_key, (name, district, coords) in LOCALITY_REGISTRY.items():
        if norm == normalize_text(name):
            return ResolvedLocality(name=name, district=district, coordinates=coords)

    fuzzy = _fuzzy_match_locality(norm)
    if fuzzy:
        return fuzzy

    return None


def resolve_locality(name: str) -> ResolvedLocality:
    """Resolve a locality name to district/coordinates when known."""
    found = _lookup_locality(name)
    if found:
        return found
    display = normalize_locality_name(name)
    return ResolvedLocality(name=display, district=display)


def resolve_routing_anchor(
    starting_location: str | None,
    district: str | None = None,
) -> tuple[str | None, tuple[float, float] | None]:
    """Resolve a user's start point to a routing district and coordinates."""
    if starting_location:
        loc = resolve_locality(starting_location)
        if loc.coordinates:
            return loc.district or loc.name, loc.coordinates
        if loc.district:
            coords = get_district_coords(loc.district)
            return loc.district, coords
    if district:
        return district, get_district_coords(district)
    if starting_location:
        coords = get_district_coords(starting_location)
        return starting_location, coords
    return None, None


_REJECTION_PHRASES = (
    "dont like", "don't like", "not what i want", "not for me", "hate it", "hate this",
    "dont want", "don't want", "something different", "not what i wanted",
)


def _is_rejection_phrase(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in _REJECTION_PHRASES)


def _looks_like_planning_request(text: str) -> bool:
    """True when input is a trip-planning phrase, not a geographic locality."""
    lower = re.sub(r"\s+", " ", (text or "").lower().strip())
    if not lower:
        return False
    if re.search(r"\bplan\b", lower) and re.search(
        r"\b(tour|trip|travel|itinerary|holiday|vacation|journey)\b", lower
    ):
        return True
    if re.search(r"\b(plan for me|help me plan|start planning|plan my)\b", lower):
        return True
    return False


def _looks_like_locality_input(text: str) -> bool:
    """Heuristic: bare user input plausibly names a town/city."""
    if _is_rejection_phrase(text):
        return False
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) < 2 or len(cleaned) > 64:
        return False
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z\s\-'().]*", cleaned):
        return False
    if re.fullmatch(r"\d+", cleaned):
        return False

    lower = cleaned.lower()
    if lower in _NON_LOCALITY_WORDS:
        return False

    words = re.findall(r"[a-z']+", lower)
    if words and all(w in _NON_LOCALITY_WORDS for w in words):
        return False

    return True


def _match_known_locality_in_text(text: str) -> ResolvedLocality | None:
    lower = text.lower()
    candidates: list[tuple[int, ResolvedLocality]] = []

    for key in LOCALITY_REGISTRY:
        if re.search(rf"\b{re.escape(key)}\b", lower):
            name, district, coords = LOCALITY_REGISTRY[key]
            candidates.append((len(key), ResolvedLocality(name=name, district=district, coordinates=coords)))

    for key in DISTRICT_CENTROIDS:
        if re.search(rf"\b{re.escape(key)}\b", lower):
            candidates.append((
                len(key),
                ResolvedLocality(
                    name=_district_display_name(key),
                    district=_district_display_name(key),
                    coordinates=DISTRICT_CENTROIDS[key],
                ),
            ))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resolve_locality_from_text(text: str, *, allow_bare: bool = False) -> ResolvedLocality | None:
    """Extract and resolve a trip starting locality from natural language.

    When ``allow_bare`` is True (user is answering the start-location question),
    accept any reasonable locality name — it does not need to be an attraction.
    """
    if not text or text.startswith("__start_planning__:"):
        return None

    lower = text.lower()
    patterns = [
        r"(?:starting|start(?:ing)?|strt(?:ing)?)\s+from\s+([a-zA-Z\s\-'()]+?)(?:\.|,|$|\s)",
        r"start(?:ing)?\s+(?:at|in)\s+([a-zA-Z\s\-'()]+?)(?:\.|,|$|\s+for|\s+create|\s+with)",
        r"from\s+([a-zA-Z\s\-'()]+?)\s+(?:for|create|generate|with)",
        r"\bfrom\s+([a-zA-Z\s\-'()]+?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            fragment = m.group(1).strip()
            if len(fragment) >= 2:
                return resolve_locality(fragment)

    known = _match_known_locality_in_text(text)
    if known:
        stripped = text.strip()
        if allow_bare:
            return known
        if "starting" in lower or "start from" in lower or "strt from" in lower or re.search(r"\bfrom\b", lower):
            return known
        if normalize_text(stripped) == normalize_text(known.name):
            return known

    if allow_bare and _looks_like_planning_request(text):
        return None

    if allow_bare and _looks_like_locality_input(text):
        return resolve_locality(text)

    return None


def extract_starting_location(text: str) -> str | None:
    resolved = resolve_locality_from_text(text, allow_bare=False)
    return resolved.name if resolved else None


def is_start_location_correction(text: str, current: str | None) -> bool:
    """True when the user sends a new locality that corrects a prior start location."""
    if not current:
        return False
    resolved = resolve_locality_from_text(text, allow_bare=True)
    if not resolved:
        return False
    if normalize_text(resolved.name) == normalize_text(current):
        return False
    return bool(resolved.coordinates or (resolved.district and resolved.district != resolved.name))

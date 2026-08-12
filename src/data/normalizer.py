"""Destination and district name normalization."""

import re
import unicodedata

# Canonical district spellings
DISTRICT_ALIASES: dict[str, str] = {
    "colombo": "Colombo",
    "kurunagela": "Kurunegala",
    "kurunegala": "Kurunegala",
    "rathnapura": "Ratnapura",
    "ratnapura": "Ratnapura",
    "badulla": "Badulla",
    "galle": "Galle",
    "gampaha": "Gampaha",
    "hambantota": "Hambantota",
    "hatton": "Hatton",
    "kalmunai": "Kalmunai",
    "kalutara": "Kalutara",
    "matale": "Matale",
    "matara": "Matara",
}

# Attraction destination -> review district mapping
DESTINATION_TO_DISTRICT: dict[str, str] = {
    "colombo": "Colombo",
    "colombo (boralesgamuwa)": "Colombo",
    "dehiwala": "Colombo",
    "galle": "Galle",
    "hambantota": "Hambantota",
    "kataragama": "Hambantota",
    "matara": "Matara",
    "gampaha": "Gampaha",
    "kalutara": "Kalutara",
    "matale": "Matale",
    "dambulla": "Matale",
    "badulla": "Badulla",
    "bandarawela": "Badulla",
    "hatton": "Hatton",
    "nuwara eliya": "Hatton",
    "kurunagela": "Kurunegala",
    "kurunegala": "Kurunegala",
    "rathnapura": "Ratnapura",
    "ratnapura": "Ratnapura",
    "ratnapura (kalthota)": "Ratnapura",
    "kalmunai": "Kalmunai",
    "kandy": "Kandy",
    "jaffna": "Jaffna",
    "ampara": "Ampara",
    "anuradhapura": "Anuradhapura",
    "batticaloa": "Batticaloa",
    "trincomalee": "Trincomalee",
    "negombo": "Gampaha",
    "kalpitiya": "Puttalam",
    "kegalle": "Kegalle",
    "kegalle (aranayaka)": "Kegalle",
}

# Manual attraction -> review destination aliases
ATTRACTION_REVIEW_ALIASES: dict[str, list[str]] = {
    "Galle Fort": ["Galle Fort Attractions and Jumpers Sri Lanka", "Black Galle Fort", "Galle Fort Clock Tower"],
    "Sigiriya": ["Sigiriya Rock", "Sigiriya Rock Fortress"],
    "Ella": ["Ella Rock", "Little Adam's Peak"],
    "Independence Square": ["Independence Square"],
    "Mirissa Beach": ["Mirissa Beach"],
}


def normalize_text(value: str | None) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if value is None or (isinstance(value, float)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_district(value: str | None) -> str:
    norm = normalize_text(value)
    return DISTRICT_ALIASES.get(norm, value.strip().title() if value else "")


def normalize_destination(value: str | None) -> str:
    if not value or str(value).strip() in ("", "nan"):
        return ""
    return str(value).strip()


def destination_to_district(destination: str | None) -> str:
    norm = normalize_text(destination)
    if norm in DESTINATION_TO_DISTRICT:
        return DESTINATION_TO_DISTRICT[norm]
    # Fallback: title-case the destination as district guess
    return normalize_district(destination)


def make_document_id(destination: str, district: str, timespan: str, review: str, row_index: int) -> str:
    """Deterministic document ID for ChromaDB idempotency."""
    import hashlib

    payload = f"{normalize_text(destination)}|{normalize_text(district)}|{timespan}|{review}|{row_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

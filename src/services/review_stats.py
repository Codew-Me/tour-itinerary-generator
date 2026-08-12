"""Precomputed review statistics from the reviews dataset (cached in memory)."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED
from src.data.normalizer import normalize_text

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "peaceful": ("peaceful", "calm", "quiet", "serene", "relaxing"),
    "historic": ("historic", "history", "heritage", "ancient", "architecture", "cultural"),
    "scenic": ("beautiful", "scenic", "views", "view", "stunning", "picturesque"),
    "wildlife": ("wildlife", "birds", "bird", "safari", "animals", "elephant"),
    "crowded": ("crowded", "busy", "queue", "lines"),
    "climb": ("climb", "steep", "hike", "walking", "steps"),
    "family": ("family", "kids", "children", "child"),
    "adventure": ("adventure", "adventurous", "explore", "exploring", "exciting", "thrill"),
    "exploration": ("explore", "exploring", "discovery", "discover", "cave", "caves"),
}


@lru_cache(maxsize=1)
def _load_review_index() -> dict[str, dict]:
    """Build destination → stats index from processed reviews CSV."""
    path = DATA_PROCESSED / "reviews_clean.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    df = df.dropna(subset=["Review"])
    df["destination_normalized"] = df["Destination"].astype(str).map(normalize_text)
    df["district_clean"] = df["District"].astype(str).str.strip()

    index: dict[str, dict] = {}
    for dest_norm, group in df.groupby("destination_normalized"):
        if not dest_norm:
            continue
        districts = group["district_clean"].value_counts()
        district = districts.index[0] if len(districts) else ""
        reviews = group["Review"].astype(str).tolist()
        themes = _extract_themes(reviews)
        index[dest_norm] = {
            "destination": group["Destination"].iloc[0],
            "district": district,
            "review_count": len(reviews),
            "themes": themes,
            "summary": _build_summary(themes, len(reviews)),
            "sample_reviews": _pick_sample_reviews(reviews),
        }
    return index


def _format_review_quote(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return cleaned


def _pick_sample_reviews(reviews: list[str], *, max_quotes: int = 2) -> list[str]:
    """Pick readable review excerpts for display on recommendation cards."""
    candidates: list[str] = []
    for raw in reviews:
        quote = _format_review_quote(raw)
        if 50 <= len(quote) <= 320:
            candidates.append(quote)
    candidates.sort(key=len, reverse=True)
    if candidates:
        return candidates[:max_quotes]
    for raw in reviews[:max_quotes]:
        quote = _format_review_quote(raw)
        if quote:
            candidates.append(quote)
    return candidates[:max_quotes]


def _extract_themes(reviews: list[str]) -> Counter:
    text = " ".join(reviews).lower()
    words = re.findall(r"[a-z']+", text)
    word_freq = Counter(words)
    theme_counts: Counter = Counter()
    for theme, keywords in THEME_KEYWORDS.items():
        theme_counts[theme] = sum(word_freq.get(kw, 0) for kw in keywords)
    return theme_counts


def _build_summary(themes: Counter, review_count: int) -> str | None:
    if review_count < 3:
        return None
    top = [t for t, c in themes.most_common(3) if c >= 2]
    if not top:
        return None
    phrases = {
        "peaceful": "peaceful atmosphere",
        "historic": "historic atmosphere and architecture",
        "scenic": "beautiful scenery and views",
        "wildlife": "wildlife and nature experiences",
        "family": "family-friendly experiences",
        "climb": "a climb or walk that visitors mention",
        "crowded": "crowds at busy times",
        "adventure": "adventurous or exploratory experiences",
        "exploration": "exploration and discovery",
    }
    parts = [phrases[t] for t in top if t in phrases]
    if not parts:
        return None
    if len(parts) == 1:
        return f"Visitors often mention {parts[0]}."
    return f"Visitors often mention {', '.join(parts[:-1])}, and {parts[-1]}."


def get_destination_review_stats(destination: str | None) -> dict | None:
    if not destination:
        return None
    return _load_review_index().get(normalize_text(destination))


def resolve_district_from_reviews(destination: str | None, fallback: str | None = None) -> str:
    """Prefer review-dataset district; fall back to attraction destination/district."""
    stats = get_destination_review_stats(destination)
    if stats and stats.get("district"):
        return stats["district"]
    return fallback or destination or "—"


def enrich_attraction(attraction: dict) -> dict:
    """Attach review-derived district + summary to an attraction dict."""
    out = dict(attraction)
    name = out.get("name")
    dest = out.get("destination")
    stats = get_destination_review_stats(name)
    if not stats and dest:
        stats = get_destination_review_stats(dest)
    out["display_district"] = resolve_district_from_reviews(
        name or dest,
        out.get("district") or dest,
    )
    if stats:
        out["review_count"] = stats.get("review_count", out.get("review_count", 0))
        out["review_summary"] = stats.get("summary")
        out["review_samples"] = stats.get("sample_reviews") or []
        out["review_themes"] = dict(stats.get("themes", {}))
    else:
        out["review_summary"] = None
        out["review_samples"] = []
    return out

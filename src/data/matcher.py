"""Link attractions to review destinations."""

from __future__ import annotations

import pandas as pd

from src.data.normalizer import ATTRACTION_REVIEW_ALIASES, normalize_text


def _substring_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or a in b or b in a


def build_review_destination_index(reviews: pd.DataFrame) -> dict[str, str]:
    """Map normalized review destination -> canonical review destination name."""
    index: dict[str, str] = {}
    for dest in reviews["Destination"].unique():
        index[normalize_text(dest)] = dest
    return index


def match_attraction_to_reviews(
    attraction_name: str,
    reviews: pd.DataFrame,
    review_index: dict[str, str],
) -> tuple[list[str], str]:
    """
    Return matched review destination names and match_type.
    match_type: exact | normalized | alias | substring | none
    """
    name_norm = normalize_text(attraction_name)
    matched: set[str] = set()

    # Exact
    if attraction_name in reviews["Destination"].values:
        matched.add(attraction_name)
        return list(matched), "exact"

    # Normalized exact
    if name_norm in review_index:
        matched.add(review_index[name_norm])
        return list(matched), "normalized"

    # Manual aliases
    aliases = ATTRACTION_REVIEW_ALIASES.get(attraction_name, [])
    for alias in aliases:
        if alias in reviews["Destination"].values:
            matched.add(alias)
    if matched:
        return list(matched), "alias"

    # Substring
    for norm, canonical in review_index.items():
        if _substring_match(name_norm, norm):
            matched.add(canonical)

    if matched:
        return list(matched), "substring"

    return [], "none"


def build_attraction_review_links(
    attractions: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    review_index = build_review_destination_index(reviews)
    rows = []
    for _, row in attractions.iterrows():
        name = row["Attraction Name"]
        matched_dests, match_type = match_attraction_to_reviews(name, reviews, review_index)
        review_count = 0
        if matched_dests:
            review_count = int(reviews[reviews["Destination"].isin(matched_dests)].shape[0])

        rows.append(
            {
                "attraction_name": name,
                "matched_review_destinations": matched_dests,
                "match_type": match_type,
                "review_count": review_count,
                "review_evidence_available": review_count > 0,
            }
        )
    return pd.DataFrame(rows)

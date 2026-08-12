"""Phase 1: Dataset inspection and cross-linkage analysis."""
import re
from pathlib import Path

import pandas as pd

base = Path(__file__).resolve().parent.parent
reviews_path = base / "Destination Reviews (final).csv"
attractions_path = base / "attractions.xlsx"


def normalize(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    reviews = pd.read_csv(reviews_path, encoding="utf-8")
    attractions = pd.read_excel(attractions_path, sheet_name="Attractions")

    review_dests = set(reviews["Destination"].unique())
    attr_names = set(attractions["Attraction Name"].unique())
    attr_dests = set(attractions["Destination"].dropna().unique())

    print("=" * 80)
    print("CROSS-DATASET LINKAGE ANALYSIS")
    print("=" * 80)

    exact_name_match = attr_names & review_dests
    print(f"\nAttractions with exact name match in reviews: {len(exact_name_match)} / {len(attr_names)}")

    norm_name_match = set()
    review_dests_norm = {normalize(d): d for d in review_dests}
    for norm, name in {normalize(n): n for n in attr_names}.items():
        if norm in review_dests_norm:
            norm_name_match.add(name)
    print(f"Attractions with normalized name match: {len(norm_name_match)} / {len(attr_names)}")

    exact_dest_match = attr_dests & review_dests
    print(f"Attraction destinations with exact match in reviews: {len(exact_dest_match)} / {len(attr_dests)}")

    partial_matches = []
    for aname in attr_names:
        anorm = normalize(aname)
        for rdest in review_dests:
            rnorm = normalize(rdest)
            if anorm == rnorm or anorm in rnorm or rnorm in anorm:
                partial_matches.append((aname, rdest))
                break

    unmatched = attr_names - {m[0] for m in partial_matches}
    print(f"Attractions with partial/substring match: {len(partial_matches)}")
    print(f"Attractions with NO partial match: {len(unmatched)}")

    review_only = review_dests - attr_names
    print(f"\nReview destinations NOT in attraction names: {len(review_only)}")
    rc = reviews.groupby("Destination").size().sort_values(ascending=False)
    print("Top 20 review-only destinations by count:")
    shown = 0
    for d in rc.index:
        if d not in attr_names:
            print(f"  {d}: {rc[d]} reviews")
            shown += 1
            if shown >= 20:
                break

    print("\n--- District Name Inconsistencies ---")
    print("Review districts:", sorted(reviews["District"].unique()))
    print("Attraction destinations:", sorted(attr_dests))

    print("\n--- Attraction Categories ---")
    print(attractions["Category"].value_counts().to_string())
    print("\n--- Attraction Moods ---")
    print(attractions["mood_tag"].value_counts().to_string())

    img = attractions["Image"]
    base64_count = img.astype(str).str.startswith("data:image").sum()
    url_count = img.astype(str).str.startswith("http").sum()
    print("\n--- Image Field Analysis ---")
    print(f"  Base64 embedded: {base64_count}")
    print(f"  URL images: {url_count}")
    print(f"  Missing/empty: {attractions['Image'].isna().sum()}")

    print("\n" + "=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    print(f"Total reviews: {len(reviews):,}")
    print(f"Total attractions: {len(attractions):,}")
    print(f"Unique review destinations: {reviews['Destination'].nunique():,}")
    print(f"Unique review districts: {reviews['District'].nunique():,}")
    print(f"Unique attraction destinations: {attractions['Destination'].nunique():,}")
    print(f"Attractions with exact review name match: {len(exact_name_match)}")
    print(f"Attractions with partial/substring match: {len(partial_matches)}")
    print(f"Attractions with NO review match: {len(unmatched)}")
    print(f"Review-only destinations: {len(review_only)}")

    # Destination name inconsistency examples
    print("\n--- Destination Name Inconsistency Examples ---")
    examples = [
        ("colombo", "Colombo"),
        ("Kurunagela", "Kurunegala"),
        ("Rathnapura", "Ratnapura"),
    ]
    for rev, expected in examples:
        count = (reviews["District"] == rev).sum() if rev in reviews["District"].values else 0
        print(f"  Review district '{rev}': {count} reviews (expected spelling: {expected})")

    # Save report
    report_dir = base / "data" / "processed"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "reviews_total": len(reviews),
        "attractions_total": len(attractions),
        "review_destinations_unique": int(reviews["Destination"].nunique()),
        "review_districts_unique": int(reviews["District"].nunique()),
        "attraction_destinations_unique": int(attractions["Destination"].nunique()),
        "attractions_exact_name_match": len(exact_name_match),
        "attractions_partial_match": len(partial_matches),
        "attractions_no_review_match": len(unmatched),
        "review_only_destinations": len(review_only),
        "categories": attractions["Category"].value_counts().to_dict(),
        "moods": attractions["mood_tag"].value_counts().to_dict(),
    }
    pd.Series(summary).to_json(report_dir / "phase1_summary.json", indent=2)
    print(f"\nReport saved to {report_dir / 'phase1_summary.json'}")


if __name__ == "__main__":
    main()

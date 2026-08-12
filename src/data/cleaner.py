"""Data cleaning for reviews and attractions."""

import pandas as pd

from src.data.normalizer import normalize_destination, normalize_district, normalize_text


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = {"Destination", "District", "Timespan", "Review"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reviews missing columns: {missing}")

    for col in required:
        df[col] = df[col].astype(str).str.strip()

    # Remove empty reviews
    df = df[df["Review"].str.len() > 0]

    # Normalize whitespace in reviews
    df["Review"] = df["Review"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Remove exact duplicate destination+review pairs (keep first)
    df = df.drop_duplicates(subset=["Destination", "Review"], keep="first")

    df["district_normalized"] = df["District"].apply(normalize_district)
    df["destination_normalized"] = df["Destination"].apply(normalize_text)
    df = df.reset_index(drop=True)
    df["row_index"] = df.index
    return df


def clean_attractions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename = {"mood_tag": "Mood", "Attraction Name": "Attraction Name"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["Attraction Name"] = df["Attraction Name"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Mood"] = df["Mood"].astype(str).str.strip()
    df["Details"] = df["Details"].astype(str).str.strip()

    # Fill missing destination from name heuristics where possible
    mask = df["Destination"].isna() | df["Destination"].astype(str).str.strip().isin(["", "nan"])
    df.loc[mask, "Destination"] = "Colombo"  # Gangarama Temple case

    df["Destination"] = df["Destination"].apply(normalize_destination)
    df["destination_normalized"] = df["Destination"].apply(normalize_text)
    df["name_normalized"] = df["Attraction Name"].apply(normalize_text)
    df["district_normalized"] = df["Destination"].apply(
        lambda d: normalize_district(d) if d else ""
    )

    # Clean image field
    df["Image"] = df["Image"].where(
        df["Image"].notna() & ~df["Image"].astype(str).str.strip().isin(["", "nan"]),
        None,
    )
    return df.reset_index(drop=True)

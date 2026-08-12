"""Load raw datasets."""

from pathlib import Path

import pandas as pd

from src.config import get_settings


def load_raw_reviews(path: str | None = None) -> pd.DataFrame:
    settings = get_settings()
    csv_path = Path(path or settings.reviews_csv)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read reviews CSV: {csv_path}")


def load_raw_attractions(path: str | None = None) -> pd.DataFrame:
    settings = get_settings()
    xlsx_path = Path(path or settings.attractions_xlsx)
    return pd.read_excel(xlsx_path, sheet_name="Attractions")

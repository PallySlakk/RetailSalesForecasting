
import pandas as pd
from config import (
    KAGGLE_STORE_SALES_PATH,
    NOAA_NORMALS_DAILY_PATH,
    DATE_COLUMN,
)


def load_kaggle_store_sales() -> pd.DataFrame:
    """Load Kaggle Store Sales dataset.

    Assumes the CSV has a 'date' column and standard Kaggle schema.
    """
    df = pd.read_csv(KAGGLE_STORE_SALES_PATH, parse_dates=[DATE_COLUMN])
    return df


def load_noaa_normals_daily() -> pd.DataFrame:
    """Load cleaned Ecuador GHCN-Daily weather.

    Expected columns (before normalization):
        ['station', 'date', 'prcp', 'snow', 'tmax', 'tmin', 'tavg', ...]

    This loader:
    - Reads the CSV
    - Lowercases column names
    - Ensures 'station' and 'date' exist
    """
    import os
    print(f"\n📄 Loading NOAA weather from: {NOAA_NORMALS_DAILY_PATH}")
    if not NOAA_NORMALS_DAILY_PATH.exists():
        raise FileNotFoundError(
            f"Weather file not found at {NOAA_NORMALS_DAILY_PATH}. "
            "Make sure 'ecuador_weather_daily.csv' is in data/raw/.")

    df = pd.read_csv(NOAA_NORMALS_DAILY_PATH)

    # Normalize columns to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Sanity check
    if "station" not in df.columns:
        raise KeyError(
            f"'station' column missing in weather file: {NOAA_NORMALS_DAILY_PATH}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    if "date" not in df.columns:
        raise KeyError(
            f"'date' column missing in weather file: {NOAA_NORMALS_DAILY_PATH}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df

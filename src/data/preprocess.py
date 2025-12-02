
import pandas as pd
from config import DATE_COLUMN, STORE_COLUMN, FAMILY_COLUMN, TARGET_COLUMN


def clean_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize Ecuador weather data.

    - Lowercase columns
    - Ensure station + date
    - Ensure prcp, tmax, tmin, tavg exist
    - Compute tavg if missing
    - Impute missing values per station
    """
    df = weather_df.copy()

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if "station" not in df.columns:
        raise KeyError("Weather CSV must contain a 'station' column.")
    if DATE_COLUMN not in df.columns:
        raise KeyError(f"Weather CSV must contain a '{DATE_COLUMN}' column.")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])

    # Ensure weather variables exist
    for col in ["prcp", "tmax", "tmin", "tavg"]:
        if col not in df.columns:
            df[col] = None

    # Compute tavg if missing
    if df["tavg"].isna().all() and ("tmax" in df.columns and "tmin" in df.columns):
        df["tavg"] = df[["tmax", "tmin"]].mean(axis=1)

    # Impute missing weather per station
    for col in ["prcp", "tmax", "tmin", "tavg"]:
        df[col] = df.groupby("station")[col].transform(lambda s: s.ffill().bfill())
        df[col] = df.groupby("station")[col].transform(lambda s: s.fillna(s.median()))

    # Clip precipitation to non-negative
    df["prcp"] = df["prcp"].clip(lower=0)

    return df[["station", DATE_COLUMN, "prcp", "tmax", "tmin", "tavg"]]


def clean_and_merge_data(sales_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Clean Kaggle sales data and merge with Ecuador weather.

    - Clean weather
    - Parse sales dates
    - Assign a single weather station to all stores (simple assumption)
    - Aggregate sales to daily store-family level
    - Merge on [date, station]
    """
    sales = sales_df.copy()
    weather = clean_weather(weather_df)

    # Ensure sales date
    sales[DATE_COLUMN] = pd.to_datetime(sales[DATE_COLUMN], errors="coerce")
    sales = sales.dropna(subset=[DATE_COLUMN])

    # TEMP: assign same station to all stores
    station_id = weather["station"].unique()[0]
    sales["station"] = station_id

    # Aggregate sales
    agg_cols = [DATE_COLUMN, STORE_COLUMN, FAMILY_COLUMN, "station"]
    sales = (
        sales.groupby(agg_cols, as_index=False)
        .agg({TARGET_COLUMN: "sum"})
    )

    # Merge on (date, station)
    merged = pd.merge(
        sales,
        weather,
        on=[DATE_COLUMN, "station"],
        how="left"
    )

    # Remove missing target
    merged = merged.dropna(subset=[TARGET_COLUMN])

    return merged

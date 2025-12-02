
import pandas as pd
from config import (
    DATE_COLUMN,
    STORE_COLUMN,
    FAMILY_COLUMN,
    TARGET_COLUMN,
    LAG_DAYS,
    ROLLING_WINDOWS,
)


def build_feature_matrix(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Construct supervised learning feature matrix from merged sales+weather.

    Features:
    - Calendar features: year, month, day, dayofweek, weekofyear, is_weekend
    - Weather: prcp, tmax, tmin, tavg, temp_x_prcp
    - Lagged sales: LAG_DAYS
    - Rolling means: ROLLING_WINDOWS
    """
    df = merged_df.copy()
    df = df.sort_values([STORE_COLUMN, FAMILY_COLUMN, DATE_COLUMN])

    # Basic calendar features
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df["year"] = df[DATE_COLUMN].dt.year.astype("int32")
    df["month"] = df[DATE_COLUMN].dt.month.astype("int32")
    df["day"] = df[DATE_COLUMN].dt.day.astype("int32")
    df["dayofweek"] = df[DATE_COLUMN].dt.dayofweek.astype("int32")
    df["weekofyear"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int64")

    # Interaction feature
    if {"tavg", "prcp"}.issubset(df.columns):
        df["temp_x_prcp"] = df["tavg"] * df["prcp"]
    else:
        df["temp_x_prcp"] = 0.0

    # Lag and rolling features per store-family
    group_cols = [STORE_COLUMN, FAMILY_COLUMN]
    df = df.set_index(DATE_COLUMN)

    for lag in LAG_DAYS:
        df[f"sales_lag_{lag}"] = (
            df.groupby(group_cols)[TARGET_COLUMN]
            .shift(lag)
        )

    for window in ROLLING_WINDOWS:
        df[f"sales_roll_mean_{window}"] = (
            df.groupby(group_cols)[TARGET_COLUMN]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    df = df.reset_index()

    # Drop any rows that still have NaN in target
    df = df.dropna(subset=[TARGET_COLUMN])

    return df

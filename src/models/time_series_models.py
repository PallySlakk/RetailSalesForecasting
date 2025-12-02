
"""Optional time series models (Prophet, ARIMA) for store-level aggregates."""
from typing import Optional

import pandas as pd

from config import DATE_COLUMN, TARGET_COLUMN, STORE_COLUMN


def prepare_univariate_series(
    df: pd.DataFrame,
    store_nbr: Optional[int] = None,
) -> pd.DataFrame:
    data = df.copy()
    if store_nbr is not None:
        data = data[data[STORE_COLUMN] == store_nbr]

    series = (
        data.groupby(DATE_COLUMN, as_index=False)[TARGET_COLUMN]
        .sum()
        .rename(columns={DATE_COLUMN: "ds", TARGET_COLUMN: "y"})
    )
    return series


def fit_prophet(series: pd.DataFrame):
    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet  # type: ignore
        except ImportError:
            raise ImportError(
                "Neither 'prophet' nor 'fbprophet' is installed. "
                "Install one of them to use Prophet models."
            )

    m = Prophet()
    m.fit(series)
    return m


def fit_arima(series: pd.DataFrame):
    try:
        import pmdarima as pm
    except ImportError as exc:
        raise ImportError(
            "pmdarima is not installed. Install it to use ARIMA models."
        ) from exc

    model = pm.auto_arima(
        series["y"],
        seasonal=False,
        error_action="ignore",
        suppress_warnings=True,
    )
    return model

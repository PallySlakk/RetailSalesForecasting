import sys
from pathlib import Path

# Ensure project root is in PATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import json
import pandas as pd
import streamlit as st
from joblib import load

from config import (
    PROJECT_ROOT,
    METRICS_DIR,
    FEATURE_DATA_PATH,
    MERGED_DATA_PATH,
    MODELS_DIR,
    DATE_COLUMN,
    STORE_COLUMN,
    FAMILY_COLUMN,
    TARGET_COLUMN,
)

# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

def load_metrics():
    path = METRICS_DIR / "all_metrics.json"
    if path.exists():
        return json.load(open(path))
    return {}

def load_val_predictions():
    path = METRICS_DIR / "val_predictions.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=[DATE_COLUMN])
    return pd.DataFrame()

def load_features():
    if FEATURE_DATA_PATH.exists():
        return pd.read_parquet(FEATURE_DATA_PATH)
    return pd.DataFrame()

def load_merged():
    if MERGED_DATA_PATH.exists():
        return pd.read_parquet(MERGED_DATA_PATH)
    return pd.DataFrame()

def load_models():
    models = {}
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob("*.joblib"):
            try:
                models[p.stem] = load(p)
            except:
                pass
    return models


# ---------------------------------------------------------
# Main App
# ---------------------------------------------------------

def main():

    st.set_page_config(page_title="Retail Forecast Dashboard", layout="wide")
    st.title("Retail Sales Forecasting Dashboard")
    st.caption("Kaggle Store Sales + Weather (GHCN-D)")

    metrics = load_metrics()
    preds_df = load_val_predictions()
    features_df = load_features()
    merged_df = load_merged()
    models = load_models()

    tab_overview, tab_validation, tab_weather, tab_whatif = st.tabs(
        ["Overview", "Validation Explorer", "Weather & Sales", "What-if Simulator"]
    )

    # ---------------------------------------------------------
    # Overview Tab
    # ---------------------------------------------------------
    with tab_overview:
        st.subheader("Model Performance Summary")

        if not metrics:
            st.warning("No metrics found. Run `python main.py` to generate them.")
        else:
            metrics_df = (
                pd.DataFrame(metrics)
                .T.reset_index()
                .rename(columns={"index": "model"})
            )
            st.dataframe(metrics_df, use_container_width=True)

        if not merged_df.empty:
            st.subheader("Sample of Merged Data")
            st.dataframe(merged_df.head(25), use_container_width=True)

    # ---------------------------------------------------------
    # VALIDATION TAB
    # ---------------------------------------------------------
    with tab_validation:
        st.subheader("Actual vs Predicted Sales (Validation Period)")

        if preds_df.empty:
            st.info("Validation predictions not found. Run pipeline first.")
        else:

            st.sidebar.header("Filters")

            model_sel = st.sidebar.selectbox("Model", sorted(preds_df["model"].unique()))
            store_sel = st.sidebar.selectbox("Store", sorted(preds_df[STORE_COLUMN].unique()))
            family_sel = st.sidebar.selectbox("Family", sorted(preds_df[FAMILY_COLUMN].unique()))

            filtered = preds_df[
                (preds_df["model"] == model_sel) &
                (preds_df[STORE_COLUMN] == store_sel) &
                (preds_df[FAMILY_COLUMN] == family_sel)
            ].sort_values(DATE_COLUMN)

            if filtered.empty:
                st.warning("No data for this combination.")
            else:
                st.markdown(f"**Store:** {store_sel} | **Family:** {family_sel} | **Model:** {model_sel}")

                chart = filtered[[DATE_COLUMN, TARGET_COLUMN, "y_pred"]].set_index(DATE_COLUMN)
                st.line_chart(chart)

                st.subheader("Sample Records")
                st.dataframe(filtered.head(100), use_container_width=True)

    # ---------------------------------------------------------
    # WEATHER TAB
    # ---------------------------------------------------------
    with tab_weather:
        st.subheader("Weather vs Sales")

        if merged_df.empty:
            st.info("Merged data not available.")
        else:
            store_sel = st.selectbox("Store", sorted(merged_df[STORE_COLUMN].unique()), key="wx_store")
            family_sel = st.selectbox("Family", sorted(merged_df[FAMILY_COLUMN].unique()), key="wx_family")

            subset = merged_df[
                (merged_df[STORE_COLUMN] == store_sel) &
                (merged_df[FAMILY_COLUMN] == family_sel)
            ].sort_values(DATE_COLUMN)

            if subset.empty:
                st.warning("No weather/sales records for this selection.")
            else:
                st.line_chart(subset.set_index(DATE_COLUMN)[[TARGET_COLUMN, "tavg", "prcp"]])

    # ---------------------------------------------------------
    # WHAT-IF SIMULATOR TAB
    # ---------------------------------------------------------
    with tab_whatif:
        st.subheader("What-if Simulator: Predict Sales Under New Weather Conditions")

        if features_df.empty or not models:
            st.info("Features/models missing. Run pipeline first.")
        else:
            model_name = st.selectbox("Model", sorted(models.keys()), key="sim_model")
            model = models[model_name]

            store_sel = st.selectbox("Store", sorted(features_df[STORE_COLUMN].unique()), key="sim_store")
            family_sel = st.selectbox("Family", sorted(features_df[FAMILY_COLUMN].unique()), key="sim_family")

            # MOST RECENT RECORD
            base = (
                features_df[
                    (features_df[STORE_COLUMN] == store_sel)
                    & (features_df[FAMILY_COLUMN] == family_sel)
                ]
                .sort_values(DATE_COLUMN)
                .tail(1)
            )

            if base.empty:
                st.warning("No feature rows for this store/family.")
            else:
                # Ensure fallbacks
                def safe_val(df, col, default):
                    if col in df.columns and not pd.isna(df[col].iloc[0]):
                        return float(df[col].iloc[0])
                    return default

                tavg0 = safe_val(base, "tavg", 20.0)
                prcp0 = safe_val(base, "prcp", 0.0)
                tmax0 = tavg0 + 3
                tmin0 = tavg0 - 3

                col1, col2 = st.columns(2)

                with col1:
                    tavg_new = st.slider(
                        "Average Temperature (°C)",
                        min_value=0.0,
                        max_value=40.0,
                        value=float(round(tavg0, 1)),
                        step=0.5,
                    )

                with col2:
                    prcp_new = st.slider(
                        "Daily Rainfall (mm)",
                        min_value=0.0,
                        max_value=200.0,
                        value=float(round(prcp0, 1)),
                        step=1.0,
                    )

                # Build modified row
                sim = base.copy()
                sim["tavg"] = tavg_new
                sim["tmax"] = tavg_new + 3
                sim["tmin"] = tavg_new - 3
                sim["prcp"] = prcp_new
                sim["temp_x_prcp"] = sim["tavg"] * sim["prcp"]

                # Predict
                y_pred = float(model.predict(sim)[0])
                y_base = float(base[TARGET_COLUMN].iloc[0])

                st.markdown("### Prediction Result")
                st.write(f"Baseline recent sales: **{y_base:,.2f}**")
                st.write(f"Predicted sales under new weather: **{y_pred:,.2f}**")

                diff = y_pred - y_base
                if diff >= 0:
                    st.success(f"Change vs baseline: +{diff:,.2f}")
                else:
                    st.error(f"Change vs baseline: {diff:,.2f}")


if __name__ == "__main__":
    main()

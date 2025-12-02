# Retail Sales Forecasting Using Machine Learning and Weather Data

This project implements the practicum proposal for **Retail Sales Forecasting Using Machine Learning Techniques**. It focuses on an end-to-end data science workflow using:

1. **Primary Retail Dataset (Kaggle)**  
   - **Name:** Store Sales — Time Series Forecasting  
   - **Source:** Kaggle Competition  
   - **Link:** https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data  
   - **Expected file (for this project):** `data/raw/kaggle_store_sales.csv`  
   - Contains columns such as:
     - `id`
     - `date`
     - `store_nbr`
     - `family`
     - `sales`
     - `onpromotion`
     - (plus additional metadata tables from the competition if you choose to integrate them)

2. **External Weather Dataset (NOAA CDO – Normals Daily)**  
   - **Source:** NOAA Climate Data Online (CDO)  
   - **Entry point:** https://www.ncdc.noaa.gov/cdo-web/datasets  
   - **Intended dataset:** Normals Daily (e.g., daily temperature, precipitation, etc.)  
   - **Expected file (for this project):** `data/raw/noaa_normals_daily.csv`  
   - Typical columns include:
     - `DATE`
     - `STATION` or `NAME`
     - Daily normal temperature & precipitation fields (e.g., `dly-tmax-normal`, `dly-tmin-normal`, `dly-prcp-normal`)

Both datasets are **public, anonymized, and properly cited**, matching the practicum proposal requirements.

---

## 1. Project Objectives

- Forecast retail sales at the **store-family** daily level.
- Integrate **external weather signals** from NOAA CDO.
- Build multiple forecasting models:
  - Traditional ML: **Linear Regression**, **Random Forest**, **XGBoost**
  - Time series: **Prophet**, **ARIMA** (via `pmdarima`)
- Evaluate with **RMSE**, **MAPE**, and **MAE**.
- Visualize results and insights through an interactive **Streamlit dashboard**.

The project primarily emphasizes core data science skills (data acquisition, cleaning, feature engineering, modeling, and evaluation). The dashboard is included as an additional visualization layer.

---

## 2. Project Structure

```text
retail_sales_forecasting_project/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/
│   ├── raw/
│   │   ├── kaggle_store_sales.csv   # from Kaggle Store Sales competition
│   │   └── noaa_normals_daily.csv   # from NOAA CDO Normals Daily
│   └── processed/
│       ├── merged_sales_weather.parquet
│       └── features.parquet
├── notebooks/
│   └── 01_eda.ipynb            # EDA and sanity checks
├── reports/
│   ├── figures/
│   └── metrics/
│       ├── all_metrics.json
│       ├── linear_regression_metrics.json
│       ├── random_forest_metrics.json
│       ├── xgboost_metrics.json
│       └── val_predictions.csv  # actual vs predicted for dashboard
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── load_data.py
    │   └── preprocess.py
    ├── features/
    │   ├── __init__.py
    │   └── build_features.py
    └── models/
        ├── __init__.py
        ├── train_models.py
        ├── time_series_models.py
        └── evaluate.py
```

---

## 3. Setup Instructions

1. **Create environment and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Place the data files**

   - Download **Kaggle Store Sales — Time Series Forecasting** `train.csv` (and optionally merged tables)  
     Rename or export the main daily store-family sales file to:  
     - `data/raw/kaggle_store_sales.csv`
   - Download **NOAA CDO Normals Daily** CSV(s) for the regions corresponding to your stores:  
     - `data/raw/noaa_normals_daily.csv`

   If your column names differ, adjust them in:
   - `src/data/load_data.py`
   - `src/data/preprocess.py`
   - `config.py`

3. **Run the full pipeline**

   ```bash
   python main.py
   ```

   This will:
   - Load Kaggle and NOAA data
   - Clean and merge them on date + store
   - Engineer calendar, lag, and weather interaction features
   - Train **Linear Regression**, **Random Forest**, and **XGBoost** models
   - Evaluate models (RMSE, MAE, MAPE)
   - Save metrics & validation predictions to `reports/metrics/`

4. **Launch the dashboard**

   ```bash
   streamlit run dashboard/app.py
   ```

   The dashboard reads:
   - `reports/metrics/all_metrics.json`
   - `reports/metrics/val_predictions.csv`

   And provides:
   - Metrics summary table
   - Model comparison
   - Interactive plots of **Actual vs Predicted** sales by:
     - Store
     - Family
     - Date

---

## 4. Configuration

Global configuration is centralized in `config.py`:

- Paths (project, data, reports)
- Column names (`TARGET_COLUMN`, `DATE_COLUMN`, `STORE_COLUMN`, `FAMILY_COLUMN`)
- Train/validation cutoff dates
- Lag/rolling windows
- Default frequency (`D` for daily)

Update this file if:
- You change dataset file names.
- You adjust time windows.
- You focus on a subset of stores/families.

---

## 5. Models

**Core supervised models (panel / tabular):**

- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

**Optional univariate time series models:**

Implemented in `src/models/time_series_models.py` to support:
- Prophet
- ARIMA (via `pmdarima`)

These are demonstrated on **store-level aggregated** series and can be extended as needed for your final report.

---

## 6. Ethical & Practical Notes

- All data is public and anonymized (Kaggle & NOAA).
- Cite sources as in the practicum proposal.
- Use model explainability (e.g., feature importance) where needed to support interpretation.
- Results should be presented with uncertainty and limitations.

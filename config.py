from pathlib import Path

# -------------------------------------------------------------------
# Detect project root (works locally AND inside Streamlit Cloud)
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# If config.py is inside the dashboard folder in Streamlit Cloud,
# move up one level to the actual project root.
if PROJECT_ROOT.name == "dashboard":
    PROJECT_ROOT = PROJECT_ROOT.parent


# -------------------------------------------------------------------
# Directory structure
# -------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"


# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------

# Raw datasets (must exist for Streamlit Cloud)
KAGGLE_STORE_SALES_PATH = RAW_DATA_DIR / "kaggle_store_sales.csv"
NOAA_NORMALS_DAILY_PATH = RAW_DATA_DIR / "ecuador_weather_daily.csv"

# Processed data created by main.py
MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_sales_weather.parquet"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "features.parquet"

# (Optional fallback for Streamlit Cloud if parquet fails)
MERGED_DATA_CSV = PROCESSED_DATA_DIR / "merged_sales_weather.csv"
FEATURE_DATA_CSV = PROCESSED_DATA_DIR / "features.csv"


# -------------------------------------------------------------------
# Core column names from Kaggle dataset
# -------------------------------------------------------------------

TARGET_COLUMN = "sales"
DATE_COLUMN = "date"
STORE_COLUMN = "store_nbr"
FAMILY_COLUMN = "family"


# -------------------------------------------------------------------
# Time configuration
# -------------------------------------------------------------------

FREQUENCY = "D"  # daily frequency

TRAIN_END_DATE = "2016-12-31"
VAL_END_DATE = "2017-08-31"


# -------------------------------------------------------------------
# Feature engineering settings
# -------------------------------------------------------------------

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28, 56]

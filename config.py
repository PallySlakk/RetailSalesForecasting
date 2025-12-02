
from pathlib import Path

# Root directory (assumes config.py is in project root)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data files
# Kaggle Store Sales CSV should be downloaded separately and placed here.
KAGGLE_STORE_SALES_PATH = RAW_DATA_DIR / "kaggle_store_sales.csv"

# Cleaned Ecuador GHCN-Daily weather (already preprocessed)
NOAA_NORMALS_DAILY_PATH = RAW_DATA_DIR / "ecuador_weather_daily.csv"

# Processed files
MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_sales_weather.parquet"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "features.parquet"

# Reports / outputs
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

# Column names (aligned with Kaggle Store Sales competition)
TARGET_COLUMN = "sales"
DATE_COLUMN = "date"
STORE_COLUMN = "store_nbr"
FAMILY_COLUMN = "family"

# Time configuration
FREQUENCY = "D"  # daily

# Train / validation cutoffs (example values, adjust to your horizon)
TRAIN_END_DATE = "2016-12-31"
VAL_END_DATE = "2017-08-31"

# Feature engineering configuration
LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28, 56]

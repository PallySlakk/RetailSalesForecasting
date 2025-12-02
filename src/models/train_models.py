from typing import Dict, Tuple, List
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from category_encoders.hashing import HashingEncoder   # IMPORTANT
from config import DATE_COLUMN, TARGET_COLUMN, TRAIN_END_DATE, VAL_END_DATE


# ------------------------------------------------------------------------------
# Train/Validation Split
# ------------------------------------------------------------------------------

def _train_val_split(df: pd.DataFrame):
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    train = df[df[DATE_COLUMN] <= TRAIN_END_DATE].copy()
    val = df[(df[DATE_COLUMN] > TRAIN_END_DATE) & (df[DATE_COLUMN] <= VAL_END_DATE)].copy()

    if train.empty or val.empty:
        raise ValueError("Train/validation split is empty. Check TRAIN_END_DATE/VAL_END_DATE.")

    return train, val


# ------------------------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------------------------

def _select_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    excluded = {TARGET_COLUMN, DATE_COLUMN}

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded
    ]

    categorical_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and c not in excluded
    ]

    return numeric_cols, categorical_cols


# ------------------------------------------------------------------------------
# Full Model Trainer
# ------------------------------------------------------------------------------

def train_all_models(features_df: pd.DataFrame):
    print("\n🔧 Preparing train/validation split...")
    train_df, val_df = _train_val_split(features_df)

    print("🔧 Selecting features...")
    numeric_cols, categorical_cols = _select_feature_columns(train_df)

    print(f"📌 Numeric features: {numeric_cols}")
    print(f"📌 Categorical features: {categorical_cols}")

    y_train = train_df[TARGET_COLUMN]
    y_val = val_df[TARGET_COLUMN]

    # --------------------------------------------------------------------------
    # Preprocessing: numeric + hashed categorical for ALL models
    # --------------------------------------------------------------------------

    hashing_preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat_hash", HashingEncoder(
                n_components=32,
                drop_invariant=True
            ), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    models: Dict[str, object] = {}

    # --------------------------------------------------------------------------
    # 1. LINEAR MODEL (SAFE) → SGDRegressor
    # --------------------------------------------------------------------------
    print("\n⏳ Training Linear Model (SGDRegressor, memory-safe)...")

    linreg = Pipeline([
        ("preprocess", hashing_preprocess),
        ("model", SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="invscaling",
            eta0=0.01,
            max_iter=800,
            tol=1e-4,
            random_state=42,
        )),
    ])

    linreg.fit(train_df, y_train)
    print("✅ Linear Model trained (SGDRegressor).")
    models["linear_regression"] = linreg

    # --------------------------------------------------------------------------
    # 2. RANDOM FOREST
    # --------------------------------------------------------------------------
    print("\n⏳ Training Random Forest...")

    rf = Pipeline([
        ("preprocess", hashing_preprocess),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            n_jobs=-1,
            random_state=42,
        )),
    ])

    rf.fit(train_df, y_train)
    print("✅ Random Forest trained.")
    models["random_forest"] = rf

    # --------------------------------------------------------------------------
    # 3. XGBOOST (optional)
    # --------------------------------------------------------------------------
    if XGBRegressor is not None:
        print("\n⏳ Training XGBoost...")

        xgb = Pipeline([
            ("preprocess", hashing_preprocess),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="rmse",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )),
        ])

        xgb.fit(train_df, y_train)
        print("✅ XGBoost trained.")
        models["xgboost"] = xgb
    else:
        print("⚠️ XGBoost not installed, skipping.")

    # Return all models + metadata
    return models, train_df, val_df, numeric_cols + categorical_cols


from pathlib import Path
from joblib import dump

from config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    MERGED_DATA_PATH,
    FEATURE_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)
from src.data.load_data import load_kaggle_store_sales, load_noaa_normals_daily
from src.data.preprocess import clean_and_merge_data
from src.features.build_features import build_feature_matrix
from src.models.train_models import train_all_models
from src.models.evaluate import evaluate_models

import matplotlib.pyplot as plt


def ensure_directories():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    print("✅ Starting retail sales forecasting pipeline...")

    ensure_directories()

    # 1. Load raw data
    print("📥 Loading Kaggle Store Sales and NOAA weather data...")
    sales_df = load_kaggle_store_sales()
    weather_df = load_noaa_normals_daily()

    # 2. Clean & merge
    print("🧹 Cleaning and merging datasets...")
    merged_df = clean_and_merge_data(sales_df, weather_df)
    merged_df.to_parquet(MERGED_DATA_PATH, index=False)
    print(f"💾 Saved merged data to: {MERGED_DATA_PATH}")

    # 3. Feature engineering
    print("🧩 Building feature matrix...")
    features_df = build_feature_matrix(merged_df)
    features_df.to_parquet(FEATURE_DATA_PATH, index=False)
    print(f"💾 Saved features to: {FEATURE_DATA_PATH}")

    # 4. Train models
    print("🤖 Training supervised models...")
    models, train_df, val_df, feature_cols = train_all_models(features_df)

    # 5. Evaluate
    print("📊 Evaluating models...")
    metrics = evaluate_models(models, train_df, val_df, feature_cols)

    # 6. Persist models
    print("💾 Saving trained models...")
    for name, model in models.items():
        out_path = MODELS_DIR / f"{name}.joblib"
        dump(model, out_path)
        print(f"   • Saved {name} to {out_path}")

    # 7. Simple figure: sales vs temperature for a sample store/family
    try:
        import pandas as pd
        df = merged_df.copy()
        sample = df.dropna(subset=["tavg"]).head(1000)
        if not sample.empty:
            plt.figure()
            plt.scatter(sample["tavg"], sample["sales"], alpha=0.3)
            plt.xlabel("Average Temperature (°C)")
            plt.ylabel("Sales")
            plt.title("Sales vs Temperature (sample)")
            fig_path = FIGURES_DIR / "sales_vs_temperature.png"
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            print(f"📈 Saved figure to {fig_path}")
    except Exception as e:
        print(f"⚠️ Could not generate figure: {e}")

    print("✅ Pipeline finished.")
    print("Metrics:")
    for model_name, metric_dict in metrics.items():
        print(f"  • {model_name}: {metric_dict}")


if __name__ == "__main__":
    run_pipeline()

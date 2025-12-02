
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import METRICS_DIR, TARGET_COLUMN, DATE_COLUMN, STORE_COLUMN, FAMILY_COLUMN


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def evaluate_models(
    models: Dict[str, object],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Dict[str, float]] = {}

    # For pipelines with ColumnTransformer, we can just pass val_df
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLUMN]

    predictions_records = []

    for name, model in models.items():
        y_pred = model.predict(X_val)
        metrics = _regression_metrics(y_val, y_pred)
        all_metrics[name] = metrics

        # Save each model's metrics as a JSON file
        out_path = METRICS_DIR / f"{name}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Validation predictions for dashboard
        tmp = val_df[[DATE_COLUMN, STORE_COLUMN, FAMILY_COLUMN, TARGET_COLUMN]].copy()
        tmp["y_pred"] = y_pred
        tmp["model"] = name
        predictions_records.append(tmp)

    # Save combined metrics
    with open(METRICS_DIR / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save stacked validation predictions
    if predictions_records:
        preds_df = pd.concat(predictions_records, axis=0, ignore_index=True)
        preds_df.to_csv(METRICS_DIR / "val_predictions.csv", index=False)

    return all_metrics

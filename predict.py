"""
Demand forecasting - inference.

Loads the trained model and predicts demand for a CSV that follows the same
schema as the training data. Because the model uses lag/rolling features, the
input must include the historical demand rows that precede the rows you want to
score (the panel history warms up the lags).

Run:
    # Score the shipped dataset (demo) and report accuracy where Demand is known
    .venv/bin/python predict.py

    # Score your own file
    .venv/bin/python predict.py --input path/to/new_data.csv --output results/my_preds.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import build_features, split_X_y

MODEL_PATH = Path("models/demand_model.joblib")
DEFAULT_INPUT = Path("data/demand_forecasting.csv")
DEFAULT_OUTPUT = Path("results/predictions.csv")


def main(input_path, output_path, tail):
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Train it first: "
            f".venv/bin/python train_model.py"
        )

    print(f"Loading model  <- {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Loading input  <- {input_path}")
    raw = pd.read_csv(input_path)

    print("Engineering features ...")
    feat = build_features(raw, dropna=True)
    X, y, _, _ = split_X_y(feat)

    preds = model.predict(X)
    preds = np.clip(preds, 0, None).round().astype(int)  # demand can't be negative

    out = feat[["Date", "Store ID", "Product ID"]].copy()
    out["Predicted_Demand"] = preds
    has_actual = "Demand" in feat.columns
    if has_actual:
        out["Actual_Demand"] = feat["Demand"].values
        out["Error"] = out["Predicted_Demand"] - out["Actual_Demand"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} predictions -> {output_path}\n")

    if has_actual:
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds))
        print("Accuracy on rows with known demand:")
        print(f"  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}\n")

    print(f"Sample (last {tail} rows):")
    print(out.tail(tail).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tail", type=int, default=10, help="rows to preview")
    args = parser.parse_args()
    main(args.input, args.output, args.tail)

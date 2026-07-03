"""
Probabilistic demand forecasting with quantile regression.

Instead of a single number, this trains three XGBoost models with the quantile
(pinball) objective to produce a P10 / P50 / P90 forecast — a prediction
interval. That interval is what inventory planning actually needs: the P90 gives
a demand level you'll exceed only ~10% of the time, so safety stock can be read
straight off the distribution instead of assuming demand is Gaussian.

Outputs:
  * results/quantile_metrics.csv  — coverage (PICP), interval width, pinball loss
  * results/11_quantile_intervals.png — actual vs P50 with the P10–P90 band

Run:
    .venv/bin/python quantile_forecast.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.features import build_features, split_X_y
from train_model import DATA_PATH, RANDOM_STATE, make_preprocessor

RESULTS_DIR = Path("results")
PARAMS_PATH = Path("models/best_params.json")
QUANTILES = [0.1, 0.5, 0.9]


def base_params():
    """Reuse the tuned hyperparameters, minus anything the quantile objective sets."""
    if PARAMS_PATH.exists():
        p = json.loads(PARAMS_PATH.read_text()).get("xgboost_params", {})
    else:
        p = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}
    return p


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Engineering features ...")
    feat = build_features(pd.read_csv(DATA_PATH), dropna=True)
    X, y, categorical, numeric = split_X_y(feat)
    dates = feat["Date"].to_numpy()

    # Time-based split: train on the past, test on the most recent 20% of dates.
    uniq = np.unique(dates)
    cutoff = uniq[int(len(uniq) * 0.8)]
    tr, te = dates < cutoff, dates >= cutoff
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    print(f"  train={tr.sum()}  test={te.sum()}\n")

    params = base_params()
    pre = make_preprocessor(categorical, numeric)

    preds = {}
    rows = []
    for alpha in QUANTILES:
        print(f"Training quantile P{int(alpha*100)} ...")
        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=alpha,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE, **params,
        )
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        preds[alpha] = p
        rows.append({
            "quantile": f"P{int(alpha*100)}",
            "pinball_loss": round(pinball_loss(y_te.to_numpy(), p, alpha), 3),
        })

    # Enforce non-crossing quantiles (P10 <= P50 <= P90) row-wise.
    stacked = np.sort(np.vstack([preds[q] for q in QUANTILES]).T, axis=1)
    p10, p50, p90 = stacked[:, 0], stacked[:, 1], stacked[:, 2]

    # Interval quality on the held-out set.
    inside = ((y_te.to_numpy() >= p10) & (y_te.to_numpy() <= p90)).mean()
    width = float(np.mean(p90 - p10))
    print("\n=== Prediction interval (P10–P90, target coverage 80%) ===")
    print(f"  PICP (coverage): {inside*100:.1f}%   (ideal ~80%)")
    print(f"  Mean interval width: {width:.1f} units")

    report = pd.DataFrame(rows)
    report.to_csv(RESULTS_DIR / "quantile_metrics.csv", index=False)
    print("\nPinball loss per quantile:")
    print(report.to_string(index=False))

    # Plot one series over the test period with its uncertainty band.
    te_df = feat[te].copy()
    te_df["p10"], te_df["p50"], te_df["p90"] = p10, p50, p90
    sample = te_df[(te_df["Store ID"] == "S001") & (te_df["Product ID"] == "P0001")].copy()
    sample["Date"] = pd.to_datetime(sample["Date"])
    sample = sample.sort_values("Date")

    plt.figure(figsize=(11, 5))
    plt.fill_between(sample["Date"], sample["p10"], sample["p90"],
                     color="#2b8cbe", alpha=0.25, label="P10–P90 interval")
    plt.plot(sample["Date"], sample["p50"], color="#2b8cbe", lw=1.8, label="P50 forecast")
    plt.plot(sample["Date"], sample["Demand"], "o", color="#e34a33", ms=3, label="Actual")
    plt.title("Probabilistic forecast — S001/P0001 (test period)")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / "11_quantile_intervals.png"
    plt.savefig(path, dpi=120)
    plt.close()

    print(f"\nSaved metrics -> {RESULTS_DIR / 'quantile_metrics.csv'}")
    print(f"Saved plot    -> {path}")


if __name__ == "__main__":
    main()

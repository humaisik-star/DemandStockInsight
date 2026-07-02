"""
Backtesting with rolling-origin (expanding-window) time-series cross-validation.

A single train/test split can be lucky. Backtesting retrains the model on an
expanding history and evaluates it on the *next* unseen block of dates, repeated
across several folds — the honest way to check a forecaster is stable over time,
not just on one recent period.

    fold 1:  train [====]            test [--]
    fold 2:  train [======]          test [--]
    fold 3:  train [========]        test [--]
    ...

The tuned XGBoost hyperparameters (models/best_params.json, produced by
train_model.py) are reused across folds so we measure temporal stability, not
per-fold tuning noise.

Run:
    .venv/bin/python backtest.py               # 5 folds
    .venv/bin/python backtest.py --folds 8
"""

import argparse
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
from train_model import DATA_PATH, RANDOM_STATE, make_preprocessor, metrics

RESULTS_DIR = Path("results")
PARAMS_PATH = Path("models/best_params.json")


def rolling_origin_folds(dates, n_splits):
    """Yield (train_dates, test_dates) for an expanding-window split by date.

    Dates are divided into n_splits+1 equal blocks; fold i trains on blocks
    0..i and tests on block i+1. Pure function of the date list -> easy to test.
    """
    uniq = np.unique(np.asarray(dates))  # np.unique returns sorted unique values
    block = len(uniq) // (n_splits + 1)
    if block == 0:
        raise ValueError("Not enough distinct dates for the requested folds.")
    folds = []
    for i in range(1, n_splits + 1):
        train_dates = uniq[: block * i]
        test_dates = uniq[block * i : block * (i + 1)]
        if len(test_dates) == 0:
            break
        folds.append((train_dates, test_dates))
    return folds


def load_params():
    if PARAMS_PATH.exists():
        params = json.loads(PARAMS_PATH.read_text()).get("xgboost_params", {})
    else:
        params = {}  # fall back to XGBoost defaults if not tuned yet
    return params


def plot_backtest(report):
    folds = report["fold"]
    plt.figure(figsize=(9, 5))
    plt.plot(folds, report["RMSE"], "o-", color="#2b8cbe", label="RMSE")
    plt.plot(folds, report["MAE"], "s--", color="#74a9cf", label="MAE")
    plt.xlabel("Fold (expanding training window)")
    plt.ylabel("Error (units)")
    plt.title("Backtest: error per time-series fold")
    plt.xticks(folds)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / "10_backtest.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def main(n_folds):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Engineering features ...")
    feat = build_features(pd.read_csv(DATA_PATH), dropna=True)
    X, y, categorical, numeric = split_X_y(feat)
    dates = feat["Date"].to_numpy()

    params = load_params()
    print(f"Using XGBoost params: {params or '(defaults)'}\n")

    folds = rolling_origin_folds(dates, n_folds)
    rows = []
    for i, (train_dates, test_dates) in enumerate(folds, 1):
        tr = np.isin(dates, train_dates)
        te = np.isin(dates, test_dates)
        pipe = Pipeline([
            ("pre", make_preprocessor(categorical, numeric)),
            ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                                   tree_method="hist", **params)),
        ])
        pipe.fit(X[tr], y[tr])
        m = metrics(f"fold{i}", y[te], pipe.predict(X[te]))
        m["fold"] = i
        m["train_days"] = len(train_dates)
        m["test_days"] = len(test_dates)
        rows.append(m)
        print(f"Fold {i}: train_days={len(train_dates):3d}  test_days={len(test_dates):3d}  "
              f"RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  R2={m['R2']:.4f}")

    report = pd.DataFrame(rows)[["fold", "train_days", "test_days", "RMSE", "MAE", "R2"]]
    report.to_csv(RESULTS_DIR / "backtest_metrics.csv", index=False)
    path = plot_backtest(report)

    print("\n=== Backtest summary (across folds) ===")
    print(f"  RMSE: {report['RMSE'].mean():.2f} ± {report['RMSE'].std():.2f}")
    print(f"  MAE : {report['MAE'].mean():.2f} ± {report['MAE'].std():.2f}")
    print(f"  R2  : {report['R2'].mean():.4f} ± {report['R2'].std():.4f}")
    print(f"\nSaved metrics -> {RESULTS_DIR / 'backtest_metrics.csv'}")
    print(f"Saved plot    -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.folds)

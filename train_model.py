"""
Demand forecasting - full training pipeline.

Steps:
  1. Load the daily panel and engineer calendar + lag/rolling features.
  2. Time-based split (train on the past, test on the most recent dates) so
     evaluation reflects real forecasting, not random interpolation.
  3. Tune XGBoost hyperparameters with Optuna against a time-based validation
     fold carved out of the training data.
  4. Train and compare several models on identical features.
  5. Save the best model, a metrics report, and evaluation plots.

Run:
    .venv/bin/python train_model.py            # default 40 Optuna trials
    .venv/bin/python train_model.py --trials 15
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.features import build_features, split_X_y

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = Path("data/demand_forecasting.csv")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame, frac: float):
    """Split a frame by Date: earliest `1-frac` for train, latest `frac` held out."""
    dates = np.sort(df["Date"].unique())
    cutoff = dates[int(len(dates) * (1 - frac))]
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]
    return train, test, cutoff


def make_preprocessor(categorical, numeric):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )


def metrics(name, y_true, preds):
    return {
        "model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, preds))),
        "MAE": float(mean_absolute_error(y_true, preds)),
        "R2": float(r2_score(y_true, preds)),
    }


# --------------------------------------------------------------------------- #
# Optuna tuning for XGBoost
# --------------------------------------------------------------------------- #
def tune_xgboost(pre, X_tr, y_tr, X_val, y_val, n_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", **params
        )
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_feature_importance(pipe, top_n=20):
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(9, 7))
    plt.barh(range(len(order)), importances[order][::-1], color="#2b8cbe")
    plt.yticks(range(len(order)), [names[i] for i in order][::-1], fontsize=8)
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.tight_layout()
    path = RESULTS_DIR / "07_feature_importance.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_predictions(y_true, preds):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, preds, s=6, alpha=0.3, color="#2b8cbe")
    lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Predicted vs Actual (test set)")
    plt.tight_layout()
    path = RESULTS_DIR / "08_pred_vs_actual.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(n_trials):
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    raw = pd.read_csv(DATA_PATH)
    print(f"  raw rows={len(raw)}")

    print("Engineering features (calendar + lag/rolling) ...")
    feat = build_features(raw, dropna=True)
    print(f"  usable rows after lag warm-up={len(feat)}")

    X, y, categorical, numeric = split_X_y(feat)
    print(f"  features={X.shape[1]}  categorical={categorical}")

    # Re-attach Date for the time-based splits, then drop it from X.
    X_dated = X.copy()
    X_dated["Date"] = feat["Date"].values

    train_mask = X_dated["Date"] < np.sort(X_dated["Date"].unique())[
        int(X_dated["Date"].nunique() * 0.8)
    ]
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"  train={len(X_train)}  test={len(X_test)} (last 20% of dates)\n")

    # Validation fold for Optuna: last 20% of the *training* dates.
    tr_dates = X_dated.loc[train_mask, "Date"]
    val_cutoff = np.sort(tr_dates.unique())[int(tr_dates.nunique() * 0.8)]
    val_mask = X_dated.loc[train_mask, "Date"] >= val_cutoff
    X_tr, y_tr = X_train[~val_mask.values], y_train[~val_mask.values]
    X_val, y_val = X_train[val_mask.values], y_train[val_mask.values]

    pre = make_preprocessor(categorical, numeric)

    print(f"Tuning XGBoost with Optuna ({n_trials} trials) ...")
    best_params, best_val = tune_xgboost(pre, X_tr, y_tr, X_val, y_val, n_trials)
    print(f"  best validation RMSE={best_val:.3f}")
    print(f"  best params={json.dumps(best_params, indent=2)}\n")

    candidates = {
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost_tuned": XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", **best_params
        ),
    }

    results, fitted = [], {}
    for name, est in candidates.items():
        print(f"Training {name} ...")
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        m = metrics(name, y_test, pipe.predict(X_test))
        results.append(m)
        fitted[name] = pipe
        print(f"  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  R2={m['R2']:.4f}\n")

    report = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("=== Model comparison (sorted by RMSE) ===")
    print(report.to_string(index=False))

    best_name = report.iloc[0]["model"]
    best_model = fitted[best_name]

    # Save artifacts
    report.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    joblib.dump(best_model, MODELS_DIR / "demand_model.joblib")
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump({"model": best_name, "xgboost_params": best_params}, f, indent=2)

    fi_path = None
    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        fi_path = plot_feature_importance(best_model)
    pred_path = plot_predictions(y_test, best_model.predict(X_test))

    print(f"\nBest model: {best_name}")
    print(f"Saved model   -> {MODELS_DIR / 'demand_model.joblib'}")
    print(f"Saved params  -> {MODELS_DIR / 'best_params.json'}")
    print(f"Saved metrics -> {RESULTS_DIR / 'model_metrics.csv'}")
    if fi_path:
        print(f"Saved plot    -> {fi_path}")
    print(f"Saved plot    -> {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials")
    args = parser.parse_args()
    main(args.trials)

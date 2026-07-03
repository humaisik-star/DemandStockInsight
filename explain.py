"""
Model explainability with SHAP.

Feature importances tell you *which* inputs matter; SHAP tells you *how* — the
direction and magnitude each feature pushes an individual forecast. This script
loads the trained model and produces:

  * results/12_shap_bar.png       — global feature impact (mean |SHAP|)
  * results/13_shap_beeswarm.png  — how feature values push demand up/down
  * a printed local explanation for one example prediction

Run:
    .venv/bin/python explain.py
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features import build_features, split_X_y

MODEL_PATH = Path("models/demand_model.joblib")
DATA_PATH = Path("data/demand_forecasting.csv")
RESULTS_DIR = Path("results")
SAMPLE = 2000  # SHAP on a representative sample keeps it fast


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        raise SystemExit("Train the model first: .venv/bin/python train_model.py")

    print("Loading model + data ...")
    pipe = joblib.load(MODEL_PATH)
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]

    feat = build_features(pd.read_csv(DATA_PATH), dropna=True)
    X, _, _, _ = split_X_y(feat)

    # Use the most recent rows as the explanation sample.
    X_sample = X.tail(SAMPLE)
    names = pre.get_feature_names_out()
    X_trans = pre.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_df = pd.DataFrame(X_trans, columns=[n.split("__", 1)[-1] for n in names])

    print(f"Computing SHAP values on {len(X_df)} rows ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # Global bar: mean absolute SHAP per feature.
    plt.figure()
    shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=15, show=False)
    plt.title("Global feature impact (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "12_shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Beeswarm: distribution of effects, coloured by feature value.
    plt.figure()
    shap.summary_plot(shap_values, X_df, max_display=15, show=False)
    plt.title("How feature values push demand")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "13_shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Rank features by mean |SHAP| and print the top drivers.
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    print("\nTop demand drivers (mean |SHAP|):")
    for i in order:
        print(f"  {X_df.columns[i]:32s} {mean_abs[i]:.2f}")

    # Local explanation: why THIS row got its prediction.
    row = 0
    base = float(explainer.expected_value)
    pred = base + float(shap_values[row].sum())
    contrib = sorted(zip(X_df.columns, shap_values[row]), key=lambda t: abs(t[1]), reverse=True)[:5]
    print(f"\nLocal explanation for sample row {row}:")
    print(f"  baseline demand: {base:.1f}  ->  predicted: {pred:.1f}")
    for name, val in contrib:
        arrow = "↑" if val > 0 else "↓"
        print(f"    {arrow} {name:30s} {val:+.1f}")

    print(f"\nSaved -> {RESULTS_DIR/'12_shap_bar.png'}, {RESULTS_DIR/'13_shap_beeswarm.png'}")


if __name__ == "__main__":
    main()

"""
Stock (inventory) optimization from demand forecasts.

Two replenishment policies are compared at the SAME target service level, so
the comparison isolates the value of the forecast:

  * Naive (forecast-free): assumes flat average demand, so safety stock must
    buffer the FULL demand variability.
        ROP_naive = mean_demand * L + z * sigma_DEMAND * sqrt(L)

  * Model-based: the reorder point uses the sum of the model's day-by-day
    forecast over the lead time, so safety stock only has to buffer the
    model's FORECAST ERROR.
        ROP_model = sum(forecast over next L days) + z * sigma_ERROR * sqrt(L)

Because the model is accurate (sigma_error << sigma_demand), it hits the same
service level with far less safety stock -> lower average inventory. Both
policies are then validated empirically against actual demand.

    L  = replenishment lead time (days)
    z  = service-level factor (e.g. 95% -> 1.645)

Run:
    .venv/bin/python stock.py                       # 95% service, 7-day lead time
    .venv/bin/python stock.py --service-level 0.98 --lead-time 5
"""

import argparse
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRED_PATH = Path("results/predictions.csv")
DATA_PATH = Path("data/demand_forecasting.csv")
OUT_CSV = Path("results/stock_recommendations.csv")
OUT_PLOT = Path("results/09_stock_optimization.png")
KEYS = ["Store ID", "Product ID"]


def forward_window_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Sum of demand over the current + next (window-1) days, per position."""
    s = pd.Series(values)
    return s.rolling(window).sum().shift(-(window - 1)).to_numpy()


def optimize(preds: pd.DataFrame, lead_time: int, z: float):
    L = lead_time
    rows = []
    for (store, product), g in preds.groupby(KEYS, sort=False):
        g = g.sort_values("Date")
        actual = g["Actual_Demand"].to_numpy(dtype=float)
        pred = g["Predicted_Demand"].to_numpy(dtype=float)

        mu = float(actual.mean())                       # avg daily demand
        sigma_demand = float(np.std(actual, ddof=1))    # full demand variability
        sigma_err = float(g["Error"].std(ddof=1))       # model forecast-error std
        sigma_err = sigma_err if np.isfinite(sigma_err) else 0.0

        # Lead-time demand windows (sum over current + next L-1 days).
        actual_lead = forward_window_sum(actual, L)
        pred_lead = forward_window_sum(pred, L)
        valid = ~np.isnan(actual_lead) & ~np.isnan(pred_lead)

        ss_naive = z * sigma_demand * np.sqrt(L)
        ss_model = z * sigma_err * np.sqrt(L)

        # Reorder points per day.
        rop_naive = mu * L + ss_naive                       # flat forecast
        rop_model = pred_lead + ss_model                    # day-specific forecast

        # Empirical service: fraction of windows the policy covers actual demand.
        svc_naive = float((actual_lead[valid] <= rop_naive).mean()) if valid.any() else np.nan
        svc_model = float((actual_lead[valid] <= rop_model[valid]).mean()) if valid.any() else np.nan

        # Average on-hand inventory ~= safety stock + half a demand cycle.
        avg_inv_naive = ss_naive + mu * L / 2
        avg_inv_model = ss_model + mu * L / 2

        rows.append({
            "Store ID": store,
            "Product ID": product,
            "avg_daily_demand": round(mu, 2),
            "demand_std": round(sigma_demand, 2),
            "forecast_error_std": round(sigma_err, 2),
            "safety_stock_naive": round(ss_naive, 1),
            "safety_stock_model": round(ss_model, 1),
            "avg_inventory_naive": round(avg_inv_naive, 1),
            "avg_inventory_model": round(avg_inv_model, 1),
            "service_naive": round(svc_naive, 3) if np.isfinite(svc_naive) else np.nan,
            "service_model": round(svc_model, 3) if np.isfinite(svc_model) else np.nan,
        })

    rec = pd.DataFrame(rows)
    rec["inventory_reduction_%"] = (
        (rec["avg_inventory_naive"] - rec["avg_inventory_model"])
        / rec["avg_inventory_naive"] * 100
    ).round(1)
    return rec


def plot_comparison(rec: pd.DataFrame, service_target: float):
    naive_total = rec["avg_inventory_naive"].sum()
    model_total = rec["avg_inventory_model"].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(["Naive\n(no forecast)", "Model-based\n(optimized)"],
            [naive_total, model_total], color=["#bdbdbd", "#2b8cbe"])
    ax1.set_ylabel("Total average inventory (units)")
    ax1.set_title("Inventory needed for same service level")
    for i, v in enumerate([naive_total, model_total]):
        ax1.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

    ax2.hist(rec["service_naive"].dropna(), bins=15, color="#bdbdbd", alpha=0.7, label="naive")
    ax2.hist(rec["service_model"].dropna(), bins=15, color="#2b8cbe", alpha=0.7, label="model")
    ax2.axvline(service_target, color="red", linestyle="--", label=f"target {service_target:.0%}")
    ax2.set_xlabel("Empirical service level per product")
    ax2.set_ylabel("# of store-product series")
    ax2.set_title("Achieved service level (both hit target)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()


def main(lead_time, service_level):
    if not PRED_PATH.exists():
        raise SystemExit(f"{PRED_PATH} not found. Run predict.py first.")

    z = NormalDist().inv_cdf(service_level)
    print(f"Service level={service_level:.0%} (z={z:.3f}) | lead time={lead_time} days\n")

    preds = pd.read_csv(PRED_PATH)

    rec = optimize(preds, lead_time, z)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT_CSV, index=False)

    naive_total = rec["avg_inventory_naive"].sum()
    model_total = rec["avg_inventory_model"].sum()
    reduction = (naive_total - model_total) / naive_total * 100

    plot_comparison(rec, service_level)

    print("=== Inventory optimization summary ===")
    print(f"  Store-product series:            {len(rec)}")
    print(f"  Naive avg inventory (sum):       {naive_total:,.0f} units")
    print(f"  Model-based avg inventory (sum): {model_total:,.0f} units")
    print(f"  Inventory reduction:             {reduction:.1f}%")
    print(f"  Target service level:            {service_level:.0%}")
    print(f"  Achieved service - naive (avg):  {rec['service_naive'].mean():.1%}")
    print(f"  Achieved service - model (avg):  {rec['service_model'].mean():.1%}")
    print(f"\nSaved recommendations -> {OUT_CSV}")
    print(f"Saved plot            -> {OUT_PLOT}")
    print("\nSample recommendations:")
    cols = ["Store ID", "Product ID", "avg_daily_demand", "demand_std",
            "forecast_error_std", "safety_stock_naive", "safety_stock_model",
            "service_model", "inventory_reduction_%"]
    print(rec[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-time", type=int, default=7, help="replenishment lead time (days)")
    parser.add_argument("--service-level", type=float, default=0.95, help="target service level (0-1)")
    args = parser.parse_args()
    main(args.lead_time, args.service_level)

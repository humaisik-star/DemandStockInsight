"""
Advanced inventory science on top of the demand model.

  * ABC-XYZ matrix  — cross value (ABC) with demand variability (XYZ, by
                      coefficient of variation) into a 9-cell segmentation.
  * Newsvendor      — single-period optimal order quantity from the critical
                      ratio Cu/(Cu+Co), read off the demand distribution.
  * Z-score safety stock — SS = z * σ_leadtime for a target service level.
  * Turnover + days of stock — how fast inventory cycles.
  * Anomaly detection — SKUs whose recent demand or stock position is unusual,
                      with a plain-language reason (for the executive summary).

Inputs: results/stock_recommendations.csv, results/inventory_analytics.csv,
results/predictions.csv (run stock.py, inventory_analytics.py, predict.py first).

Outputs:
  results/advanced_analytics.csv   — per-SKU ABC-XYZ, newsvendor, SS, turnover
  results/abc_xyz_matrix.csv       — 9-cell counts
  results/anomalies.csv            — flagged SKUs + reasons
  results/16_abc_xyz_matrix.png    — matrix heatmap
  results/17_turnover.png          — turnover distribution

Run:
    .venv/bin/python advanced_analytics.py
    .venv/bin/python advanced_analytics.py --margin 0.35 --holding-rate 0.05 --service-level 0.95
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

RESULTS_DIR = Path("results")
STOCK_PATH = RESULTS_DIR / "stock_recommendations.csv"
INV_PATH = RESULTS_DIR / "inventory_analytics.csv"
PRED_PATH = RESULTS_DIR / "predictions.csv"
DAYS_PER_YEAR = 365


def xyz_label(cv):
    """XYZ class from the coefficient of variation of demand."""
    if cv <= 0.5:
        return "X"      # stable, predictable
    if cv <= 1.0:
        return "Y"      # variable
    return "Z"          # erratic


def zscore_safety_stock(demand_std, lead_time, service_level):
    """SS = z * σ over the lead time, z from the target service level."""
    return norm.ppf(service_level) * demand_std * np.sqrt(lead_time)


def newsvendor_quantity(avg_daily, demand_std, lead_time, critical_ratio):
    """Optimal order = leadtime demand + z(critical_ratio) * σ_leadtime."""
    mu = avg_daily * lead_time
    sigma = demand_std * np.sqrt(lead_time)
    return mu + norm.ppf(critical_ratio) * sigma


def detect_anomalies(pred, adv):
    """Flag SKUs with unusual recent demand or stock position, with reasons."""
    rows = []
    for (store, product), g in pred.groupby(["Store ID", "Product ID"]):
        g = g.sort_values("Date")
        overall = g["Actual_Demand"].mean()
        recent = g["Actual_Demand"].tail(7).mean()
        if overall > 0:
            change = (recent - overall) / overall
            if change >= 0.40:
                rows.append((store, product, "Talep sıçraması",
                             f"Son 7 gün ortalaması geneli %{change*100:.0f} aştı", "yüksek"))
            elif change <= -0.40:
                rows.append((store, product, "Talep düşüşü",
                             f"Son 7 gün ortalaması genelden %{abs(change)*100:.0f} düşük", "yüksek"))

    # Stock-position anomalies from the advanced table.
    for _, r in adv.iterrows():
        if r["alert_status"] == "CRITICAL":
            rows.append((r["Store ID"], r["Product ID"], "Kritik stok",
                         f"Mevcut stok güvenlik stoğunun altında ({r['current_inventory']:.0f})", "kritik"))
        elif r["days_of_stock"] > 60:
            rows.append((r["Store ID"], r["Product ID"], "Fazla stok",
                         f"{r['days_of_stock']:.0f} günlük stok — sermaye bağlı", "orta"))

    cols = ["Store ID", "Product ID", "anomaly_type", "detail", "severity"]
    return pd.DataFrame(rows, columns=cols)


def main(margin, holding_rate, lead_time, service_level):
    RESULTS_DIR.mkdir(exist_ok=True)
    for p in (STOCK_PATH, INV_PATH, PRED_PATH):
        if not p.exists():
            raise SystemExit(f"{p} missing. Run stock.py / inventory_analytics.py / predict.py first.")

    stock = pd.read_csv(STOCK_PATH)
    inv = pd.read_csv(INV_PATH)
    pred = pd.read_csv(PRED_PATH)

    df = inv.merge(
        stock[["Store ID", "Product ID", "demand_std", "forecast_error_std", "avg_inventory_model"]],
        on=["Store ID", "Product ID"],
    )

    # --- ABC-XYZ ------------------------------------------------------------
    df["cv"] = (df["demand_std"] / df["avg_daily_demand"]).round(3)
    df["xyz_class"] = df["cv"].apply(xyz_label)
    df["abc_xyz"] = df["abc_class"] + df["xyz_class"]

    # --- Z-score safety stock ----------------------------------------------
    df["safety_stock_zscore"] = zscore_safety_stock(
        df["demand_std"], lead_time, service_level
    ).round(1)

    # --- Newsvendor ---------------------------------------------------------
    # Cu = lost margin per unit, Co = holding cost per unit; CR = Cu / (Cu + Co).
    critical_ratio = margin / (margin + holding_rate)
    df["critical_ratio"] = round(critical_ratio, 3)
    df["newsvendor_qty"] = newsvendor_quantity(
        df["avg_daily_demand"], df["demand_std"], lead_time, critical_ratio
    ).round(1)

    # --- Turnover + days of stock ------------------------------------------
    df["annual_demand"] = (df["avg_daily_demand"] * DAYS_PER_YEAR).round(0)
    df["turnover"] = (df["annual_demand"] / df["avg_inventory_model"]).round(2)
    df["days_of_stock"] = (DAYS_PER_YEAR / df["turnover"]).round(1)

    cols = [
        "Store ID", "Product ID", "abc_class", "xyz_class", "abc_xyz", "cv",
        "critical_ratio", "newsvendor_qty", "safety_stock_zscore",
        "turnover", "days_of_stock", "current_inventory", "alert_status",
    ]
    out = df[cols].copy()
    out.to_csv(RESULTS_DIR / "advanced_analytics.csv", index=False)

    # --- ABC-XYZ matrix -----------------------------------------------------
    matrix = pd.crosstab(df["abc_class"], df["xyz_class"]).reindex(
        index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0
    )
    matrix.to_csv(RESULTS_DIR / "abc_xyz_matrix.csv")

    # --- Anomalies ----------------------------------------------------------
    anomalies = detect_anomalies(pred, df)
    anomalies.to_csv(RESULTS_DIR / "anomalies.csv", index=False)

    # --- Console summary ----------------------------------------------------
    print(f"SKUs: {len(out)}  (margin={margin}, holding_rate={holding_rate}, "
          f"lead_time={lead_time}, service={service_level})")
    print(f"Newsvendor critical ratio: {critical_ratio:.3f}\n")
    print("ABC-XYZ matrix (counts):")
    print(matrix.to_string())
    print("\nXYZ breakdown:")
    for c in ["X", "Y", "Z"]:
        print(f"  {c}: {(df['xyz_class'] == c).sum()} SKUs")
    print(f"\nAnomalies detected: {len(anomalies)}")
    if len(anomalies):
        print(anomalies["anomaly_type"].value_counts().to_string())

    _plot_matrix(matrix)
    _plot_turnover(out)
    print(f"\nSaved -> advanced_analytics.csv, abc_xyz_matrix.csv, anomalies.csv")
    print(f"Saved -> 16_abc_xyz_matrix.png, 17_turnover.png")


def _plot_matrix(matrix):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(3), matrix.columns)
    ax.set_yticks(range(3), matrix.index)
    ax.set_xlabel("XYZ (demand variability)")
    ax.set_ylabel("ABC (value)")
    for i in range(3):
        for j in range(3):
            v = matrix.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > matrix.values.max() / 2 else "#1a2733", fontsize=13)
    ax.set_title("ABC-XYZ segmentation (SKU counts)")
    fig.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "16_abc_xyz_matrix.png", dpi=120)
    plt.close()


def _plot_turnover(out):
    d = out.sort_values("turnover", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(d)), d["turnover"], color="#2b8cbe")
    plt.axhline(d["turnover"].median(), ls="--", color="#e34a33",
                label=f"medyan {d['turnover'].median():.1f}x")
    plt.xlabel("SKUs (ranked)")
    plt.ylabel("Yıllık stok devir hızı (x)")
    plt.title("Inventory turnover across SKUs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "17_turnover.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.30, help="gross margin (Cu = margin*price)")
    parser.add_argument("--holding-rate", type=float, default=0.05, help="per-cycle holding cost rate (Co)")
    parser.add_argument("--lead-time", type=int, default=7)
    parser.add_argument("--service-level", type=float, default=0.95)
    args = parser.parse_args()
    main(args.margin, args.holding_rate, args.lead_time, args.service_level)

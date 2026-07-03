"""
Classic inventory-management analytics on top of the demand model.

Three tools planners actually use day to day:

  * ABC analysis   — Pareto-rank SKUs by annual revenue and label them A/B/C so
                     attention (and safety stock) goes where the money is.
  * EOQ + ROP      — Economic Order Quantity (how much to order) and Reorder
                     Point (when to order), from demand + cost assumptions.
  * Stockout alerts— compare each SKU's current on-hand stock to its reorder
                     point / safety stock and flag CRITICAL / REORDER / OK.

Inputs: the raw panel (current inventory, price) + results/stock_recommendations.csv
(avg daily demand, model safety stock). Run `stock.py` first if that file is missing.

Outputs:
  results/inventory_analytics.csv   — full per-SKU policy table
  results/stockout_alerts.csv       — only SKUs that need action
  results/14_abc_pareto.png         — Pareto chart with A/B/C bands
  results/15_stockout_alerts.png    — alert breakdown / most urgent SKUs

Run:
    .venv/bin/python inventory_analytics.py
    .venv/bin/python inventory_analytics.py --ordering-cost 75 --holding-rate 0.25 --lead-time 7
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path("data/demand_forecasting.csv")
STOCK_PATH = Path("results/stock_recommendations.csv")
RESULTS_DIR = Path("results")
DAYS_PER_YEAR = 365


def abc_label(cum_share):
    """Standard ABC cut-offs on cumulative revenue share."""
    if cum_share <= 0.80:
        return "A"
    if cum_share <= 0.95:
        return "B"
    return "C"


def eoq(annual_demand, ordering_cost, holding_cost):
    """Economic Order Quantity = sqrt(2 * D * S / H)."""
    return np.sqrt(2 * annual_demand * ordering_cost / holding_cost)


def reorder_point(avg_daily_demand, lead_time, safety_stock):
    """Reorder point = expected demand over the lead time + safety stock."""
    return avg_daily_demand * lead_time + safety_stock


def alert_status(current_inventory, safety_stock, rop, days_of_cover=None):
    """CRITICAL below safety stock or under 2 days of cover (imminent stockout),
    REORDER at/below the reorder point, else OK."""
    if current_inventory < safety_stock or (days_of_cover is not None and days_of_cover < 2):
        return "CRITICAL"
    if current_inventory <= rop:
        return "REORDER"
    return "OK"


def main(ordering_cost, holding_rate, lead_time):
    RESULTS_DIR.mkdir(exist_ok=True)
    if not STOCK_PATH.exists():
        raise SystemExit(f"{STOCK_PATH} missing. Run stock.py first.")

    raw = pd.read_csv(DATA_PATH)
    stock = pd.read_csv(STOCK_PATH)

    # Per-SKU price (mean) and current on-hand stock (latest date's inventory).
    raw = raw.sort_values("Date")
    price = raw.groupby(["Store ID", "Product ID"])["Price"].mean().rename("unit_price")
    current_inv = (
        raw.groupby(["Store ID", "Product ID"])["Inventory Level"].last().rename("current_inventory")
    )

    df = stock.merge(price, on=["Store ID", "Product ID"]).merge(
        current_inv, on=["Store ID", "Product ID"]
    )

    # --- ABC analysis: rank by annual revenue -------------------------------
    df["annual_demand"] = df["avg_daily_demand"] * DAYS_PER_YEAR
    df["annual_revenue"] = df["annual_demand"] * df["unit_price"]
    df = df.sort_values("annual_revenue", ascending=False).reset_index(drop=True)
    df["cum_revenue_share"] = df["annual_revenue"].cumsum() / df["annual_revenue"].sum()
    df["abc_class"] = df["cum_revenue_share"].apply(abc_label)

    # --- EOQ + reorder point ------------------------------------------------
    # EOQ = sqrt(2 * D * S / H);  H = holding_rate * unit_price (annual, per unit)
    holding_cost = holding_rate * df["unit_price"]
    df["EOQ"] = eoq(df["annual_demand"], ordering_cost, holding_cost).round(1)
    df["reorder_point"] = reorder_point(
        df["avg_daily_demand"], lead_time, df["safety_stock_model"]
    ).round(1)
    df["days_of_cover"] = (df["current_inventory"] / df["avg_daily_demand"]).round(1)

    # --- Stockout alerts ----------------------------------------------------
    df["alert_status"] = df.apply(
        lambda r: alert_status(r["current_inventory"], r["safety_stock_model"],
                               r["reorder_point"], r["days_of_cover"]),
        axis=1,
    )

    cols = [
        "Store ID", "Product ID", "abc_class", "annual_revenue", "unit_price",
        "avg_daily_demand", "EOQ", "reorder_point", "safety_stock_model",
        "current_inventory", "days_of_cover", "alert_status",
    ]
    out = df[cols].copy()
    out["annual_revenue"] = out["annual_revenue"].round(0)
    out.to_csv(RESULTS_DIR / "inventory_analytics.csv", index=False)

    alerts = out[out["alert_status"] != "OK"].sort_values(
        ["alert_status", "annual_revenue"], ascending=[True, False]
    )
    alerts.to_csv(RESULTS_DIR / "stockout_alerts.csv", index=False)

    # --- Console summary ----------------------------------------------------
    print(f"SKUs analysed: {len(out)}  (lead_time={lead_time}d, "
          f"ordering_cost={ordering_cost}, holding_rate={holding_rate})\n")
    print("ABC breakdown (by count and revenue share):")
    for cls in ["A", "B", "C"]:
        sub = out[out["abc_class"] == cls]
        rev = sub["annual_revenue"].sum() / out["annual_revenue"].sum() * 100
        print(f"  {cls}: {len(sub):3d} SKUs  ->  {rev:4.1f}% of revenue")
    print("\nStockout alerts:")
    for st in ["CRITICAL", "REORDER", "OK"]:
        print(f"  {st:8s}: {(out['alert_status'] == st).sum()} SKUs")

    # --- Charts -------------------------------------------------------------
    _plot_abc(out)
    _plot_alerts(out)
    print(f"\nSaved -> {RESULTS_DIR/'inventory_analytics.csv'}, {RESULTS_DIR/'stockout_alerts.csv'}")
    print(f"Saved -> {RESULTS_DIR/'14_abc_pareto.png'}, {RESULTS_DIR/'15_stockout_alerts.png'}")


def _plot_abc(out):
    d = out.sort_values("annual_revenue", ascending=False).reset_index(drop=True)
    cum = d["annual_revenue"].cumsum() / d["annual_revenue"].sum() * 100
    colors = {"A": "#2b8cbe", "B": "#7bccc4", "C": "#bae4bc"}
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(range(len(d)), d["annual_revenue"], color=[colors[c] for c in d["abc_class"]])
    ax1.set_xlabel("SKUs (ranked by annual revenue)")
    ax1.set_ylabel("Annual revenue")
    ax2 = ax1.twinx()
    ax2.plot(range(len(d)), cum, color="#e34a33", lw=1.8)
    ax2.axhline(80, ls="--", color="#888", lw=0.8)
    ax2.axhline(95, ls="--", color="#888", lw=0.8)
    ax2.set_ylabel("Cumulative revenue %")
    ax2.set_ylim(0, 105)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in ["A", "B", "C"]]
    ax1.legend(handles, ["A (top 80%)", "B (80–95%)", "C (95–100%)"], loc="center right")
    plt.title("ABC analysis — Pareto of SKU revenue")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "14_abc_pareto.png", dpi=120)
    plt.close()


def _plot_alerts(out):
    counts = out["alert_status"].value_counts().reindex(["CRITICAL", "REORDER", "OK"]).fillna(0)
    colors = ["#e34a33", "#fdae61", "#2b8cbe"]
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        plt.text(i, v, int(v), ha="center", va="bottom")
    plt.ylabel("Number of SKUs")
    plt.title("Stock status across SKUs")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "15_stockout_alerts.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordering-cost", type=float, default=50.0, help="cost per order (S)")
    parser.add_argument("--holding-rate", type=float, default=0.20,
                        help="annual holding cost as a fraction of unit price (H = rate*price)")
    parser.add_argument("--lead-time", type=int, default=7, help="supplier lead time in days")
    args = parser.parse_args()
    main(args.ordering_cost, args.holding_rate, args.lead_time)

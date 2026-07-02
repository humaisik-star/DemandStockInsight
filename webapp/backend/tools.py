"""Assistant tools for the web backend.

Same logic as src/assistant_tools.py, but the CSV paths are resolved relative to
this file (so it works both locally and inside the container). Reads the
forecast + stock-recommendation outputs and exposes them as callable tools.
"""

import json
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))
PRED_PATH = DATA_DIR / "predictions.csv"
STOCK_PATH = DATA_DIR / "stock_recommendations.csv"

_pred_cache = None
_stock_cache = None


def _predictions() -> pd.DataFrame:
    global _pred_cache
    if _pred_cache is None:
        _pred_cache = pd.read_csv(PRED_PATH)
    return _pred_cache


def _stock() -> pd.DataFrame:
    global _stock_cache
    if _stock_cache is None:
        _stock_cache = pd.read_csv(STOCK_PATH)
    return _stock_cache


def list_series() -> dict:
    s = _stock()
    return {
        "stores": sorted(s["Store ID"].unique().tolist()),
        "products": sorted(s["Product ID"].unique().tolist()),
    }


def get_demand_forecast(store_id: str, product_id: str, last_n_days: int = 7) -> dict:
    df = _predictions()
    m = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No data for {store_id}/{product_id}."}
    m = m.sort_values("Date").tail(last_n_days)
    return {
        "store_id": store_id,
        "product_id": product_id,
        "rows": m[["Date", "Predicted_Demand", "Actual_Demand"]].to_dict("records"),
        "avg_predicted": round(float(m["Predicted_Demand"].mean()), 1),
    }


def get_stock_recommendation(store_id: str, product_id: str) -> dict:
    s = _stock()
    m = s[(s["Store ID"] == store_id) & (s["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No recommendation for {store_id}/{product_id}."}
    r = m.iloc[0]
    return {
        "store_id": store_id,
        "product_id": product_id,
        "avg_daily_demand": float(r["avg_daily_demand"]),
        "safety_stock_recommended": float(r["safety_stock_model"]),
        "safety_stock_naive": float(r["safety_stock_naive"]),
        "avg_inventory_recommended": float(r["avg_inventory_model"]),
        "service_level_achieved": float(r["service_model"]),
        "inventory_reduction_pct": float(r["inventory_reduction_%"]),
    }


def list_top_stockout_risks(top_n: int = 5) -> dict:
    s = _stock().sort_values("service_model").head(top_n)
    return {
        "top_risks": s[
            ["Store ID", "Product ID", "service_model", "demand_std", "safety_stock_model"]
        ].to_dict("records")
    }


def inventory_summary() -> dict:
    s = _stock()
    naive = float(s["avg_inventory_naive"].sum())
    model = float(s["avg_inventory_model"].sum())
    return {
        "series_count": int(len(s)),
        "total_inventory_naive": round(naive),
        "total_inventory_model": round(model),
        "inventory_reduction_pct": round((naive - model) / naive * 100, 1),
        "avg_service_level_model": round(float(s["service_model"].mean()), 3),
    }


_FUNCS = {
    "list_series": list_series,
    "get_demand_forecast": get_demand_forecast,
    "get_stock_recommendation": get_stock_recommendation,
    "list_top_stockout_risks": list_top_stockout_risks,
    "inventory_summary": inventory_summary,
}

TOOL_SPECS = [
    {"type": "function", "function": {"name": "list_series",
        "description": "List available store IDs and product IDs.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_demand_forecast",
        "description": "Recent forecasted vs actual daily demand for a store-product.",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "string", "description": "e.g. S001"},
            "product_id": {"type": "string", "description": "e.g. P0001"},
            "last_n_days": {"type": "integer", "default": 7}},
            "required": ["store_id", "product_id"]}}},
    {"type": "function", "function": {"name": "get_stock_recommendation",
        "description": "Safety stock, service level, and inventory savings for a store-product.",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "string"}, "product_id": {"type": "string"}},
            "required": ["store_id", "product_id"]}}},
    {"type": "function", "function": {"name": "list_top_stockout_risks",
        "description": "Store-products with the lowest achieved service level (highest risk).",
        "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 5}}}}},
    {"type": "function", "function": {"name": "inventory_summary",
        "description": "Portfolio-wide inventory: naive vs model policy and total savings.",
        "parameters": {"type": "object", "properties": {}}}},
]


def dispatch(name: str, arguments: dict) -> dict:
    if name not in _FUNCS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _FUNCS[name](**(arguments or {}))
    except Exception as e:
        return {"error": str(e)}

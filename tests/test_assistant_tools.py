"""Assistant-tool tests (LLM-agnostic; no API key needed).

Rather than depend on generated CSVs, we inject synthetic data straight into the
module's caches so these tests are fast and hermetic.
"""

import pandas as pd
import pytest

import src.assistant_tools as at


@pytest.fixture(autouse=True)
def inject_caches():
    """Populate the module caches with tiny synthetic frames, then reset."""
    at._pred_cache = pd.DataFrame(
        {
            "Date": ["2024-01-28", "2024-01-29", "2024-01-30"] * 2,
            "Store ID": ["S001"] * 3 + ["S002"] * 3,
            "Product ID": ["P0001"] * 3 + ["P0002"] * 3,
            "Predicted_Demand": [100, 110, 105, 80, 82, 79],
            "Actual_Demand": [102, 108, 106, 78, 85, 80],
            "Error": [-2, 2, -1, 2, -3, -1],
        }
    )
    at._stock_cache = pd.DataFrame(
        {
            "Store ID": ["S001", "S002"],
            "Product ID": ["P0001", "P0002"],
            "avg_daily_demand": [105.0, 80.0],
            "demand_std": [40.0, 45.0],
            "forecast_error_std": [6.0, 7.0],
            "safety_stock_naive": [154.0, 170.0],
            "safety_stock_model": [26.0, 28.0],
            "avg_inventory_naive": [500.0, 450.0],
            "avg_inventory_model": [380.0, 320.0],
            "service_naive": [0.88, 0.86],
            "service_model": [0.95, 0.84],
            "inventory_reduction_%": [24.0, 28.9],
        }
    )
    yield
    at._pred_cache = None
    at._stock_cache = None


def test_list_series():
    out = at.list_series()
    assert out["stores"] == ["S001", "S002"]
    assert "P0001" in out["products"]


def test_get_demand_forecast_returns_rows():
    out = at.get_demand_forecast("S001", "P0001", last_n_days=2)
    assert len(out["rows"]) == 2
    assert out["avg_predicted"] > 0


def test_get_demand_forecast_unknown_series():
    out = at.get_demand_forecast("S999", "P9999")
    assert "error" in out


def test_get_stock_recommendation_keys():
    out = at.get_stock_recommendation("S001", "P0001")
    for k in ["safety_stock_recommended", "service_level_achieved", "inventory_reduction_pct"]:
        assert k in out
    assert out["safety_stock_recommended"] < out["safety_stock_naive"]


def test_top_risks_sorted_ascending_by_service():
    out = at.list_top_stockout_risks(top_n=2)
    services = [r["service_model"] for r in out["top_risks"]]
    assert services == sorted(services)          # riskiest (lowest) first
    assert out["top_risks"][0]["Store ID"] == "S002"


def test_inventory_summary_math():
    out = at.inventory_summary()
    assert out["series_count"] == 2
    assert out["total_inventory_naive"] == 950
    assert out["total_inventory_model"] == 700
    assert out["inventory_reduction_pct"] == pytest.approx(26.3, abs=0.1)


def test_dispatch_routes_and_handles_unknown():
    assert at.dispatch("inventory_summary", {})["series_count"] == 2
    assert "error" in at.dispatch("nonexistent_tool", {})
    # Bad arguments are surfaced as an error, not raised.
    assert "error" in at.dispatch("get_stock_recommendation", {"store_id": "S001"})

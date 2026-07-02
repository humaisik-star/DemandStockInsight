"""Inventory-optimization tests."""

import numpy as np
import pandas as pd
import pytest

from stock import forward_window_sum, optimize


def test_forward_window_sum():
    vals = np.array([1, 2, 3, 4, 5], dtype=float)
    out = forward_window_sum(vals, 2)
    # position i = vals[i] + vals[i+1]; last position is NaN (no full window).
    assert np.allclose(out[:4], [3, 5, 7, 9])
    assert np.isnan(out[4])


@pytest.fixture
def toy_preds():
    """One series where the model is near-perfect but demand is volatile."""
    rng = np.arange(40)
    actual = 100 + 30 * np.sin(rng)          # volatile demand (large std)
    pred = actual + 3 * np.cos(rng)          # accurate but non-constant error
    return pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-01", periods=40).strftime("%Y-%m-%d"),
            "Store ID": "S001",
            "Product ID": "P0001",
            "Predicted_Demand": pred.round(),
            "Actual_Demand": actual.round(),
            "Error": (pred - actual).round(),
        }
    )


def test_model_safety_stock_below_naive(toy_preds):
    """Accurate model buffers only forecast error -> less safety stock than naive."""
    z = 1.645
    rec = optimize(toy_preds, lead_time=7, z=z)
    r = rec.iloc[0]
    assert r["safety_stock_model"] < r["safety_stock_naive"]
    assert r["inventory_reduction_%"] > 0


def test_service_level_is_a_probability(toy_preds):
    rec = optimize(toy_preds, lead_time=7, z=1.645)
    r = rec.iloc[0]
    assert 0.0 <= r["service_model"] <= 1.0
    assert 0.0 <= r["service_naive"] <= 1.0


def test_higher_service_target_raises_safety_stock(toy_preds):
    low = optimize(toy_preds, lead_time=7, z=1.28).iloc[0]   # ~90%
    high = optimize(toy_preds, lead_time=7, z=2.33).iloc[0]  # ~99%
    assert high["safety_stock_model"] > low["safety_stock_model"]

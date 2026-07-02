"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_panel():
    """A small, deterministic 2-store x 2-product daily panel.

    Using a synthetic panel (instead of the 76k-row CSV) keeps feature tests
    fast and lets us assert exact lag values we control.
    """
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    rows = []
    for store in ["S001", "S002"]:
        for product in ["P0001", "P0002"]:
            for i, d in enumerate(dates):
                rows.append(
                    {
                        "Date": d.strftime("%Y-%m-%d"),
                        "Store ID": store,
                        "Product ID": product,
                        "Category": "Electronics",
                        "Region": "North",
                        "Inventory Level": 100 + i,
                        "Units Sold": 50,
                        "Units Ordered": 60,
                        "Price": 20.0,
                        "Discount": 5,
                        "Weather Condition": "Sunny",
                        "Promotion": i % 2,
                        "Competitor Pricing": 21.0,
                        "Seasonality": "Winter",
                        "Epidemic": 0,
                        # Deterministic, series-specific demand so lags are checkable.
                        "Demand": 100 + i,
                    }
                )
    return pd.DataFrame(rows)

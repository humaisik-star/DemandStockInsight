"""Feature-engineering tests — the highest-risk area (data leakage)."""

import numpy as np
import pandas as pd

from src.features import (
    LEAK_COLS,
    TARGET,
    add_calendar_features,
    build_features,
    split_X_y,
)


def test_calendar_features_present(synthetic_panel):
    out = add_calendar_features(synthetic_panel.copy())
    for col in ["year", "month", "dayofweek", "is_weekend", "month_sin", "dow_cos"]:
        assert col in out.columns
    # Cyclical encodings stay within [-1, 1].
    assert out["month_sin"].abs().le(1.0).all()
    assert out["dow_cos"].abs().le(1.0).all()


def test_lag_1_matches_previous_day(synthetic_panel):
    """demand_lag_1 at row t must equal Demand at t-1 within the SAME series."""
    feat = build_features(synthetic_panel, dropna=False)
    for (_, _), g in feat.groupby(["Store ID", "Product ID"]):
        g = g.sort_values("Date")
        expected = g[TARGET].shift(1)
        assert np.allclose(
            g["demand_lag_1"].dropna().values,
            expected.dropna().values,
        )


def test_rolling_excludes_current_day(synthetic_panel):
    """Rolling mean at row t must NOT include Demand[t] (shift(1) applied)."""
    feat = build_features(synthetic_panel, dropna=False)
    g = feat[(feat["Store ID"] == "S001") & (feat["Product ID"] == "P0001")].sort_values("Date")
    # With demand = 100,101,102,... the 7-day rolling mean ending at t-1 for the
    # row at index i (i>=7) is mean(demand[i-7:i]) = 100 + (i-4).
    i = 10
    window_prev = g[TARGET].iloc[i - 7 : i].mean()
    assert np.isclose(g["demand_rollmean_7"].iloc[i], window_prev)
    # It must differ from a window that includes the current day.
    window_incl = g[TARGET].iloc[i - 6 : i + 1].mean()
    assert not np.isclose(g["demand_rollmean_7"].iloc[i], window_incl)


def test_no_leak_columns_in_features(synthetic_panel):
    feat = build_features(synthetic_panel, dropna=True)
    X, y, categorical, numeric = split_X_y(feat)
    for leak in LEAK_COLS + ["Date"]:
        assert leak not in X.columns, f"{leak} leaked into features"
    assert TARGET not in X.columns
    assert y.name == TARGET


def test_dropna_removes_warmup_rows(synthetic_panel):
    feat = build_features(synthetic_panel, dropna=True)
    lag_cols = [c for c in feat.columns if c.startswith("demand_lag_") or c.startswith("demand_roll")]
    assert not feat[lag_cols].isnull().any().any(), "NaNs remain after warm-up drop"
    # Longest window is 30 -> first ~30 rows per series dropped.
    assert len(feat) < len(synthetic_panel)


def test_categorical_numeric_partition(synthetic_panel):
    feat = build_features(synthetic_panel, dropna=True)
    X, _, categorical, numeric = split_X_y(feat)
    # Every feature is classified exactly once.
    assert set(categorical) | set(numeric) == set(X.columns)
    assert set(categorical) & set(numeric) == set()
    assert "Store ID" in categorical and "Product ID" in categorical

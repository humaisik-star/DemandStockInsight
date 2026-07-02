"""
Feature engineering for demand forecasting.

The dataset is a clean daily panel: 5 stores x 20 products x 760 dates.
That structure lets us build proper time-series features (calendar +
lag/rolling statistics of demand per store-product series), which are the
biggest lever on forecast accuracy.

The functions here are shared by training (train_model.py) and inference
(predict.py) so the exact same transformation is applied in both places.
"""

import numpy as np
import pandas as pd

TARGET = "Demand"

# Identifier / outcome columns that must never be used as model inputs.
# Units Sold and Units Ordered are measured alongside demand -> target leakage.
ID_COLS = ["Store ID", "Product ID"]
LEAK_COLS = ["Units Sold", "Units Ordered"]

# Lag / rolling windows (in days) computed per (Store ID, Product ID) series.
LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 30]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the Date column."""
    d = pd.to_datetime(df["Date"])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    # Cyclical encodings so the model sees Dec (12) and Jan (1) as adjacent.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-window demand features per store-product series.

    Rolling stats are shifted by one day so a row never sees its own demand
    (no leakage). Assumes df is sorted by Date within each series.
    """
    grp = df.groupby(ID_COLS, observed=True)[TARGET]

    for lag in LAGS:
        df[f"demand_lag_{lag}"] = grp.shift(lag)

    # transform keeps original-index alignment and rolls strictly within each
    # (store, product) series; the shift(1) prevents a row seeing its own demand.
    for window in ROLL_WINDOWS:
        df[f"demand_rollmean_{window}"] = grp.transform(
            lambda s: s.shift(1).rolling(window).mean()
        )
        df[f"demand_rollstd_{window}"] = grp.transform(
            lambda s: s.shift(1).rolling(window).std()
        )
    return df


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Full feature pipeline: sort, calendar features, lag features.

    Parameters
    ----------
    dropna : if True, drop early rows whose lag/rolling windows are undefined.
             Set False during inference when you have pre-warmed history.
    """
    df = df.copy()
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    if dropna:
        lag_cols = [c for c in df.columns if c.startswith("demand_lag_") or c.startswith("demand_roll")]
        df = df.dropna(subset=lag_cols).reset_index(drop=True)
    return df


def split_X_y(df: pd.DataFrame):
    """Split engineered frame into X, y and report categorical/numeric columns.

    Store ID / Product ID are kept as categorical inputs (low cardinality:
    5 and 20) since they identify each demand series. Date and leak columns
    are dropped.
    """
    drop = ["Date"] + LEAK_COLS
    frame = df.drop(columns=[c for c in drop if c in df.columns])

    y = frame[TARGET]
    X = frame.drop(columns=[TARGET])

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return X, y, categorical, numeric

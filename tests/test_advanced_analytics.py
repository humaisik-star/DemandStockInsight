"""Tests for ABC-XYZ / newsvendor / z-score safety stock / turnover logic."""

import numpy as np
from scipy.stats import norm

from advanced_analytics import newsvendor_quantity, xyz_label, zscore_safety_stock


def test_xyz_label_thresholds():
    assert xyz_label(0.3) == "X"     # stable
    assert xyz_label(0.5) == "X"     # boundary
    assert xyz_label(0.8) == "Y"     # variable
    assert xyz_label(1.0) == "Y"     # boundary
    assert xyz_label(1.5) == "Z"     # erratic


def test_zscore_safety_stock():
    # z(0.95) ≈ 1.645; SS = z * σ * sqrt(L)
    ss = zscore_safety_stock(demand_std=10, lead_time=4, service_level=0.95)
    assert np.isclose(ss, norm.ppf(0.95) * 10 * 2)   # sqrt(4)=2


def test_higher_service_raises_safety_stock():
    low = zscore_safety_stock(10, 7, 0.90)
    high = zscore_safety_stock(10, 7, 0.99)
    assert high > low


def test_newsvendor_at_median_equals_mean_demand():
    # critical_ratio 0.5 -> z=0 -> order = lead-time mean demand
    q = newsvendor_quantity(avg_daily=20, demand_std=5, lead_time=3, critical_ratio=0.5)
    assert np.isclose(q, 20 * 3)


def test_newsvendor_high_ratio_orders_more_than_mean():
    mean = 20 * 3
    q = newsvendor_quantity(20, 5, 3, critical_ratio=0.9)   # CR>0.5 -> buffer up
    assert q > mean

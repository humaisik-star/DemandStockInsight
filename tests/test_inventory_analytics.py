"""Tests for ABC / EOQ / reorder-point / stockout-alert logic."""

import numpy as np

from inventory_analytics import abc_label, alert_status, eoq, reorder_point


def test_abc_label_thresholds():
    assert abc_label(0.5) == "A"      # within top 80%
    assert abc_label(0.80) == "A"     # boundary counts as A
    assert abc_label(0.90) == "B"     # 80–95%
    assert abc_label(0.95) == "B"     # boundary counts as B
    assert abc_label(0.99) == "C"     # tail


def test_eoq_formula():
    # D=1000, S=50, H=2 -> EOQ = sqrt(2*1000*50/2) = sqrt(50000) ≈ 223.6
    assert np.isclose(eoq(1000, 50, 2), np.sqrt(50000))


def test_eoq_grows_with_demand_and_order_cost():
    base = eoq(1000, 50, 2)
    assert eoq(4000, 50, 2) > base          # more demand -> bigger orders
    assert eoq(1000, 200, 2) > base         # costlier ordering -> bigger orders
    assert eoq(1000, 50, 8) < base          # costlier holding -> smaller orders


def test_reorder_point():
    # 10/day over a 7-day lead time + 15 safety = 85
    assert reorder_point(10, 7, 15) == 85


def test_alert_status_transitions():
    # safety=20, reorder point=85
    assert alert_status(10, 20, 85) == "CRITICAL"   # below safety stock
    assert alert_status(50, 20, 85) == "REORDER"    # between safety and ROP
    assert alert_status(85, 20, 85) == "REORDER"    # exactly at ROP -> reorder
    assert alert_status(200, 20, 85) == "OK"        # well stocked

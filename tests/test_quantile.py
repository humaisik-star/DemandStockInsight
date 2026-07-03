"""Tests for probabilistic forecasting helpers."""

import numpy as np

from quantile_forecast import pinball_loss


def test_pinball_loss_zero_when_perfect():
    y = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y, y, 0.5) == 0.0


def test_pinball_loss_asymmetry():
    # For the P90 quantile, under-prediction is penalised ~9x more than over-prediction.
    y = np.array([100.0])
    under = pinball_loss(y, np.array([90.0]), 0.9)   # predicted below actual
    over = pinball_loss(y, np.array([110.0]), 0.9)   # predicted above actual
    assert np.isclose(under, 0.9 * 10)
    assert np.isclose(over, 0.1 * 10)
    assert under > over


def test_quantile_sorting_enforces_monotonicity():
    # Raw quantile models can cross; sorting each row fixes P10<=P50<=P90.
    raw = np.array([[12.0, 8.0, 10.0], [5.0, 30.0, 20.0]])  # (rows, quantiles) unsorted
    ordered = np.sort(raw, axis=1)
    assert np.all(ordered[:, 0] <= ordered[:, 1])
    assert np.all(ordered[:, 1] <= ordered[:, 2])

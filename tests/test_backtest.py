"""Tests for the rolling-origin fold splitter (fast, no model fitting)."""

import numpy as np
import pytest

from backtest import rolling_origin_folds


def test_folds_are_time_ordered_and_expanding():
    dates = [f"2022-01-{d:02d}" for d in range(1, 25)]  # 24 distinct dates
    folds = rolling_origin_folds(dates, n_splits=3)
    assert len(folds) == 3
    prev_train_len = 0
    for train_dates, test_dates in folds:
        # Training window expands each fold.
        assert len(train_dates) > prev_train_len
        prev_train_len = len(train_dates)
        # Test dates come strictly AFTER every training date (no leakage).
        assert max(train_dates) < min(test_dates)
        assert len(test_dates) > 0


def test_no_overlap_between_train_and_test():
    dates = [f"2022-02-{d:02d}" for d in range(1, 21)]
    for train_dates, test_dates in rolling_origin_folds(dates, n_splits=4):
        assert set(train_dates).isdisjoint(set(test_dates))


def test_raises_when_too_few_dates():
    with pytest.raises(ValueError):
        rolling_origin_folds(["2022-01-01", "2022-01-02"], n_splits=5)

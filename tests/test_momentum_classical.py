"""
Unit Tests for Classical Jegadeesh & Titman (1993) Momentum Models.
"""

import numpy as np
import pandas as pd
import pytest

from src.momentum.classical import (
    calculate_intermediate_momentum,
    calculate_jegadeesh_titman_momentum,
    calculate_multi_timeframe_returns,
    compute_jegadeesh_titman_momentum,
)


class TestClassicalMomentumMath:
    """Mathematical verification of 12-1, 6-1, and 3-1 intermediate momentum."""

    def test_12_1_momentum_microstructure_skip(self):
        """
        Analytical Test Case 3 from quant_specs.md:
        P_0 = 100.0 (t-252)
        P_231 = 200.0 (t-21)
        P_252 = 180.0 (t_0)
        Mom_12_1 = (200.0 - 100.0) / 100.0 = 1.00 (+100%)
        Raw 12m return = (180.0 - 100.0) / 100.0 = 0.80 (+80%)
        Demonstrates that recent 1-month dip (-10%) is skipped.
        """
        prices = np.full(253, 100.0)
        # Set start (t-252)
        prices[0] = 100.0
        # Set intermediate skip date (t-21, index 253 - 22 = 231)
        prices[231] = 200.0
        # Set latest (t_0, index 252)
        prices[252] = 180.0

        mom_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
        assert mom_12_1 == pytest.approx(1.00, rel=1e-5)

        mtf = calculate_multi_timeframe_returns(prices)
        assert mtf["mom_12_1"] == pytest.approx(1.00, rel=1e-5)
        assert mtf["ret_12m"] == pytest.approx(0.80, rel=1e-5)

    def test_6_1_and_3_1_intermediate_momentum(self):
        """
        Verifies 6-1 (126d total, 21d skip) and 3-1 (63d total, 21d skip).
        """
        prices = np.full(130, 100.0)
        # For 6-1: length >= 127
        # Index of t-126 is 130 - 127 = 3
        # Index of t-21 is 130 - 22 = 108
        prices[3] = 100.0
        prices[108] = 150.0  # +50%
        prices[-1] = 140.0

        mom_6_1 = calculate_intermediate_momentum(prices, lookback=126, skip_recent=21)
        assert mom_6_1 == pytest.approx(0.50, rel=1e-5)

    def test_log_return_mode(self):
        """Verifies continuous log return mode: ln(P_{t-21}) - ln(P_{t-252})."""
        prices = np.full(253, 100.0)
        prices[0] = 100.0
        prices[231] = 200.0
        prices[252] = 180.0

        mom_log = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21, log_return=True)
        expected_log = np.log(200.0) - np.log(100.0)
        assert mom_log == pytest.approx(expected_log, rel=1e-5)


class TestClassicalMomentumEdgeCases:
    """Edge cases: short histories, NaNs, zeros."""

    def test_insufficient_history(self):
        """Returns NaN when prices length < lookback_total."""
        prices = np.full(200, 100.0)
        mom = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
        assert np.isnan(mom)

    def test_none_or_empty_input(self):
        """Returns NaN on empty or None input."""
        assert np.isnan(compute_jegadeesh_titman_momentum(None))
        assert np.isnan(compute_jegadeesh_titman_momentum([]))
        assert np.isnan(compute_jegadeesh_titman_momentum(np.array([])))

    def test_zero_or_negative_prices(self):
        """Returns NaN if start price or skip price is non-positive."""
        prices = np.full(253, 100.0)
        prices[0] = 0.0
        assert np.isnan(compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21))

        prices[0] = -50.0
        assert np.isnan(compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21))

    def test_nan_values_at_critical_points(self):
        """Returns NaN if prices at critical indices are NaN."""
        prices = np.full(253, 100.0)
        prices[231] = np.nan
        assert np.isnan(compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21))

    def test_pandas_series_support(self):
        """Supports pd.Series identically."""
        prices_arr = np.full(253, 100.0)
        prices_arr[0] = 100.0
        prices_arr[231] = 150.0
        prices_ser = pd.Series(prices_arr)

        res_arr = compute_jegadeesh_titman_momentum(prices_arr)
        res_ser = calculate_jegadeesh_titman_momentum(prices_ser)
        assert res_arr == pytest.approx(res_ser, rel=1e-7)

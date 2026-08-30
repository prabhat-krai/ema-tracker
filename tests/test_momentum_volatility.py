"""
Unit Tests for Volatility Estimation, Downside Risk & Smoothness Penalty Models.
"""

import numpy as np
import pandas as pd
import pytest

from src.momentum.volatility import (
    calculate_atr,
    calculate_downside_deviation,
    calculate_jump_ratio_penalty,
    calculate_realized_volatility,
    calculate_volatility_adjusted_momentum,
    compute_volatility_metrics,
)


class TestVolatilityMetricsMath:
    """Mathematical verification of realized volatility, downside risk, and ATR."""

    def test_realized_volatility_flat_series(self):
        """Constant prices have zero daily returns and zero volatility."""
        prices = np.full(90, 100.0)
        vol = calculate_realized_volatility(prices, lookback=90, trading_days=252)
        assert vol == 0.0

    def test_realized_volatility_constant_compounding(self):
        """
        Constant compounding P_t = 100 * (1.01)^t has constant daily log returns ln(1.01),
        so sample variance of daily log returns is zero -> volatility is 0.0.
        """
        t = np.arange(90, dtype=np.float64)
        prices = 100.0 * (1.01 ** t)
        vol = calculate_realized_volatility(prices, lookback=90, trading_days=252)
        assert vol == pytest.approx(0.0, abs=1e-8)

    def test_realized_volatility_known_series(self):
        """
        Construct a series with alternating daily returns +2% and -2%.
        Log returns are +ln(1.02) and -ln(1.02).
        Daily standard deviation can be analytically computed.
        """
        n = 100
        log_ret = np.array([np.log(1.02), -np.log(1.02)] * (n // 2))
        prices = 100.0 * np.exp(np.cumsum(np.insert(log_ret, 0, 0.0)))

        vol = calculate_realized_volatility(prices, lookback=90, trading_days=252)
        expected_daily_std = np.std(log_ret[-90:], ddof=1)
        expected_vol = expected_daily_std * np.sqrt(252.0)

        assert vol == pytest.approx(expected_vol, rel=1e-5)
        assert vol > 0.10  # Meaningful positive volatility

    def test_downside_deviation_pure_uptrend(self):
        """In a pure uptrend with no negative returns, downside deviation is exactly 0.0."""
        t = np.arange(90, dtype=np.float64)
        prices = 100.0 * (1.005 ** t)
        down_dev = calculate_downside_deviation(prices, lookback=90, trading_days=252)
        assert down_dev == 0.0

    def test_downside_deviation_with_drawdowns(self):
        """Series with drawdowns produces positive downside deviation."""
        prices = np.array([100.0, 95.0, 90.0, 92.0, 88.0, 95.0] * 15, dtype=np.float64)
        down_dev = calculate_downside_deviation(prices, lookback=90, trading_days=252)
        assert down_dev > 0.0

    def test_jump_ratio_smooth_vs_spiky(self):
        """
        Smooth steady series has jump ratio <= 3.0 -> penalty = 1.0 (no penalty).
        Spiky series with 40% single-day jump has high jump ratio and penalty < 1.0.
        """
        # Smooth series
        t = np.arange(90, dtype=np.float64)
        smooth_prices = 100.0 * (1.002 ** t)
        ratio_smooth, penalty_smooth = calculate_jump_ratio_penalty(smooth_prices, lookback=90)
        assert penalty_smooth == 1.0

        # Spiky series with a massive jump
        np.random.seed(42)
        normal_daily = np.random.normal(0.0005, 0.005, size=90)
        normal_daily[50] = 0.15  # +15% single-day spike (30x std)
        spiky_prices = 100.0 * np.exp(np.cumsum(np.insert(normal_daily, 0, 0.0)))

        ratio_spiky, penalty_spiky = calculate_jump_ratio_penalty(spiky_prices, lookback=90)
        assert ratio_spiky > 5.0
        assert penalty_spiky < 0.50  # Significantly penalized

    def test_atr_and_natr(self):
        """
        Test 14-day ATR with constant bar ranges:
        High = 105, Low = 95, Close = 100 -> TR = 10 on all days.
        ATR should converge to 10.0, NATR should be (10 / 100) * 100 = 10.0%.
        """
        n = 30
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)

        atr, natr = calculate_atr(high, low, close, period=14)
        assert atr == pytest.approx(10.0, rel=1e-3)
        assert natr == pytest.approx(10.0, rel=1e-3)

    def test_volatility_adjusted_momentum(self):
        """Tests momentum / vol normalization ratio."""
        mom = 0.30  # +30%
        vol = 0.15  # 15% annualized vol
        ratio = calculate_volatility_adjusted_momentum(mom, vol)
        assert ratio == pytest.approx(2.0, rel=1e-5)

        # Zero volatility is protected by epsilon
        ratio_zero_vol = calculate_volatility_adjusted_momentum(mom, 0.0, epsilon=1e-6)
        assert ratio_zero_vol == pytest.approx(0.30 / 1e-6, rel=1e-5)


class TestVolatilityEdgeCases:
    """Edge-case tests for volatility and risk metrics."""

    def test_insufficient_length(self):
        """Returns NaN when prices length < 2."""
        assert np.isnan(calculate_realized_volatility([100.0]))
        assert np.isnan(calculate_realized_volatility([]))
        assert np.isnan(calculate_downside_deviation([]))

    def test_invalid_prices(self):
        """Non-positive or NaN prices return NaN."""
        prices = np.full(90, 100.0)
        prices[10] = -5.0
        assert np.isnan(calculate_realized_volatility(prices))

        prices_nan = np.full(90, 100.0)
        prices_nan[10] = np.nan
        assert np.isnan(calculate_realized_volatility(prices_nan))

    def test_compute_volatility_metrics_helper(self):
        """compute_volatility_metrics returns all required dictionary keys."""
        prices = np.full(100, 100.0)
        res = compute_volatility_metrics(prices, lookback=90)
        assert "vol_ann" in res
        assert "vol_down" in res
        assert "max_jump_ratio" in res
        assert "jump_penalty" in res

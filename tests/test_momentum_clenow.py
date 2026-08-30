"""
Unit Tests for Andreas Clenow Exponential Regression Momentum Model.
"""

import numpy as np
import pandas as pd
import pytest

from src.momentum.clenow import (
    calculate_clenow_momentum,
    calculate_multi_timeframe_clenow,
    compute_clenow_momentum,
)


class TestClenowMomentumMath:
    """Mathematical verification tests against analytical ground truth."""

    def test_pure_exponential_compounding_uptrend(self):
        """
        Analytical Test Case 1: P_t = 100 * (1.002)^t for t = 0..89 (N=90)
        Exact beta = ln(1.002)
        Exact annualized slope = exp(250 * ln(1.002)) - 1 = (1.002)^250 - 1 ≈ 0.647898 (+64.79%)
        Exact R^2 = 1.0 (perfect fit in log space)
        Exact Clenow Score = Slope_ann * R^2 ≈ 0.647898
        """
        n = 90
        t = np.arange(n, dtype=np.float64)
        growth_rate = 1.002
        prices = 100.0 * (growth_rate ** t)

        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90, trading_days=250)

        expected_slope = (growth_rate ** 250) - 1.0
        expected_r2 = 1.0
        expected_score = expected_slope * expected_r2

        assert slope_ann == pytest.approx(expected_slope, rel=1e-5)
        assert r2 == pytest.approx(expected_r2, rel=1e-5)
        assert score == pytest.approx(expected_score, rel=1e-5)
        assert 0.647 < score < 0.649

    def test_pure_exponential_compounding_downtrend(self):
        """
        Analytical Test Case 2: P_t = 100 * (0.998)^t for t = 0..89 (N=90)
        Exact annualized slope = (0.998)^250 - 1 ≈ -0.3938
        Exact R^2 = 1.0
        Exact Score ≈ -0.3938
        """
        n = 90
        t = np.arange(n, dtype=np.float64)
        growth_rate = 0.998
        prices = 100.0 * (growth_rate ** t)

        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90, trading_days=250)

        expected_slope = (growth_rate ** 250) - 1.0
        assert slope_ann == pytest.approx(expected_slope, rel=1e-5)
        assert r2 == pytest.approx(1.0, rel=1e-5)
        assert score == pytest.approx(expected_slope, rel=1e-5)
        assert -0.395 < score < -0.392

    def test_pure_flat_series(self):
        """
        Analytical Test Case 3: Constant price P_t = 100.0 for all t
        Zero variance in log prices -> beta = 0, Slope = 0, R^2 = 0, Score = 0
        """
        prices = np.full(90, 100.0)
        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90)

        assert slope_ann == 0.0
        assert r2 == 0.0
        assert score == 0.0

    def test_noisy_exponential_series(self):
        """
        Exponential trend with added noise has lower R^2 (< 1.0) and penalizes the score.
        """
        np.random.seed(42)
        n = 90
        t = np.arange(n, dtype=np.float64)
        clean_prices = 100.0 * (1.002 ** t)
        noise = np.random.normal(0.0, 0.02, size=n)
        noisy_prices = clean_prices * np.exp(noise)

        slope_clean, r2_clean, score_clean = compute_clenow_momentum(clean_prices, lookback=90)
        slope_noisy, r2_noisy, score_noisy = compute_clenow_momentum(noisy_prices, lookback=90)

        assert r2_clean == pytest.approx(1.0, rel=1e-5)
        assert 0.70 < r2_noisy < 0.99
        assert score_noisy < score_clean  # Noise penalty reduces score


class TestClenowEdgeCases:
    """Robustness and edge-case handling tests."""

    def test_insufficient_history(self):
        """If prices length < lookback, returns NaNs."""
        prices = np.array([100.0, 101.0, 102.0])
        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90)

        assert np.isnan(slope_ann)
        assert np.isnan(r2)
        assert np.isnan(score)

    def test_empty_or_none_input(self):
        """Empty or None inputs return NaNs."""
        assert np.isnan(compute_clenow_momentum(None)[0])
        assert np.isnan(compute_clenow_momentum([])[0])
        assert np.isnan(compute_clenow_momentum(np.array([]))[0])

    def test_zero_and_negative_prices(self):
        """Non-positive prices cannot be log-transformed -> return NaNs."""
        prices_with_zero = np.full(90, 100.0)
        prices_with_zero[45] = 0.0
        assert np.isnan(compute_clenow_momentum(prices_with_zero, lookback=90)[0])

        prices_with_neg = np.full(90, 100.0)
        prices_with_neg[10] = -50.0
        assert np.isnan(compute_clenow_momentum(prices_with_neg, lookback=90)[0])

    def test_nan_and_inf_prices(self):
        """NaN or Inf prices return NaNs."""
        prices_nan = np.full(90, 100.0)
        prices_nan[20] = np.nan
        assert np.isnan(compute_clenow_momentum(prices_nan, lookback=90)[0])

        prices_inf = np.full(90, 100.0)
        prices_inf[20] = np.inf
        assert np.isnan(compute_clenow_momentum(prices_inf, lookback=90)[0])

    def test_pandas_series_input(self):
        """pd.Series input produces identical results as NumPy array."""
        t = np.arange(90, dtype=np.float64)
        prices = 50.0 * (1.0015 ** t)
        series = pd.Series(prices)

        res_arr = compute_clenow_momentum(prices, lookback=90)
        res_ser = calculate_clenow_momentum(series, lookback=90)

        assert res_arr[0] == pytest.approx(res_ser[0], rel=1e-7)
        assert res_arr[1] == pytest.approx(res_ser[1], rel=1e-7)
        assert res_arr[2] == pytest.approx(res_ser[2], rel=1e-7)


class TestMultiTimeframeClenow:
    """Multi-timeframe Clenow composite calculation tests."""

    def test_full_history_composite(self):
        """With 300 bars of pure exponential growth, computes 90d, 126d, 252d and composite."""
        n = 300
        t = np.arange(n, dtype=np.float64)
        prices = 100.0 * (1.002 ** t)

        mtf = calculate_multi_timeframe_clenow(prices)
        assert "score_90" in mtf
        assert "score_126" in mtf
        assert "score_252" in mtf
        assert "composite_score" in mtf

        expected = (1.002 ** 250) - 1.0
        assert mtf["score_90"] == pytest.approx(expected, rel=1e-4)
        assert mtf["score_126"] == pytest.approx(expected, rel=1e-4)
        assert mtf["score_252"] == pytest.approx(expected, rel=1e-4)
        assert mtf["composite_score"] == pytest.approx(expected, rel=1e-4)

    def test_partial_history_composite(self):
        """With 100 bars, 90d is computed while 126d and 252d are NaN; composite reweights to 90d."""
        n = 100
        t = np.arange(n, dtype=np.float64)
        prices = 100.0 * (1.002 ** t)

        mtf = calculate_multi_timeframe_clenow(prices)
        assert not np.isnan(mtf["score_90"])
        assert np.isnan(mtf["score_126"])
        assert np.isnan(mtf["score_252"])
        assert mtf["composite_score"] == pytest.approx(mtf["score_90"], rel=1e-4)

    def test_too_short_history_composite(self):
        """With 50 bars (< 90), composite is NaN."""
        prices = np.full(50, 100.0)
        mtf = calculate_multi_timeframe_clenow(prices)
        assert np.isnan(mtf["composite_score"])

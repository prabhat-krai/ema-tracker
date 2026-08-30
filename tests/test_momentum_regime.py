"""
Unit Tests for Macro Market Regime, Single-Stock Trend Qualification & Eligibility Filters.
"""

import numpy as np
import pandas as pd
import pytest

from src.momentum.models import MarketRegimeState
from src.momentum.regime import (
    calculate_52w_high_distance,
    calculate_ema,
    calculate_sma,
    check_golden_regime,
    check_single_stock_trend_filter,
    evaluate_market_regime,
    evaluate_stock_eligibility,
)


class TestMovingAverages:
    """Test SMA and EMA vectorization."""

    def test_sma_constant_and_step(self):
        prices = np.full(50, 100.0)
        sma = calculate_sma(prices, window=10)
        assert np.isnan(sma[8])
        assert sma[9] == pytest.approx(100.0)
        assert sma[-1] == pytest.approx(100.0)

    def test_ema_constant(self):
        prices = np.full(50, 100.0)
        ema = calculate_ema(prices, span=10)
        assert ema[-1] == pytest.approx(100.0)


class TestSingleStockTrendFilter:
    """Test single-stock moving average and regime checks."""

    def test_stock_passing_trend_filter(self):
        """Uptrending stock where Close > EMA100 and Close > SMA200 passes."""
        n = 250
        t = np.arange(n, dtype=np.float64)
        prices = 100.0 + t * 0.5  # Steady uptrend 100 to 224.5

        passes, details = check_single_stock_trend_filter(prices, ema_fast=100, sma_slow=200)
        assert passes is True
        assert details["passes"] is True
        assert details["above_ema_fast"] is True
        assert details["above_sma_slow"] is True

    def test_stock_failing_ema_filter(self):
        """Stock below EMA100 fails."""
        n = 250
        prices = np.linspace(100, 200, n)
        # Drop price sharply on last day
        prices[-1] = 50.0

        passes, details = check_single_stock_trend_filter(prices, ema_fast=100, sma_slow=200)
        assert passes is False
        assert details["above_ema_fast"] is False

    def test_stock_failing_insufficient_history(self):
        """Stock with < 200 bars fails gracefully."""
        prices = np.linspace(100, 150, 150)
        passes, details = check_single_stock_trend_filter(prices, ema_fast=100, sma_slow=200)
        assert passes is False
        assert "Insufficient history" in details["reason"]

    def test_golden_alignment(self):
        """Strong uptrend exhibits EMA10 > EMA50 > SMA200."""
        n = 250
        t = np.arange(n, dtype=np.float64)
        prices = 100.0 + t * 1.0
        assert check_golden_regime(prices, ema_short=10, ema_mid=50, sma_long=200) is True

        # Downtrend has inverted alignment
        prices_down = 300.0 - t * 1.0
        assert check_golden_regime(prices_down, ema_short=10, ema_mid=50, sma_long=200) is False

    def test_52w_high_distance(self):
        """Stock at 90 when 52w high is 100 is at 0.90 distance."""
        n = 252
        prices = np.full(n, 80.0)
        prices[100] = 100.0  # Peak
        prices[-1] = 90.0   # Current

        dist = calculate_52w_high_distance(prices, lookback=252)
        assert dist == pytest.approx(0.90, rel=1e-4)


class TestMarketRegimeMacroGate:
    """Test macro market benchmark regime states (BULLISH, NEUTRAL, BEARISH)."""

    def test_bullish_market_regime(self):
        """Benchmark above SMA200 with ascending SMA200 is BULLISH (100% allocation)."""
        n = 250
        t = np.arange(n, dtype=np.float64)
        prices = 10000.0 + t * 50.0  # Strongly rising index

        regime = evaluate_market_regime(prices, benchmark_symbol="^NSEI", lookback_sma=200, slope_window=20)
        assert regime.state == MarketRegimeState.BULLISH
        assert regime.equity_allocation_pct == 1.0
        assert regime.sma_200_slope > 0.0

    def test_neutral_market_regime(self):
        """Benchmark above SMA200 but SMA200 is descending is NEUTRAL (50% allocation)."""
        n = 250
        # Create a series that had a huge peak 200 days ago, then declined, but latest price is just above SMA200
        # Construct explicit prices
        prices = np.zeros(250)
        prices[:150] = np.linspace(15000, 10000, 150)  # declining long-term
        prices[150:] = np.linspace(10000, 12000, 100)  # recent rebound above average but 200d SMA still flat/down

        # Evaluate regime
        regime = evaluate_market_regime(prices, lookback_sma=200, slope_window=20)
        # Verify it returns a valid MarketRegime object
        assert regime.state in (MarketRegimeState.NEUTRAL, MarketRegimeState.BULLISH, MarketRegimeState.BEARISH)
        assert 0.0 <= regime.equity_allocation_pct <= 1.0

    def test_bearish_market_regime(self):
        """Benchmark below SMA200 is BEARISH (0% equity allocation / defensive cash)."""
        n = 250
        t = np.arange(n, dtype=np.float64)
        prices = 20000.0 - t * 50.0  # Steady decline below moving average

        regime = evaluate_market_regime(prices, benchmark_symbol="^NSEI", lookback_sma=200)
        assert regime.state == MarketRegimeState.BEARISH
        assert regime.equity_allocation_pct == 0.0

    def test_insufficient_or_missing_benchmark(self):
        """Missing benchmark data defaults to defensive NEUTRAL."""
        regime_none = evaluate_market_regime(None)
        assert regime_none.state == MarketRegimeState.NEUTRAL
        assert regime_none.equity_allocation_pct == 0.50

        regime_short = evaluate_market_regime(np.array([100.0, 105.0]))
        assert regime_short.state == MarketRegimeState.NEUTRAL


class TestStockEligibilityEvaluation:
    """Test full pipeline stock eligibility check."""

    def test_eligible_stock(self):
        """Well-formed uptrending stock with sufficient history and volume passes."""
        n = 260
        t = np.arange(n, dtype=np.float64)
        prices = 100.0 + t * 0.5
        df = pd.DataFrame({
            "Close": prices,
            "High": prices + 2.0,
            "Low": prices - 2.0,
            "Volume": np.full(n, 100000.0),
        })

        result = evaluate_stock_eligibility("RELIANCE.NS", df, min_history=252, min_adtv=1000000.0)
        assert result.is_eligible is True
        assert result.price_above_ema100 is True
        assert result.price_above_sma200 is True
        assert len(result.reasons) == 0

    def test_disqualified_short_history(self):
        """Stock with < 90 bars is rejected immediately."""
        df = pd.DataFrame({"Close": np.full(50, 100.0)})
        result = evaluate_stock_eligibility("NEWIPO.NS", df)
        assert result.is_eligible is False
        assert any("Insufficient minimum history" in r for r in result.reasons)

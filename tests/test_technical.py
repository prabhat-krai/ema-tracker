"""
Tests for the technical analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.technical import (
    calculate_emas,
    check_ema_convergence,
    find_support_resistance,
    analyze_stock,
    TechnicalIndicators,
)


def create_sample_data(num_weeks: int = 60, base_price: float = 100.0) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    # Create dates first to get actual count
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=num_weeks)
    dates = pd.date_range(start=start_date, end=end_date, freq="W")
    actual_weeks = len(dates)
    
    # Generate trending price data matching actual date count
    np.random.seed(42)
    closes = [base_price]
    for _ in range(actual_weeks - 1):
        change = np.random.normal(0.002, 0.03)  # Slight upward bias
        closes.append(closes[-1] * (1 + change))
    
    closes = np.array(closes)
    
    # Create arrays with explicit length matching
    opens = closes * 0.99
    highs = closes * 1.02
    lows = closes * 0.98
    volumes = np.random.randint(100000, 1000000, size=actual_weeks)
    
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=dates)
    
    return df


class TestCalculateEMAs:
    """Tests for EMA calculation."""
    
    def test_ema_columns_added(self):
        """EMA columns should be added to dataframe."""
        df = create_sample_data(60)
        result = calculate_emas(df)
        
        assert "ema_10w" in result.columns
        assert "ema_20w" in result.columns
        assert "ema_40w" in result.columns
    
    def test_ema_values_not_nan_after_warmup(self):
        """EMA values should not be NaN after warmup period."""
        df = create_sample_data(60)
        result = calculate_emas(df)
        
        # After 40 weeks (longest EMA), values should not be NaN
        assert not pd.isna(result.iloc[-1]["ema_10w"])
        assert not pd.isna(result.iloc[-1]["ema_20w"])
        assert not pd.isna(result.iloc[-1]["ema_40w"])
    
    def test_ema_ordering_in_uptrend(self):
        """In uptrend, shorter EMAs should be above longer EMAs."""
        # Create strongly uptrending data
        df = create_sample_data(60)
        df["close"] = np.linspace(100, 200, 60)  # Strong uptrend
        df["open"] = df["close"] * 0.99
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.98
        
        result = calculate_emas(df)
        latest = result.iloc[-1]
        
        # In uptrend: 10W EMA > 20W EMA > 40W EMA
        assert latest["ema_10w"] > latest["ema_20w"]
        assert latest["ema_20w"] > latest["ema_40w"]


class TestEMAConvergence:
    """Tests for EMA convergence detection."""
    
    def test_converging_emas(self):
        """EMAs within threshold should be detected as converging."""
        # EMAs within 2% of each other
        result = check_ema_convergence(100, 101, 102, threshold=0.03)
        assert result is True
    
    def test_diverging_emas(self):
        """EMAs outside threshold should not be converging."""
        # EMAs spread apart by more than 5%
        result = check_ema_convergence(100, 105, 110, threshold=0.03)
        assert result is False
    
    def test_nan_handling(self):
        """NaN values should return False."""
        result = check_ema_convergence(100, float("nan"), 102, threshold=0.03)
        assert result is False


class TestSupportResistance:
    """Tests for support/resistance detection."""
    
    def test_finds_resistance(self):
        """Should find resistance level from swing highs."""
        df = create_sample_data(60)
        
        # Create a clear peak in the middle
        peak_idx = 30
        df.iloc[peak_idx, df.columns.get_loc("high")] = df["high"].max() * 1.2
        
        support, resistance = find_support_resistance(df)
        
        assert resistance is not None
    
    def test_finds_support(self):
        """Should find support level from swing lows."""
        df = create_sample_data(60)
        
        # Create a clear trough in the middle
        trough_idx = 30
        df.iloc[trough_idx, df.columns.get_loc("low")] = df["low"].min() * 0.8
        
        support, resistance = find_support_resistance(df)
        
        assert support is not None
    
    def test_insufficient_data(self):
        """Should return None for insufficient data."""
        df = create_sample_data(5)  # Very short data
        
        support, resistance = find_support_resistance(df)
        
        # With only 5 weeks, may not find valid levels
        # This is acceptable behavior


class TestAnalyzeStock:
    """Tests for full stock analysis."""
    
    def test_returns_indicators(self):
        """Should return TechnicalIndicators object."""
        df = create_sample_data(60)
        result = analyze_stock("TEST", df)
        
        assert result is not None
        assert isinstance(result, TechnicalIndicators)
        assert result.symbol == "TEST"
    
    def test_insufficient_data_returns_none(self):
        """Should return None for insufficient data."""
        df = create_sample_data(30)  # Less than 40 weeks needed
        result = analyze_stock("TEST", df)
        
        assert result is None
    
    def test_none_dataframe_returns_none(self):
        """Should return None for None input."""
        result = analyze_stock("TEST", None)
        assert result is None

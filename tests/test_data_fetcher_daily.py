"""
Unit Tests for Daily Price Data Fetching, Benchmark Fetching, and Universe Pipeline.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.data_fetcher import (
    fetch_benchmark_daily_data,
    fetch_daily_data,
    fetch_universe_daily_data,
)


def _make_dummy_history(n_bars: int = 100, valid: bool = True) -> pd.DataFrame:
    """Helper to create dummy daily OHLCV DataFrame for testing."""
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="B")
    if not valid:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 150, n_bars),
            "High": np.linspace(102, 155, n_bars),
            "Low": np.linspace(98, 148, n_bars),
            "Close": np.linspace(101, 152, n_bars),
            "Volume": np.full(n_bars, 1_000_000.0),
        },
        index=dates,
    )


class TestFetchDailyData:
    """Tests for fetch_daily_data helper."""

    @patch("yfinance.Ticker")
    def test_fetch_daily_data_success_indian_ticker(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_dummy_history(100)
        mock_ticker_cls.return_value = mock_instance

        df = fetch_daily_data("RELIANCE", period="2y", min_bars=90, delay=0.0, market="india")

        assert df is not None
        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        mock_ticker_cls.assert_called_once_with("RELIANCE.NS")

    @patch("yfinance.Ticker")
    def test_fetch_daily_data_success_us_ticker(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_dummy_history(120)
        mock_ticker_cls.return_value = mock_instance

        df = fetch_daily_data("BRK.B", period="1y", min_bars=90, delay=0.0, market="usa")

        assert df is not None
        assert len(df) == 120
        mock_ticker_cls.assert_called_once_with("BRK-B")

    @patch("yfinance.Ticker")
    def test_fetch_daily_data_benchmark_symbol_preserved(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_dummy_history(150)
        mock_ticker_cls.return_value = mock_instance

        df = fetch_daily_data("^NSEI", period="2y", min_bars=90, delay=0.0)

        assert df is not None
        mock_ticker_cls.assert_called_once_with("^NSEI")

    @patch("yfinance.Ticker")
    def test_fetch_daily_data_empty_returns_none(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance

        df = fetch_daily_data("EMPTY", period="2y", min_bars=90, delay=0.0)
        assert df is None

    @patch("yfinance.Ticker")
    def test_fetch_daily_data_fewer_than_min_bars_returns_none(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_dummy_history(50)
        mock_ticker_cls.return_value = mock_instance

        df = fetch_daily_data("SHORT", period="2y", min_bars=90, delay=0.0)
        assert df is None

    @patch("yfinance.Ticker")
    def test_fetch_daily_data_handles_exception(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = RuntimeError("yfinance network error")

        df = fetch_daily_data("ERROR", period="2y", min_bars=90, delay=0.0)
        assert df is None


class TestFetchBenchmarkDailyData:
    """Tests for fetch_benchmark_daily_data helper."""

    @patch("src.data_fetcher.fetch_daily_data")
    def test_fetch_benchmark_daily_data_calls_fetch_daily(self, mock_fetch):
        mock_fetch.return_value = _make_dummy_history(100)

        res = fetch_benchmark_daily_data("^NSEI", period="2y", delay=0.0)
        assert res is not None
        mock_fetch.assert_called_once_with(
            ticker="^NSEI",
            period="2y",
            min_bars=90,
            delay=0.0,
        )


class TestFetchUniverseDailyData:
    """Tests for fetch_universe_daily_data helper."""

    @patch("src.data_fetcher.fetch_daily_data")
    def test_fetch_universe_with_custom_symbols(self, mock_fetch):
        mock_fetch.return_value = _make_dummy_history(100)
        progress_calls = []

        def callback(curr, tot, sym):
            progress_calls.append((curr, tot, sym))

        symbols = ["STOCK_A", "STOCK_B", "STOCK_C"]
        results = fetch_universe_daily_data(
            universe="india",
            symbols=symbols,
            period="1y",
            min_bars=90,
            delay=0.0,
            progress_callback=callback,
        )

        assert len(results) == 3
        assert set(results.keys()) == set(symbols)
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3, "STOCK_A")
        assert progress_calls[-1] == (3, 3, "STOCK_C")

    def test_fetch_universe_invalid_universe_raises(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            fetch_universe_daily_data(universe="crypto")

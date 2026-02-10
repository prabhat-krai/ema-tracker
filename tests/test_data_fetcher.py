"""
Tests for data fetcher utility helpers.
"""

from src.data_fetcher import get_nse_ticker


def test_get_nse_ticker_appends_ns_suffix():
    assert get_nse_ticker("RELIANCE") == "RELIANCE.NS"


def test_get_nse_ticker_normalizes_input():
    assert get_nse_ticker("  reliance  ") == "RELIANCE.NS"


def test_get_nse_ticker_preserves_ampersand_symbols():
    assert get_nse_ticker("M&M") == "M&M.NS"


def test_get_nse_ticker_does_not_double_append_suffix():
    assert get_nse_ticker("RELIANCE.NS") == "RELIANCE.NS"

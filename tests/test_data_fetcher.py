"""
Tests for data fetcher utility helpers.
"""

from src.data_fetcher import get_market_ticker, get_nse_ticker, get_us_ticker


def test_get_nse_ticker_appends_ns_suffix():
    assert get_nse_ticker("RELIANCE") == "RELIANCE.NS"


def test_get_nse_ticker_normalizes_input():
    assert get_nse_ticker("  reliance  ") == "RELIANCE.NS"


def test_get_nse_ticker_preserves_ampersand_symbols():
    assert get_nse_ticker("M&M") == "M&M.NS"


def test_get_nse_ticker_does_not_double_append_suffix():
    assert get_nse_ticker("RELIANCE.NS") == "RELIANCE.NS"


def test_get_us_ticker_normalizes_input():
    assert get_us_ticker("  msft  ") == "MSFT"


def test_get_us_ticker_converts_dot_to_dash():
    assert get_us_ticker("BRK.B") == "BRK-B"


def test_get_market_ticker_switches_by_market():
    assert get_market_ticker("RELIANCE", market="india") == "RELIANCE.NS"
    assert get_market_ticker("MSFT", market="usa") == "MSFT"

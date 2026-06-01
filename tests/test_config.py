"""
Tests for stock universe configuration.
"""

from src import config


def test_usa_universe_size_and_uniqueness():
    stocks = config.get_usa_stocks()
    assert len(stocks) >= 99
    assert len(set(stocks)) == len(stocks)

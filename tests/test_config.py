"""
Tests for stock universe configuration.
"""

from src import config


def test_usa_top_100_universe_size_and_uniqueness():
    stocks = config.get_usa_top_100_stocks()
    assert len(stocks) == 100
    assert len(set(stocks)) == 100

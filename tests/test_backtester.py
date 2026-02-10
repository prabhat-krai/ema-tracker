"""
Tests for backtester position management rules.
"""

from datetime import datetime, timedelta

from src.backtester import Portfolio
from src.ta_rules_engine import Signal, SignalResult


def make_result(symbol: str, signal: Signal, price: float) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        signal=signal,
        reason="test",
        current_price=price,
        ema_10w=price * 0.99,
        ema_20w=price * 0.98,
        ema_40w=price * 0.97,
        emas_converging=False,
    )


def test_enters_on_bullish_and_holds_on_wait():
    portfolio = Portfolio()
    t1 = datetime(2025, 1, 1)
    t2 = t1 + timedelta(weeks=1)

    portfolio.process_signal(t1, make_result("TEST", Signal.BULLISH, 100.0))
    assert "TEST" in portfolio.holdings

    portfolio.process_signal(t2, make_result("TEST", Signal.WAIT, 105.0))
    assert "TEST" in portfolio.holdings
    assert len(portfolio.closed_trades) == 0


def test_enters_on_hold_add_and_holds_on_cautious():
    portfolio = Portfolio()
    t1 = datetime(2025, 1, 1)
    t2 = t1 + timedelta(weeks=1)

    portfolio.process_signal(t1, make_result("TEST", Signal.HOLD_ADD, 200.0))
    assert "TEST" in portfolio.holdings

    portfolio.process_signal(t2, make_result("TEST", Signal.CAUTIOUS, 190.0))
    assert "TEST" in portfolio.holdings
    assert len(portfolio.closed_trades) == 0


def test_exits_on_exit_signal():
    portfolio = Portfolio()
    t1 = datetime(2025, 1, 1)
    t2 = t1 + timedelta(weeks=1)

    portfolio.process_signal(t1, make_result("TEST", Signal.BULLISH, 200.0))
    assert "TEST" in portfolio.holdings

    portfolio.process_signal(t2, make_result("TEST", Signal.EXIT, 190.0))
    assert "TEST" not in portfolio.holdings
    assert len(portfolio.closed_trades) == 1
    assert portfolio.closed_trades[0].return_pct == -0.05


def test_holds_position_while_signal_stays_entry_eligible():
    portfolio = Portfolio()
    t1 = datetime(2025, 1, 1)
    t2 = t1 + timedelta(weeks=1)
    t3 = t2 + timedelta(weeks=1)

    portfolio.process_signal(t1, make_result("TEST", Signal.BULLISH, 100.0))
    portfolio.process_signal(t2, make_result("TEST", Signal.BULLISH, 102.0))
    portfolio.process_signal(t3, make_result("TEST", Signal.HOLD_ADD, 104.0))

    assert len(portfolio.holdings) == 1
    assert len(portfolio.closed_trades) == 0

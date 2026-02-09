"""
Tests for the TA Rules Engine.
"""

import pytest
from unittest.mock import MagicMock

from src.ta_rules_engine import (
    Signal,
    SignalResult,
    analyze_with_ta_rules,
    format_signal_line,
    get_signal_emoji,
)
from src.technical import TechnicalIndicators


def create_mock_indicators(
    symbol: str = "TEST",
    current_price: float = 100.0,
    ema_10w: float = 98.0,
    ema_20w: float = 96.0,
    ema_40w: float = 94.0,
    emas_converging: bool = False,
    above_ema_10w: bool = True,
    above_ema_20w: bool = True,
    above_ema_40w: bool = True,
    broke_resistance: bool = False,
    broke_support: bool = False,
    resistance: float = 110.0,
    support: float = 90.0,
) -> TechnicalIndicators:
    """Create mock TechnicalIndicators for testing."""
    return TechnicalIndicators(
        symbol=symbol,
        current_price=current_price,
        ema_10w=ema_10w,
        ema_20w=ema_20w,
        ema_40w=ema_40w,
        resistance=resistance,
        support=support,
        emas_converging=emas_converging,
        above_ema_10w=above_ema_10w,
        above_ema_20w=above_ema_20w,
        above_ema_40w=above_ema_40w,
        broke_resistance=broke_resistance,
        broke_support=broke_support,
    )


class TestTARulesConverging:
    """Tests for converging EMAs branch of flowchart."""
    
    def test_converging_broke_support_is_exit(self):
        """Converging + broke support = EXIT signal."""
        indicators = create_mock_indicators(
            emas_converging=True,
            broke_support=True,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.EXIT
        assert "support" in result.reason.lower()
    
    def test_converging_below_support_is_exit(self):
        """Converging + price below support = EXIT signal."""
        indicators = create_mock_indicators(
            emas_converging=True,
            current_price=85.0,  # Below support of 90
            support=90.0,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.EXIT
    
    def test_converging_broke_resistance_is_bullish(self):
        """Converging + broke resistance = BULLISH signal."""
        indicators = create_mock_indicators(
            emas_converging=True,
            broke_resistance=True,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.BULLISH
        assert "resistance" in result.reason.lower() or "breakout" in result.reason.lower()
    
    def test_converging_above_resistance_is_bullish(self):
        """Converging + price above resistance = BULLISH signal."""
        indicators = create_mock_indicators(
            emas_converging=True,
            current_price=115.0,  # Above resistance of 110
            resistance=110.0,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.BULLISH
    
    def test_converging_no_breakout_is_wait(self):
        """Converging + no breakout = WAIT signal."""
        indicators = create_mock_indicators(
            emas_converging=True,
            current_price=100.0,
            support=90.0,
            resistance=110.0,
            broke_support=False,
            broke_resistance=False,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.WAIT


class TestTARulesNotConverging:
    """Tests for non-converging EMAs branch of flowchart."""
    
    def test_below_40w_ema_is_exit(self):
        """Below 40W EMA = EXIT signal."""
        indicators = create_mock_indicators(
            emas_converging=False,
            above_ema_40w=False,
            above_ema_20w=False,
            above_ema_10w=False,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.EXIT
        assert "40W" in result.reason
    
    def test_below_20w_ema_is_cautious(self):
        """Below 20W EMA (but above 40W) = CAUTIOUS signal."""
        indicators = create_mock_indicators(
            emas_converging=False,
            above_ema_40w=True,
            above_ema_20w=False,
            above_ema_10w=False,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.CAUTIOUS
        assert "20W" in result.reason
    
    def test_below_10w_ema_is_fading(self):
        """Below 10W EMA (but above 20W and 40W) = FADING signal."""
        indicators = create_mock_indicators(
            emas_converging=False,
            above_ema_40w=True,
            above_ema_20w=True,
            above_ema_10w=False,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.FADING
        assert "10W" in result.reason or "fading" in result.reason.lower()
    
    def test_above_all_emas_is_hold_add(self):
        """Above all EMAs = HOLD_ADD signal."""
        indicators = create_mock_indicators(
            emas_converging=False,
            above_ema_40w=True,
            above_ema_20w=True,
            above_ema_10w=True,
        )
        
        result = analyze_with_ta_rules(indicators)
        
        assert result.signal == Signal.HOLD_ADD


class TestSignalFormatting:
    """Tests for signal formatting utilities."""
    
    def test_get_signal_emoji(self):
        """Should return correct emoji for each signal."""
        assert get_signal_emoji(Signal.EXIT) == "🔴"
        assert get_signal_emoji(Signal.BULLISH) == "🟢"
        assert get_signal_emoji(Signal.WAIT) == "🟡"
        assert get_signal_emoji(Signal.CAUTIOUS) == "🟠"
        assert get_signal_emoji(Signal.FADING) == "🟣"
        assert get_signal_emoji(Signal.HOLD_ADD) == "🟢"
    
    def test_format_signal_line(self):
        """Should format signal result as readable line."""
        result = SignalResult(
            symbol="RELIANCE",
            signal=Signal.BULLISH,
            reason="Resistance breakout",
            current_price=2500.0,
            ema_10w=2400.0,
            ema_20w=2350.0,
            ema_40w=2300.0,
            emas_converging=True,
        )
        
        line = format_signal_line(result)
        
        assert "RELIANCE" in line
        assert "BULLISH" in line
        assert "2500" in line
        assert "🟢" in line

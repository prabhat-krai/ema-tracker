"""
TA Rules Engine - Implements the flowchart decision logic.

Flowchart:
1. Are EMAs converging?
   - YES: Check support/resistance breakouts
   - NO: Check EMA hierarchy
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals based on EMA TA Rules."""
    EXIT = "EXIT"              # Sell position
    BULLISH = "BULLISH"        # Buy signal
    WAIT = "WAIT"              # Watch list
    CAUTIOUS = "CAUTIOUS"      # Reduce exposure
    FADING = "FADING"          # Momentum weakening
    HOLD_ADD = "HOLD_ADD"      # Maintain or add position
    UNKNOWN = "UNKNOWN"        # Analysis inconclusive


@dataclass
class SignalResult:
    """Result of TA rules analysis for a stock."""
    symbol: str
    signal: Signal
    reason: str
    current_price: float
    ema_10w: float
    ema_20w: float
    ema_40w: float
    emas_converging: bool
    support: Optional[float] = None
    resistance: Optional[float] = None


def get_signal_emoji(signal: Signal) -> str:
    """Get emoji representation for a signal."""
    return {
        Signal.EXIT: "🔴",
        Signal.BULLISH: "🟢",
        Signal.WAIT: "🟡",
        Signal.CAUTIOUS: "🟠",
        Signal.FADING: "🟣",
        Signal.HOLD_ADD: "🟢",
        Signal.UNKNOWN: "⚪",
    }.get(signal, "⚪")


def analyze_with_ta_rules(indicators: TechnicalIndicators) -> SignalResult:
    """
    Apply TA Rules flowchart to technical indicators.
    
    Flowchart Logic:
    
    1. Are EMAs converging? (YES branch)
       ├─ Has it broken support? → EXIT
       └─ Has it broken resistance? → BULLISH
          └─ No breakout → WAIT/WATCH
    
    2. Are EMAs converging? (NO branch)
       ├─ Below 40W EMA? → EXIT
       ├─ Below 20W EMA? → BE CAUTIOUS
       ├─ Below 10W EMA? → MOMENTUM FADING
       └─ Above all EMAs → MAINTAIN/ADD
    
    Args:
        indicators: TechnicalIndicators from technical analysis
        
    Returns:
        SignalResult with signal and reasoning
    """
    symbol = indicators.symbol
    
    # Base result with indicators
    base_kwargs = {
        "symbol": symbol,
        "current_price": indicators.current_price,
        "ema_10w": indicators.ema_10w,
        "ema_20w": indicators.ema_20w,
        "ema_40w": indicators.ema_40w,
        "emas_converging": indicators.emas_converging,
        "support": indicators.support,
        "resistance": indicators.resistance,
    }
    
    # Branch 1: EMAs ARE converging
    if indicators.emas_converging:
        logger.debug(f"{symbol}: EMAs converging - checking support/resistance")
        has_support = indicators.support is not None
        has_resistance = indicators.resistance is not None
        
        # Check if broken support
        if indicators.broke_support or (has_support and indicators.current_price < indicators.support):
            return SignalResult(
                signal=Signal.EXIT,
                reason="Broke support with EMAs converging",
                **base_kwargs
            )
        
        # Check if broken resistance
        if indicators.broke_resistance or (has_resistance and indicators.current_price > indicators.resistance):
            return SignalResult(
                signal=Signal.BULLISH,
                reason="Resistance breakout with EMAs converging",
                **base_kwargs
            )
        
        # No clear breakout - wait and watch
        return SignalResult(
            signal=Signal.WAIT,
            reason="EMAs converging, no breakout yet",
            **base_kwargs
        )
    
    # Branch 2: EMAs are NOT converging
    else:
        logger.debug(f"{symbol}: EMAs not converging - checking EMA hierarchy")
        
        # Check if below 40W EMA (most bearish)
        if not indicators.above_ema_40w:
            return SignalResult(
                signal=Signal.EXIT,
                reason="Below 40W EMA",
                **base_kwargs
            )
        
        # Check if below 20W EMA
        if not indicators.above_ema_20w:
            return SignalResult(
                signal=Signal.CAUTIOUS,
                reason="Below 20W EMA",
                **base_kwargs
            )
        
        # Check if below 10W EMA
        if not indicators.above_ema_10w:
            return SignalResult(
                signal=Signal.FADING,
                reason="Below 10W EMA - momentum fading",
                **base_kwargs
            )
        
        # Above all EMAs - strong position
        return SignalResult(
            signal=Signal.HOLD_ADD,
            reason="Above all weekly EMAs",
            **base_kwargs
        )


def format_signal_line(result: SignalResult, currency_symbol: str = "₹") -> str:
    """Format a signal result as a single log line."""
    emoji = get_signal_emoji(result.signal)
    return (
        f"{emoji} {result.signal.value:10} | {result.symbol:15} | "
        f"{currency_symbol}{result.current_price:>10.2f} | {result.reason}"
    )

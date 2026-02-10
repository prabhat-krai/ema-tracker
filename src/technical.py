"""
Technical analysis module for EMA calculations and support/resistance detection.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from scipy.signal import find_peaks
import numpy as np

from . import config

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Container for technical indicators for a stock."""
    symbol: str
    current_price: float
    ema_10w: float
    ema_20w: float
    ema_40w: float
    resistance: Optional[float]
    support: Optional[float]
    emas_converging: bool
    
    # Price vs EMA comparisons
    above_ema_10w: bool
    above_ema_20w: bool
    above_ema_40w: bool
    
    # Breakout flags
    broke_resistance: bool
    broke_support: bool


def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 10W, 20W, and 40W EMAs on the dataframe.
    Uses pandas native ewm() method for compatibility.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with EMA columns added
    """
    df = df.copy()
    
    # Using pandas native ewm (exponential weighted moving average)
    df["ema_10w"] = df["close"].ewm(span=config.EMA_PERIODS["short"], adjust=False).mean()
    df["ema_20w"] = df["close"].ewm(span=config.EMA_PERIODS["medium"], adjust=False).mean()
    df["ema_40w"] = df["close"].ewm(span=config.EMA_PERIODS["long"], adjust=False).mean()
    
    return df


def check_ema_convergence(
    ema_10w: float,
    ema_20w: float,
    ema_40w: float,
    threshold: float = config.CONVERGENCE_THRESHOLD
) -> bool:
    """
    Check if EMAs are converging (within threshold of each other).
    
    Convergence indicates consolidation phase and potential breakout setup.
    
    Args:
        ema_10w: 10-week EMA value
        ema_20w: 20-week EMA value
        ema_40w: 40-week EMA value
        threshold: Maximum percentage spread to consider "converging"
        
    Returns:
        True if EMAs are converging
    """
    if any(pd.isna([ema_10w, ema_20w, ema_40w])):
        return False
    
    # Calculate the spread as percentage of the average
    emas = [ema_10w, ema_20w, ema_40w]
    avg = sum(emas) / 3
    max_ema = max(emas)
    min_ema = min(emas)
    if avg <= 0:
        return False
    
    spread = (max_ema - min_ema) / avg
    
    return spread <= threshold


def find_support_resistance(
    df: pd.DataFrame,
    lookback_weeks: int = config.SUPPORT_RESISTANCE_LOOKBACK_WEEKS,
    swing_lookback: int = config.SWING_LOOKBACK
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find support and resistance levels using swing highs and lows.
    
    Uses scipy.signal.find_peaks to detect local maxima (resistance)
    and minima (support) in price data.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_weeks: Number of weeks to look back for levels
        swing_lookback: Number of candles on each side for swing detection
        
    Returns:
        Tuple of (support_level, resistance_level)
    """
    # Use only recent data for finding levels
    df_recent = df.tail(lookback_weeks).copy()
    
    if len(df_recent) < swing_lookback * 2 + 1:
        logger.debug("Not enough data for support/resistance detection")
        return None, None
    
    highs = df_recent["high"].values
    lows = df_recent["low"].values
    
    # Find resistance (peaks in highs)
    resistance_peaks, _ = find_peaks(highs, distance=swing_lookback)
    
    # Find support (peaks in inverted lows = troughs in lows)
    support_peaks, _ = find_peaks(-lows, distance=swing_lookback)
    
    # Get the most recent resistance level
    resistance = None
    if len(resistance_peaks) > 0:
        # Get the highest recent peak as resistance
        recent_peaks = resistance_peaks[-3:] if len(resistance_peaks) >= 3 else resistance_peaks
        resistance = float(np.max(highs[recent_peaks]))
    
    # Get the most recent support level
    support = None
    if len(support_peaks) > 0:
        # Get the lowest recent trough as support
        recent_troughs = support_peaks[-3:] if len(support_peaks) >= 3 else support_peaks
        support = float(np.min(lows[recent_troughs]))
    
    return support, resistance


def analyze_stock(symbol: str, df: pd.DataFrame) -> Optional[TechnicalIndicators]:
    """
    Perform full technical analysis on a stock.
    
    Args:
        symbol: Stock symbol
        df: DataFrame with weekly OHLCV data
        
    Returns:
        TechnicalIndicators object or None if analysis fails
    """
    if df is None or len(df) < config.EMA_PERIODS["long"] + 10:
        logger.warning(f"{symbol}: Insufficient data for analysis")
        return None
    
    try:
        # Calculate EMAs
        df = calculate_emas(df)
        
        # Get latest values
        latest = df.iloc[-1]
        current_price = float(latest["close"])
        ema_10w = float(latest["ema_10w"])
        ema_20w = float(latest["ema_20w"])
        ema_40w = float(latest["ema_40w"])
        
        # Check for NaN values
        if any(pd.isna([current_price, ema_10w, ema_20w, ema_40w])):
            logger.warning(f"{symbol}: NaN values in indicators")
            return None
        
        # Find support and resistance
        support, resistance = find_support_resistance(df)
        
        # Check EMA convergence
        emas_converging = check_ema_convergence(ema_10w, ema_20w, ema_40w)
        
        # Price vs EMA comparisons
        above_ema_10w = current_price > ema_10w
        above_ema_20w = current_price > ema_20w
        above_ema_40w = current_price > ema_40w
        
        # Check for breakouts (comparing current price to recent levels)
        # We need to check if price recently crossed these levels
        prev_close = float(df.iloc[-2]["close"]) if len(df) > 1 else current_price
        
        broke_resistance = False
        broke_support = False
        
        if resistance is not None:
            # Broke resistance if current price is above but previous was below
            broke_resistance = current_price > resistance and prev_close <= resistance
        
        if support is not None:
            # Broke support if current price is below but previous was above
            broke_support = current_price < support and prev_close >= support
        
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
        
    except Exception as e:
        logger.error(f"{symbol}: Error during analysis - {e}")
        return None

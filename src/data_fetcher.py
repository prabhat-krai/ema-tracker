"""
Data fetcher module for retrieving stock price data from yfinance.
Handles NSE ticker conversion and rate limiting.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Literal

import pandas as pd
import yfinance as yf

from . import config

logger = logging.getLogger(__name__)


def get_nse_ticker(symbol: str) -> str:
    """
    Convert a stock symbol to NSE yfinance ticker format.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        
    Returns:
        NSE ticker for yfinance (e.g., "RELIANCE.NS")
    """
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def get_us_ticker(symbol: str) -> str:
    """
    Convert a US stock symbol to yfinance ticker format.

    Examples:
        BRK.B -> BRK-B
        msft -> MSFT
    """
    symbol = symbol.strip().upper()
    return symbol.replace(".", "-")


def get_market_ticker(symbol: str, market: Literal["india", "usa"] = "india") -> str:
    """Convert a symbol to a market-specific yfinance ticker."""
    if market == "usa":
        return get_us_ticker(symbol)
    return get_nse_ticker(symbol)


def fetch_weekly_data(
    symbol: str,
    years: int = config.HISTORY_YEARS,
    delay: float = config.API_DELAY_SECONDS,
    market: Literal["india", "usa"] = "india",
) -> Optional[pd.DataFrame]:
    """
    Fetch weekly OHLCV data for a stock from yfinance.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        years: Number of years of historical data to fetch
        delay: Seconds to wait after API call (rate limiting)
        market: Market identifier ("india" or "usa")
        
    Returns:
        DataFrame with weekly OHLCV data or None if fetch fails
    """
    ticker = get_market_ticker(symbol, market=market)
    
    try:
        logger.debug(f"Fetching data for {ticker}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Fetch data
        stock = yf.Ticker(ticker)
        df = stock.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1wk"
        )
        
        # Rate limiting
        time.sleep(delay)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        # Clean up the dataframe
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have required columns
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing columns for {symbol}: {df.columns.tolist()}")
            return None
        
        # Set date as index
        df.set_index("date", inplace=True)
        
        logger.debug(f"Successfully fetched {len(df)} weeks of data for {symbol}")
        return df[["open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        # Still apply rate limiting even on error
        time.sleep(delay)
        return None


def fetch_batch_data(
    symbols: list[str],
    years: int = config.HISTORY_YEARS,
    delay: float = config.API_DELAY_SECONDS,
    market: Literal["india", "usa"] = "india",
    progress_callback=None
) -> dict[str, Optional[pd.DataFrame]]:
    """
    Fetch weekly data for multiple stocks with progress tracking.
    
    Args:
        symbols: List of stock symbols
        years: Number of years of historical data
        delay: Seconds between API calls
        market: Market identifier ("india" or "usa")
        progress_callback: Optional callback(current, total, symbol) for progress
        
    Returns:
        Dictionary mapping symbol to DataFrame (or None if fetch failed)
    """
    results = {}
    total = len(symbols)
    
    for i, symbol in enumerate(symbols, 1):
        if progress_callback:
            progress_callback(i, total, symbol)
        
        results[symbol] = fetch_weekly_data(symbol, years, delay, market=market)
        
    return results

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


def fetch_daily_data(
    ticker: str,
    period: str = "2y",
    min_bars: int = 90,
    delay: float = 0.05,
    market: Optional[Literal["india", "usa"]] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV price data for a ticker with automatic retry and rate-limiting.

    Args:
        ticker: Stock symbol or benchmark ticker (e.g. "RELIANCE", "AAPL", "^NSEI", "^GSPC")
        period: yfinance historical period (default: "2y")
        min_bars: Minimum required daily bars (default: 90)
        delay: Seconds between API calls for rate limiting (default: 0.05)
        market: Market identifier ("india" or "usa", optional if ticker is formatted)

    Returns:
        DataFrame indexed by Datetime with columns: ['open', 'high', 'low', 'close', 'volume']
        or None if data is missing / length < min_bars.
    """
    raw_ticker = ticker.strip()
    if raw_ticker.startswith("^"):
        formatted_ticker = raw_ticker
    elif market is not None:
        formatted_ticker = get_market_ticker(raw_ticker, market=market)
    elif raw_ticker.upper().endswith(".NS"):
        formatted_ticker = raw_ticker.upper()
    elif "." in raw_ticker:
        formatted_ticker = get_us_ticker(raw_ticker)
    else:
        # Default as provided
        formatted_ticker = raw_ticker

    try:
        logger.debug(f"Fetching daily data for {formatted_ticker}")
        stock = yf.Ticker(formatted_ticker)
        df = stock.history(period=period, interval="1d", auto_adjust=True)

        if delay > 0:
            time.sleep(delay)

        if df is None or df.empty:
            logger.warning(f"No daily data returned for {formatted_ticker}")
            return None

        # Clean up columns and index
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]

        # Normalize date/datetime/index column to 'date'
        if "date" not in df.columns:
            if "datetime" in df.columns:
                df.rename(columns={"datetime": "date"}, inplace=True)
            elif "index" in df.columns:
                df.rename(columns={"index": "date"}, inplace=True)

        required_cols = ["date", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {formatted_ticker}: {df.columns.tolist()}")
            return None

        # Discard invalid prices
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        if len(df) < min_bars:
            logger.warning(
                f"Insufficient daily bars for {formatted_ticker}: got {len(df)}, required {min_bars}"
            )
            return None

        df.set_index("date", inplace=True)
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        logger.debug(f"Successfully fetched {len(df)} daily bars for {formatted_ticker}")
        return df[["open", "high", "low", "close", "volume"]]

    except Exception as e:
        logger.error(f"Error fetching daily data for {formatted_ticker}: {e}")
        if delay > 0:
            time.sleep(delay)
        return None


def fetch_benchmark_daily_data(
    benchmark_symbol: str = "^NSEI",
    period: str = "2y",
    delay: float = 0.05,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily benchmark index data (e.g. Nifty 50 '^NSEI', S&P 500 '^GSPC').

    Args:
        benchmark_symbol: Index ticker symbol (default: "^NSEI")
        period: Historical lookback period (default: "2y")
        delay: Seconds between API calls (default: 0.05)

    Returns:
        DataFrame with daily OHLCV data or None if fetch fails.
    """
    return fetch_daily_data(
        ticker=benchmark_symbol,
        period=period,
        min_bars=90,
        delay=delay,
    )


def fetch_universe_daily_data(
    universe: Literal["india", "usa"] = "india",
    period: str = "2y",
    min_bars: int = 90,
    delay: float = 0.05,
    symbols: Optional[list[str]] = None,
    progress_callback=None,
) -> dict[str, Optional[pd.DataFrame]]:
    """
    Fetch daily OHLCV data for all stocks in the selected universe with progress tracking.

    Args:
        universe: Target market universe ("india" or "usa")
        period: Historical period to fetch (default: "2y")
        min_bars: Minimum daily bars required (default: 90)
        delay: Seconds between requests (default: 0.05)
        symbols: Optional custom list of symbols. If None, uses universe constituents from config.
        progress_callback: Optional callback(current, total, symbol) for UI/CLI progress updates.

    Returns:
        Dictionary mapping symbol to daily DataFrame (or None if fetch failed / disqualified).
    """
    if symbols is None:
        if universe == "india":
            symbols = config.get_all_stocks()
        elif universe == "usa":
            symbols = config.get_usa_stocks()
        else:
            raise ValueError(f"Unknown universe: {universe}. Must be 'india' or 'usa'")

    results: dict[str, Optional[pd.DataFrame]] = {}
    total = len(symbols)

    logger.info(f"Starting daily data fetch for {total} {universe} stocks")

    for i, symbol in enumerate(symbols, 1):
        if progress_callback:
            progress_callback(i, total, symbol)

        df = fetch_daily_data(
            ticker=symbol,
            period=period,
            min_bars=min_bars,
            delay=delay,
            market=universe,
        )
        results[symbol] = df

    logger.info(f"Completed daily data fetch for {universe} universe: {len([d for d in results.values() if d is not None])}/{total} successful")
    return results


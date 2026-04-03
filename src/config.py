"""
Configuration for EMA TA Rules Screener.
Contains stock universe and analysis parameters.
"""

from typing import List
import logging
import requests
import time
import sys
import pandas as pd
import io

# EMA periods in weeks
EMA_PERIODS = {
    "short": 10,   # 10-week EMA
    "medium": 20,  # 20-week EMA
    "long": 40,    # 40-week EMA
}

# EMA convergence threshold (3% = EMAs are "converging" if within 3% of each other)
CONVERGENCE_THRESHOLD = 0.03

# Support/Resistance detection parameters
SWING_LOOKBACK = 5  # Number of candles on each side for swing detection
SUPPORT_RESISTANCE_LOOKBACK_WEEKS = 52  # Look back 1 year for levels

# Rate limiting for yfinance (seconds between requests)
API_DELAY_SECONDS = 2

# Years of historical data to fetch
HISTORY_YEARS = 2

logger = logging.getLogger(__name__)


# --- Stock Universe ---

def _fetch_nse_index(index_name: str) -> List[str]:
    """
    Fetches the list of stocks from the NSE website for a given index.
    """
    base_url = "https://www.nseindia.com/"
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_name}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.nseindia.com/market-data/live-equity-market",
    }
    
    session = requests.Session()
    
    try:
        logger.info(f"NSE fetch start for {index_name.replace('%20', ' ')}")

        # 1. Hit homepage to get cookies
        home_response = session.get(base_url, headers=headers, timeout=10)
        if home_response.status_code != 200:
            logger.warning(
                f"NSE homepage cookie request returned status {home_response.status_code} "
                f"for {index_name.replace('%20', ' ')}"
            )
        time.sleep(1)
        
        # 2. Get API data
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Filter generic metadata items (like 'NIFTY 100' summary item) by checking for 'priority' or existence of 'open'
            # The structure usually contains the index itself as an item, often with priority 1 or similar.
            # Real stocks usually have priority 0 or are just in the list.
            #Safest way: Exclude if symbol is the index name itself
            
            stocks = []
            for record in data.get('data', []):
                symbol = record.get('symbol')
                if symbol and symbol != index_name.replace("%20", " "):
                     stocks.append(symbol)
            
            if stocks:
                logger.info(
                    f"NSE fetch success for {index_name.replace('%20', ' ')}: {len(stocks)} symbols"
                )
                return stocks
            
            logger.warning(
                f"NSE fetch returned 0 symbols for {index_name.replace('%20', ' ')}"
            )
        else:
            logger.warning(
                f"NSE API request failed for {index_name.replace('%20', ' ')} "
                f"with status {response.status_code}"
            )
    except Exception as e:
        logger.warning(
            f"NSE fetch exception for {index_name.replace('%20', ' ')}: {e}"
        )
        
    return []


_NIFTY_100_FALLBACK = [
    "ETERNAL", "MOTHERSON", "TATASTEEL", "DMART", "TORNTPHARM",
    "ONGC", "M&M", "BAJAJ-AUTO", "NAUKRI", "TECHM",
    "POWERGRID", "LT", "EICHERMOT", "NTPC", "TCS",
    "MAXHEALTH", "GRASIM", "VEDL", "MARUTI", "IOC",
    "AXISBANK", "NESTLEIND", "TITAN", "INDHOTEL", "ICICIGI",
    "DIVISLAB", "ABB", "HINDUNILVR", "BAJAJHFL", "ICICIBANK",
    "BRITANNIA", "WIPRO", "JIOFIN", "HINDALCO", "HAL",
    "GAIL", "LODHA", "TMPV", "JSWENERGY", "TATAPOWER",
    "HYUNDAI", "SOLARINDS", "GODREJCP", "LTIM", "UNITDSPR",
    "DLF", "SIEMENS", "TRENT", "IRFC", "SUNPHARMA",
    "APOLLOHOSP", "INDIGO", "HAVELLS", "KOTAKBANK", "BAJAJFINSV",
    "JSWSTEEL", "SBILIFE", "JINDALSTEL", "CIPLA", "INFY",
    "TVSMOTOR", "BEL", "BOSCHLTD", "VBL", "PNB",
    "SBIN", "BANKBARODA", "COALINDIA", "RELIANCE", "LICI",
    "PFC", "BAJAJHLDNG", "ITC", "ULTRACEMCO", "ENRIN",
    "HDFCBANK", "HDFCLIFE", "ADANIGREEN", "CANBK", "AMBUJACEM",
    "ADANIPORTS", "ASIANPAINT", "PIDILITIND", "BPCL", "ADANIENT",
    "RECLTD", "TATACONSUM", "BHARTIARTL", "SHRIRAMFIN", "CGPOWER",
    "DRREDDY", "MAZDOCK", "ADANIPOWER", "ADANIENSOL", "BAJFINANCE",
    "HINDZINC", "HCLTECH", "SHREECEM", "CHOLAFIN", "ZYDUSLIFE",
]

_USA_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "GOOG", "META", "BRK.B", "AVGO", "TSLA",
    "JPM", "V", "UNH", "XOM", "LLY",
    "WMT", "JNJ", "MA", "PG", "ORCL",
    "COST", "HD", "MRK", "ABBV", "BAC",
    "KO", "CRM", "NFLX", "CVX", "ACN",
    "AMD", "PEP", "TMO", "MCD", "CSCO",
    "LIN", "ABT", "WFC", "GE", "DHR",
    "PM", "IBM", "TXN", "GS", "INTU",
    "QCOM", "RTX", "CAT", "NOW", "AMGN",
    "SPGI", "BLK", "AXP", "BKNG", "DIS",
    "LOW", "SCHW", "SYK", "PFE", "HON",
    "TJX", "ADP", "C", "MS", "UPS",
    "COP", "BA", "MDT", "MO", "GILD",
    "LRCX", "MU", "SO", "AMAT", "EQIX",
    "DE", "NEE", "DUK", "KLAC", "PANW",
    "ANET", "CMCSA", "CRWD", "ADI", "TT",
    "UNP", "NKE", "T", "VZ", "MMC",
    "CB", "CME", "ELV", "FI", "UBER",
    "BX", "PGR", "AON", "APH", "GD",
]


def get_all_stocks() -> List[str]:
    """Returns list of Nifty 500 constituents."""
    logger.info("Fetching NIFTY 500 stocks from NSE API")
    stocks = _fetch_nse_index("NIFTY%20500")
    if stocks:
        logger.info(f"NIFTY 500 source: NSE live list ({len(stocks)} stocks)")
        return stocks
    else:
        logger.warning(
            f"NIFTY 500 fetch failed, using fallback list ({len(_NIFTY_100_FALLBACK)} stocks)"
        )
        return _NIFTY_100_FALLBACK


def get_usa_stocks() -> List[str]:
    """Returns list of S&P 500 constituents."""
    logger.info("Fetching S&P 500 stocks from Wikipedia")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        sp500_df = tables[0]
        stocks = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P 500 source: Wikipedia live list ({len(stocks)} stocks)")
        return stocks
    except Exception as e:
        logger.warning(f"S&P 500 fetch failed: {e}. Using fallback list.")
        return _USA_FALLBACK

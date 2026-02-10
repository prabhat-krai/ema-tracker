"""
Configuration for EMA TA Rules Screener.
Contains stock universe and analysis parameters.
"""

from typing import List
import requests
import time
import sys

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
        # 1. Hit homepage to get cookies
        session.get(base_url, headers=headers, timeout=10)
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
                return stocks
    except Exception as e:
        # Silently fail here and let the caller handle fallback, or print debug info
        # Since we don't have the logger here, we can print to stderr if needed,
        # but for now we'll just return empty list to trigger fallback.
        pass
        
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

_MIDCAP_150_FALLBACK = [
    "TATACOMM", "BSE", "UBL", "SWIGGY", "NIACL",
    "TORNTPOWER", "LINDEINDIA", "IDBI", "GLAXO", "VMM",
    "TIINDIA", "GICRE", "3MINDIA", "MOTILALOFS", "MUTHOOTFIN",
    "TATAELXSI", "ASTRAL", "THERMAX", "HDFCAMC", "ASHOKLEY",
    "UNOMINDA", "PREMIERENE", "MRF", "360ONE", "IRCTC",
    "TATATECH", "DEEPAKNTR", "NTPCGREEN", "BHARATFORG", "COLPAL",
    "SAIL", "TATAINVEST", "BALKRISIND", "JUBLFOOD", "MAHABANK",
    "NYKAA", "PETRONET", "WAAREEENER", "GODREJPROP", "MARICO",
    "BLUESTARCO", "KPITTECH", "APOLLOTYRE", "NATIONALUM", "RVNL",
    "OFSS", "NMDC", "DABUR", "AJANTPHARM", "INDIANB",
    "INDUSTOWER", "ENDURANCE", "PHOENIXLTD", "ALKEM", "GUJGASLTD",
    "BHEL", "COFORGE", "M&MFIN", "PGHH", "LUPIN",
    "AIAENG", "OBEROIRLTY", "IRB", "APLAPOLLO", "GODFRYPHLP",
    "ESCORTS", "UPL", "IREDA", "ABCAPITAL", "COCHINSHIP",
    "LTTS", "CONCOR", "FORTIS", "UCOBANK", "SONACOMS",
    "HEROMOTOCO", "CUMMINSIND", "IGL", "SBICARD", "INDUSINDBK",
    "MPHASIS", "KEI", "IOB", "SUPREMEIND", "COROMANDEL",
    "GMRAIRPORT", "GODREJIND", "SYNGENE", "FACT", "PERSISTENT",
    "EXIDEIND", "BDL", "MFSL", "PRESTIGE", "AUBANK",
    "MANKIND", "ATGL", "DALBHARAT", "LICHSGFIN", "UNIONBANK",
    "POWERINDIA", "PAGEIND", "ITCHOTELS", "ABBOTINDIA", "ACC",
    "JSL", "CRISIL", "YESBANK", "AWL", "POLYCAB",
    "HUDCO", "SJVN", "BANKINDIA", "NHPC", "HINDPETRO",
    "BIOCON", "NAM-INDIA", "JSWINFRA", "HONAUT", "IDEA",
    "LLOYDSME", "SCHAEFFLER", "IDFCFIRSTB", "APARINDS", "GLENMARK",
    "LTF", "ICICIPRULI", "SUZLON", "KALYANKJIL", "POLICYBZR",
    "SRF", "OIL", "PIIND", "DIXON", "FLUOROCHEM",
    "VOLTAS", "BERGEPAINT", "FEDERALBNK", "PAYTM", "NLCINDIA",
    "PATANJALI", "IPCALAB", "MEDANTA", "GVT&D", "JKCEMENT",
    "SUNDARMFIN", "KPRMILL", "BHARTIHEXA", "HEXT", "AUROPHARMA",
]

_USA_TOP_100_FALLBACK = [
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


def get_nifty_100_stocks() -> List[str]:
    """
    Returns Nifty 100 constituents.
    Tries to fetch from NSE API, falls back to hardcoded list.
    """
    print("Fetching NIFTY 100 stocks...", end=" ", flush=True)
    stocks = _fetch_nse_index("NIFTY%20100")
    if stocks:
        print(f"Done ({len(stocks)} stocks found)")
        return stocks
    else:
        print("Failed (using fallback)")
        return _NIFTY_100_FALLBACK


def get_midcap_150_stocks() -> List[str]:
    """
    Returns Nifty Midcap 150 constituents.
    Tries to fetch from NSE API, falls back to hardcoded list.
    """
    print("Fetching NIFTY MIDCAP 150 stocks...", end=" ", flush=True)
    stocks = _fetch_nse_index("NIFTY%20MIDCAP%20150")
    if stocks:
        print(f"Done ({len(stocks)} stocks found)")
        return stocks
    else:
        print("Failed (using fallback)")
        return _MIDCAP_150_FALLBACK


def get_all_stocks() -> List[str]:
    """Returns combined list of Nifty 100 + Midcap 150 stocks."""
    nifty_100 = get_nifty_100_stocks()
    midcap_150 = get_midcap_150_stocks()
    
    # Remove duplicates while preserving order
    seen = set()
    all_stocks = []
    for stock in nifty_100 + midcap_150:
        if stock not in seen:
            seen.add(stock)
            all_stocks.append(stock)
    
    return all_stocks


def get_usa_top_100_stocks() -> List[str]:
    """Returns a fallback list of top 100 US stocks by size/liquidity."""
    print("Using USA Top 100 stock list (fallback)")
    return _USA_TOP_100_FALLBACK

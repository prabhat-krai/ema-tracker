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
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "BHARTIARTL", "SBIN", "BAJFINANCE", "KOTAKBANK",
    "ITC", "LT", "AXISBANK", "HCLTECH", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "ONGC",
    "NTPC", "BAJAJFINSV", "WIPRO", "POWERGRID", "M&M",
    "TATAMOTORS", "NESTLEIND", "JSWSTEEL", "COALINDIA", "TATASTEEL",
    "ADANIENT", "ADANIPORTS", "TECHM", "HDFCLIFE", "DRREDDY",
    "SBILIFE", "GRASIM", "DIVISLAB", "BRITANNIA", "CIPLA",
    "APOLLOHOSP", "EICHERMOT", "BAJAJ-AUTO", "INDUSINDBK", "HINDALCO",
    "TATACONSUM", "BPCL", "HEROMOTOCO", "SHREECEM", "VEDL",
    "DABUR", "GODREJCP", "HAVELLS", "PIDILITIND", "SIEMENS",
    "BIOCON", "BERGEPAINT", "AMBUJACEM", "MARICO", "SBICARD",
    "ICICIPRULI", "ICICIGI", "COLPAL", "TORNTPHARM", "DLF",
    "ACC", "BANDHANBNK", "MUTHOOTFIN", "NAUKRI", "PEL",
    "LUPIN", "CHOLAFIN", "INDIGO", "GAIL", "IOC",
    "HDFCAMC", "JUBLFOOD", "PGHH", "VOLTAS", "LICI",
    "TRENT", "LTIM", "BANKBARODA", "CANBK", "PNB",
    "UNIONBANK", "IDFCFIRSTB", "INDIANB", "FEDERALBNK", "ADANIGREEN",
    "ADANITRANS", "RECLTD", "PFC", "NHPC", "IRFC",
    "BEL", "HAL", "BHEL", "IRCTC", "ZOMATO",
]

_MIDCAP_150_FALLBACK = [
    "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ALKEM", "AMARAJABAT",
    "APLAPOLLO", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA",
    "BALRAMCHIN", "BATAINDIA", "BHARATFORG", "BHEL", "CAMS",
    "CANFINHOME", "CENTRALBK", "CGCL", "COFORGE", "CUB",
    "CUMMINSIND", "DEEPAKNTR", "DELHIVERY", "DIXON", "EMAMILTD",
    "ENDURANCE", "ESCORTS", "EXIDEIND", "FACT", "FSL",
    "GLENMARK", "GMRINFRA", "GNFC", "GRANULES", "GSPL",
    "HATSUN", "HINDZINC", "HONAUT", "IBULHSGFIN", "IDBI",
    "IEX", "IIFL", "IPCALAB", "IRB", "ISEC",
    "JKCEMENT", "JSWENERGY", "JUSTDIAL", "KAJARIACER", "KANSAINER",
    "KEI", "KPITTECH", "LATENTVIEW", "LAURUSLABS", "LICHSGFIN",
    "LLOYDSME", "LODHA", "LTF", "LTTS", "MANAPPURAM",
    "MANYAVAR", "MAPMYINDIA", "MAXHEALTH", "MCX", "METROPOLIS",
    "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF",
    "NAM-INDIA", "NATIONALUM", "NAVINFLUOR", "NBCC", "NCC",
    "NLCINDIA", "NMDC", "NOCIL", "OBEROIRLTY", "OFSS",
    "OIL", "PAGEIND", "PATANJALI", "PERSISTENT", "PETRONET",
    "PHOENIXLTD", "POLYCAB", "POLYPLEX", "PRESTIGE", "PVRINOX",
    "RAIN", "RAJESHEXPO", "RAMCOCEM", "RATNAMANI", "RAYMOND",
    "RELAXO", "RVNL", "SAIL", "SANOFI", "SCHAEFFLER",
    "SCI", "SFL", "SHRIRAMFIN", "SJVN", "SKFINDIA",
    "SONACOMS", "STARHEALTH", "SUNTV", "SUVENPHAR", "SYNGENE",
    "TATACOMM", "TATAELXSI", "TATAPOWER", "TATVA", "TEAMLEASE",
    "THERMAX", "TIINDIA", "TIMKEN", "TORNTPOWER", "TVSMOTOR",
    "UBL", "UJJIVAN", "UNOMINDA", "UPL", "VBL",
    "VINATIORGA", "VIPIND", "VMART", "WHIRLPOOL", "YESBANK",
    "ZEEL", "ZENSARTECH", "ZYDUSLIFE", "AFFLE", "CLEAN",
    "CROMPTON", "CESC", "INDIAMART", "INTELLECT", "JINDALSAW",
    "JSL", "JUBLPHARMA", "KALYANKJIL", "KIMS", "KRBL",
    "LAXMIMACH", "WESTLIFE", "APTUS", "AAVAS", "ALKYLAMINE",
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

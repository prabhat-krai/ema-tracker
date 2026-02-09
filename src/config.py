"""
Configuration for EMA TA Rules Screener.
Contains stock universe and analysis parameters.
"""

from typing import List

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


def get_nifty_100_stocks() -> List[str]:
    """
    Returns Nifty 100 constituents.
    This is periodically updated based on NSE index composition.
    """
    return [
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


def get_midcap_150_stocks() -> List[str]:
    """
    Returns Nifty Midcap 150 constituents (sample - top 150).
    This is periodically updated based on NSE index composition.
    """
    return [
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

"""
Comprehensive Test Fixtures & Deterministic Generators for E2E Momentum Testing.
Provides mathematical oracles, synthetic price histories, and multi-market universes.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def generate_deterministic_series(
    initial_price: float = 100.0,
    n_bars: int = 300,
    regime: str = "smooth_exponential_uptrend",
    daily_drift: float = 0.002,
    noise_std: float = 0.002,
    seed: int = 42,
    end_date: str = "2026-08-28",
) -> pd.DataFrame:
    """
    Generates a deterministic OHLCV DataFrame with controlled statistical properties.
    
    Regimes supported:
    - 'smooth_exponential_uptrend': High Clenow slope, R^2 > 0.90, Close > EMA100 > SMA200.
    - 'choppy_uptrend': High annualized slope, but low R^2 < 0.40 due to high volatility.
    - 'strong_downtrend': Negative slope, Close < EMA100 < SMA200.
    - 'flat_series': Constant price (Slope = 0, R^2 = 0, Vol = 0).
    - 'earnings_gap_up': Smooth trend with single +25% jump at t-15 days.
    - 'short_term_crash_skip_month': Strong up to t-21, then -20% crash in last month.
    - 'mean_reverting': Sinusoidal oscillating price series around mean.
    - 'low_history_80d': Exactly 80 bars (below 90d Clenow threshold).
    - 'medium_history_150d': Exactly 150 bars (valid for 90d/126d, invalid for 252d 12-1).
    - 'zero_volume': Regular price trend but volume = 0.
    - 'nan_series': Contains scattered NaN values in prices.
    - 'extreme_outlier': 1000% jump single day.
    """
    rng = np.random.RandomState(seed)
    
    # Adjust bar count for specific length regimes
    if regime == "low_history_80d":
        n_bars = 80
    elif regime == "medium_history_150d":
        n_bars = 150
        
    dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n_bars)
    t = np.arange(n_bars, dtype=np.float64)
    
    if regime == "smooth_exponential_uptrend":
        # log(P_t) = log(P_0) + drift * t + noise
        log_p = np.log(initial_price) + daily_drift * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)
    elif regime == "choppy_uptrend":
        # High slope but heavy noise reducing R^2 below 0.35
        log_p = np.log(initial_price) + (daily_drift * 1.5) * t + rng.normal(0, 0.25, n_bars)
        close = np.exp(log_p)
    elif regime == "strong_downtrend":
        # Negative drift
        log_p = np.log(initial_price) - abs(daily_drift) * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)
    elif regime == "flat_series":
        close = np.full(n_bars, initial_price, dtype=np.float64)
    elif regime == "earnings_gap_up":
        log_p = np.log(initial_price) + daily_drift * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)
        if n_bars > 20:
            close[-15:] *= 1.25  # +25% jump at t-15
    elif regime == "short_term_crash_skip_month":
        log_p = np.log(initial_price) + (daily_drift * 1.2) * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)
        if n_bars > 25:
            close[-20:] *= 0.80  # -20% crash in recent month
    elif regime == "mean_reverting":
        close = initial_price * (1.0 + 0.15 * np.sin(t * 2 * np.pi / 40.0) + rng.normal(0, 0.01, n_bars))
    elif regime == "zero_volume":
        log_p = np.log(initial_price) + daily_drift * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)
    elif regime == "nan_series":
        log_p = np.log(initial_price) + daily_drift * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)
        if n_bars > 10:
            close[10:15] = np.nan
            close[50] = np.nan
    elif regime == "extreme_outlier":
        log_p = np.log(initial_price) + daily_drift * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)
        if n_bars > 10:
            close[-5:] *= 11.0  # +1000% jump
    else:
        # Default smooth uptrend
        log_p = np.log(initial_price) + daily_drift * t + rng.normal(0, noise_std, n_bars)
        close = np.exp(log_p)

    # Derive Open, High, Low from Close
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    
    daily_vol = np.abs(rng.normal(0.005, 0.002, n_bars))
    high_p = np.maximum(close, open_p) * (1.0 + daily_vol)
    low_p = np.minimum(close, open_p) * (1.0 - daily_vol)
    
    if regime == "zero_volume":
        volume = np.zeros(n_bars, dtype=np.float64)
    else:
        base_vol = 1_000_000.0
        volume = base_vol * (1.0 + rng.uniform(-0.2, 0.5, n_bars))
        volume = np.maximum(volume, 1000.0)

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def generate_mock_benchmark(
    regime: str = "bullish",
    n_bars: int = 300,
    symbol: str = "^NSEI",
    seed: int = 123,
    end_date: str = "2026-08-28",
) -> pd.DataFrame:
    """
    Generates a deterministic benchmark index DataFrame (^NSEI / ^GSPC).
    
    - 'bullish': Cleanly above ascending 200 SMA.
    - 'neutral': Above 200 SMA but descending slope over last 20 days.
    - 'bearish': Dropping below 200 SMA (macro crash regime).
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n_bars)
    t = np.arange(n_bars, dtype=np.float64)
    
    if symbol == "^NSEI":
        p0 = 20000.0
    else:
        p0 = 5000.0

    if regime == "bullish":
        drift = 0.0008
        log_p = np.log(p0) + drift * t + rng.normal(0, 0.003, n_bars)
        close = np.exp(log_p)
    elif regime == "neutral":
        # Peak at t=85, trough at 200, current near 22500 above 200 SMA with negative SMA slope
        scale = p0 / 22000.0
        close = (22000.0 + 3000.0 * np.exp(-((t - 85)/30.0)**2) - 1500.0 * np.exp(-((t - 200)/50.0)**2) + 500.0 * (t > 270)) * scale
        close += rng.normal(0, 5.0, n_bars)
    elif regime == "bearish":
        # Sharp downtrend dropping below 200 SMA
        log_p = np.log(p0) + 0.0005 * np.minimum(t, 180) - 0.0025 * np.maximum(0, t - 180) + rng.normal(0, 0.005, n_bars)
        close = np.exp(log_p)
    else:
        log_p = np.log(p0) + 0.0006 * t + rng.normal(0, 0.004, n_bars)
        close = np.exp(log_p)

    df = pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n_bars, 50_000_000.0),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def generate_mock_indian_universe(
    n_stocks: int = 30,
    seed: int = 101,
) -> Dict[str, pd.DataFrame]:
    """
    Generates a deterministic universe of Indian NSE stocks with varied profiles.
    """
    universe = {}
    
    # 1. Top Leaders (Strong smooth exponential uptrend, high R^2, high 12-1 mom)
    leaders = [
        ("TRENT.NS", 6000.0, 0.0030, 0.002),
        ("DIXON.NS", 11000.0, 0.0028, 0.003),
        ("BEL.NS", 280.0, 0.0026, 0.002),
        ("HAL.NS", 4200.0, 0.0025, 0.003),
        ("SUZLON.NS", 65.0, 0.0027, 0.004),
        ("BHEL.NS", 260.0, 0.0024, 0.003),
        ("POWERGRID.NS", 310.0, 0.0020, 0.0015),
        ("COALINDIA.NS", 480.0, 0.0021, 0.002),
        ("VBL.NS", 1450.0, 0.0022, 0.002),
        ("CUMMINSIND.NS", 3400.0, 0.0020, 0.0025),
    ]
    for i, (sym, p0, drift, noise) in enumerate(leaders):
        universe[sym] = generate_deterministic_series(
            initial_price=p0,
            n_bars=300,
            regime="smooth_exponential_uptrend",
            daily_drift=drift,
            noise_std=noise,
            seed=seed + i * 10,
        )

    # 2. Choppy Momentum (high slope but erratic, lower R^2)
    choppy = [
        ("TATAMOTORS.NS", 950.0),
        ("ZOMATO.NS", 220.0),
        ("PAYTM.NS", 450.0),
        ("ADANIENT.NS", 3100.0),
        ("JIOFIN.NS", 340.0),
    ]
    for i, (sym, p0) in enumerate(choppy):
        universe[sym] = generate_deterministic_series(
            initial_price=p0,
            n_bars=300,
            regime="choppy_uptrend",
            daily_drift=0.0025,
            noise_std=0.10,
            seed=seed + 100 + i * 10,
        )

    # 3. Downtrend / Broken Trend (Fails EMA100/SMA200)
    downtrends = [
        ("INFY.NS", 1700.0),
        ("HDFCBANK.NS", 1600.0),
        ("KOTAKBANK.NS", 1750.0),
        ("ASIANPAINT.NS", 2900.0),
        ("BAJFINANCE.NS", 6800.0),
    ]
    for i, (sym, p0) in enumerate(downtrends):
        universe[sym] = generate_deterministic_series(
            initial_price=p0,
            n_bars=300,
            regime="strong_downtrend",
            daily_drift=0.0015,
            noise_std=0.003,
            seed=seed + 200 + i * 10,
        )

    # 4. Special cases: gap-up, low history, zero volume
    universe["GAPSTOCK.NS"] = generate_deterministic_series(
        initial_price=500.0,
        n_bars=300,
        regime="earnings_gap_up",
        seed=seed + 301,
    )
    universe["IPONEW.NS"] = generate_deterministic_series(
        initial_price=200.0,
        regime="low_history_80d",
        seed=seed + 302,
    )
    universe["MEDHIST.NS"] = generate_deterministic_series(
        initial_price=350.0,
        regime="medium_history_150d",
        seed=seed + 303,
    )
    universe["ZEROVOL.NS"] = generate_deterministic_series(
        initial_price=100.0,
        n_bars=300,
        regime="zero_volume",
        seed=seed + 304,
    )
    universe["FLATSTOCK.NS"] = generate_deterministic_series(
        initial_price=100.0,
        n_bars=300,
        regime="flat_series",
        seed=seed + 305,
    )
    
    return universe


def generate_mock_usa_universe(
    n_stocks: int = 30,
    seed: int = 202,
) -> Dict[str, pd.DataFrame]:
    """
    Generates a deterministic universe of US S&P 500 stocks with varied profiles.
    """
    universe = {}
    
    # 1. Top Leaders (Strong smooth exponential uptrend, high R^2)
    leaders = [
        ("NVDA", 120.0, 0.0032, 0.003),
        ("LLY", 850.0, 0.0028, 0.002),
        ("AVGO", 1500.0, 0.0026, 0.0025),
        ("META", 500.0, 0.0025, 0.002),
        ("MSFT", 420.0, 0.0022, 0.002),
        ("AAPL", 220.0, 0.0021, 0.0018),
        ("AMZN", 180.0, 0.0020, 0.002),
        ("COST", 820.0, 0.0019, 0.0015),
        ("NFLX", 650.0, 0.0022, 0.0025),
        ("AMD", 160.0, 0.0023, 0.003),
    ]
    for i, (sym, p0, drift, noise) in enumerate(leaders):
        universe[sym] = generate_deterministic_series(
            initial_price=p0,
            n_bars=300,
            regime="smooth_exponential_uptrend",
            daily_drift=drift,
            noise_std=noise,
            seed=seed + i * 10,
        )

    # 2. Choppy Momentum
    choppy = [
        ("TSLA", 210.0),
        ("PLTR", 28.0),
        ("COIN", 220.0),
        ("SMCI", 550.0),
        ("ARM", 130.0),
    ]
    for i, (sym, p0) in enumerate(choppy):
        universe[sym] = generate_deterministic_series(
            initial_price=p0,
            n_bars=300,
            regime="choppy_uptrend",
            daily_drift=0.0026,
            noise_std=0.10,
            seed=seed + 100 + i * 10,
        )

    # 3. Downtrend / Underperforming
    downtrends = [
        ("INTC", 30.0),
        ("WBA", 12.0),
        ("NKE", 78.0),
        ("BA", 170.0),
        ("DIS", 92.0),
    ]
    for i, (sym, p0) in enumerate(downtrends):
        universe[sym] = generate_deterministic_series(
            initial_price=p0,
            n_bars=300,
            regime="strong_downtrend",
            daily_drift=0.0016,
            noise_std=0.003,
            seed=seed + 200 + i * 10,
        )

    # 4. Special cases
    universe["BRK-B"] = generate_deterministic_series(
        initial_price=430.0,
        n_bars=300,
        regime="smooth_exponential_uptrend",
        daily_drift=0.0012,
        noise_std=0.0015,
        seed=seed + 301,
    )
    universe["NEWIPO"] = generate_deterministic_series(
        initial_price=50.0,
        regime="low_history_80d",
        seed=seed + 302,
    )
    universe["MEDUS"] = generate_deterministic_series(
        initial_price=75.0,
        regime="medium_history_150d",
        seed=seed + 303,
    )
    universe["FLATUS"] = generate_deterministic_series(
        initial_price=100.0,
        n_bars=300,
        regime="flat_series",
        seed=seed + 304,
    )

    return universe


# ==============================================================================
# Closed-Form Mathematical Oracle Helper Functions
# ==============================================================================

def oracle_exponential_regression(prices: np.ndarray, trading_days: int = 250) -> Tuple[float, float, float]:
    """
    Authoritative reference oracle for Clenow exponential regression:
    y = ln(P_t) = alpha + beta * t
    slope_ann = exp(beta * trading_days) - 1
    r_squared = (cov(t, y)^2) / (var(t) * var(y))
    clenow_score = slope_ann * r_squared
    """
    N = len(prices)
    if N < 2:
        return (np.nan, np.nan, np.nan)
    
    y = np.log(prices)
    t = np.arange(N, dtype=np.float64)
    
    t_mean = (N - 1.0) / 2.0
    y_mean = np.mean(y)
    
    t_dev = t - t_mean
    y_dev = y - y_mean
    
    var_t = np.sum(t_dev ** 2)
    var_y = np.sum(y_dev ** 2)
    cov_ty = np.sum(t_dev * y_dev)
    
    if var_t <= 0:
        return (0.0, 0.0, 0.0)
    
    beta = cov_ty / var_t
    slope_ann = np.exp(beta * trading_days) - 1.0
    
    if var_y <= 1e-14:
        r2 = 0.0
    else:
        r2 = (cov_ty ** 2) / (var_t * var_y)
        r2 = float(np.clip(r2, 0.0, 1.0))
        
    score = slope_ann * r2
    return (float(slope_ann), float(r2), float(score))


def oracle_jegadeesh_titman_12_1(prices: np.ndarray, lookback_total: int = 252, skip_recent: int = 21) -> float:
    """
    Authoritative reference oracle for 12-1 momentum:
    (P_{t-21} - P_{t-252}) / P_{t-252}
    """
    if len(prices) < lookback_total:
        return np.nan
    p_start = prices[-lookback_total]
    p_end = prices[-skip_recent - 1]
    if p_start <= 0:
        return np.nan
    return float((p_end - p_start) / p_start)


def oracle_realized_volatility(prices: np.ndarray, trading_days: int = 252) -> float:
    """
    Authoritative reference oracle for annualized realized log volatility:
    sigma_ann = std(diff(ln(P_t)), ddof=1) * sqrt(trading_days)
    """
    if len(prices) < 2:
        return np.nan
    log_ret = np.diff(np.log(prices))
    daily_std = np.std(log_ret, ddof=1)
    return float(daily_std * np.sqrt(trading_days))


def oracle_inverse_volatility_weights(volatilities: np.ndarray, min_vol: float = 0.05) -> np.ndarray:
    """
    Authoritative reference oracle for inverse-volatility weighting:
    w_i = (1 / max(vol_i, min_vol)) / sum(1 / max(vol_j, min_vol))
    """
    floored_vols = np.maximum(volatilities, min_vol)
    inv_vols = 1.0 / floored_vols
    total_inv = np.sum(inv_vols)
    if total_inv <= 0:
        return np.full_like(volatilities, 1.0 / len(volatilities))
    return inv_vols / total_inv


def oracle_bounded_risk_parity_weights(
    volatilities: np.ndarray,
    w_min: float = 0.05,
    w_max: float = 0.20,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Authoritative reference oracle for bounded risk parity projection.
    Ensures sum(w_i) == 1.0 and w_min <= w_i <= w_max.
    """
    n = len(volatilities)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n * w_min > 1.0 or n * w_max < 1.0:
        return np.full(n, 1.0 / n, dtype=np.float64)
        
    raw_inv_vol = 1.0 / np.maximum(volatilities, 1e-4)
    weights = raw_inv_vol / np.sum(raw_inv_vol)
    
    for _ in range(max_iter):
        clipped = np.clip(weights, w_min, w_max)
        excess = 1.0 - np.sum(clipped)
        if abs(excess) < 1e-8:
            return clipped
            
        if excess > 0:
            eligible = clipped < (w_max - 1e-8)
            if not np.any(eligible):
                return clipped
            weights = clipped.copy()
            weights[eligible] += excess * (weights[eligible] / np.sum(weights[eligible]))
        else:
            eligible = clipped > (w_min + 1e-8)
            if not np.any(eligible):
                return clipped
            weights = clipped.copy()
            weights[eligible] += excess * (weights[eligible] / np.sum(weights[eligible]))
            
    return np.clip(weights, w_min, w_max)

"""
Trend Regime Qualification, Moving Average Filters & Macro Market Regime.

Implements single-stock trend qualification rules and macro benchmark gates:
    - Moving Average Filters: Close > EMA_100 and Close > SMA_200
    - Golden Alignment: EMA_10 > EMA_50 > SMA_200 (or EMA_50 > EMA_100 > SMA_200)
    - 52-Week High Distance: Close / max(High_252) >= 0.75 (within 25% of high)
    - Macro Market Regime: Benchmark > SMA_200 and SMA_200 ascending -> BULLISH (100%),
      NEUTRAL (50%), BEARISH (0% equity allocation / cash defensive)

References:
    - Faber, M. T. (2007), 'A Quantitative Approach to Tactical Asset Allocation',
      The Journal of Wealth Management.
    - Clenow, A. (2015), 'Stocks on the Move'.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.momentum.models import FilterResult, MarketRegime, MarketRegimeState


def calculate_sma(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    window: int = 200,
) -> np.ndarray:
    """
    Computes Simple Moving Average (SMA) series for 1D prices.
    Returns array of same length with initial values padded or NaN.
    """
    arr = np.asarray(prices, dtype=np.float64).flatten()
    n = len(arr)
    sma = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 1:
        return sma

    # Vectorized cumulative sum for O(N) calculation
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    sma[window - 1:] = (cumsum[window:] - cumsum[:-window]) / float(window)
    return sma


def calculate_ema(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    span: int = 100,
) -> np.ndarray:
    """
    Computes Exponential Moving Average (EMA) series for 1D prices.
    Uses pandas ewm or NumPy equivalent for exact alignment.
    """
    arr = np.asarray(prices, dtype=np.float64).flatten()
    n = len(arr)
    if n == 0 or span < 1:
        return np.full(n, np.nan, dtype=np.float64)

    s = pd.Series(arr)
    ema_series = s.ewm(span=span, adjust=False).mean().to_numpy()
    return ema_series


def check_single_stock_trend_filter(
    close_prices: Union[np.ndarray, pd.Series, Sequence[float]],
    ema_fast: int = 100,
    sma_slow: int = 200,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates whether a single stock satisfies the macro moving average trend filters:
    Close > EMA_100 AND Close > SMA_200.

    Args:
        close_prices: 1D historical close prices.
        ema_fast: Fast EMA period (default: 100).
        sma_slow: Slow SMA period (default: 200).

    Returns:
        Tuple of (passes_filter: bool, details_dict: dict)
    """
    arr = np.asarray(close_prices, dtype=np.float64).flatten()
    n = len(arr)

    details: Dict[str, Any] = {
        "history_bars": n,
        "latest_close": np.nan,
        "ema_fast": np.nan,
        "sma_slow": np.nan,
        "above_ema_fast": False,
        "above_sma_slow": False,
        "passes": False,
        "reason": "",
    }

    if n < max(ema_fast, sma_slow):
        details["reason"] = f"Insufficient history: {n} bars < {max(ema_fast, sma_slow)} required"
        return False, details

    latest_close = float(arr[-1])
    ema_arr = calculate_ema(arr, span=ema_fast)
    sma_arr = calculate_sma(arr, window=sma_slow)

    latest_ema = float(ema_arr[-1])
    latest_sma = float(sma_arr[-1])

    above_ema = bool(latest_close > latest_ema)
    above_sma = bool(latest_close > latest_sma)
    passes = bool(above_ema and above_sma)

    details["latest_close"] = latest_close
    details["ema_fast"] = latest_ema
    details["sma_slow"] = latest_sma
    details["above_ema_fast"] = above_ema
    details["above_sma_slow"] = above_sma
    details["passes"] = passes

    if not passes:
        reasons = []
        if not above_ema:
            reasons.append(f"Close ({latest_close:.2f}) <= EMA_{ema_fast} ({latest_ema:.2f})")
        if not above_sma:
            reasons.append(f"Close ({latest_close:.2f}) <= SMA_{sma_slow} ({latest_sma:.2f})")
        details["reason"] = "; ".join(reasons)
    else:
        details["reason"] = "Passes moving average trend filters"

    return passes, details


def check_golden_regime(
    close_prices: Union[np.ndarray, pd.Series, Sequence[float]],
    ema_short: int = 10,
    ema_mid: int = 50,
    sma_long: int = 200,
) -> bool:
    """
    Checks if moving averages exhibit bullish Golden Alignment:
    EMA_short > EMA_mid > SMA_long (e.g. 10 > 50 > 200 or 50 > 100 > 200).
    """
    arr = np.asarray(close_prices, dtype=np.float64).flatten()
    if len(arr) < sma_long:
        return False

    ema_s = float(calculate_ema(arr, span=ema_short)[-1])
    ema_m = float(calculate_ema(arr, span=ema_mid)[-1])
    sma_l = float(calculate_sma(arr, window=sma_long)[-1])

    if np.isnan(ema_s) or np.isnan(ema_m) or np.isnan(sma_l):
        return False

    return bool(ema_s > ema_m > sma_l)


def calculate_52w_high_distance(
    close_prices: Union[np.ndarray, pd.Series, Sequence[float]],
    high_prices: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
    lookback: int = 252,
) -> float:
    """
    Calculates distance to 52-week High: Close_t / max(High_{252}).
    Returns a float in [0.0, 1.0] (1.0 = at or above 52w high).
    """
    c_arr = np.asarray(close_prices, dtype=np.float64).flatten()
    if len(c_arr) == 0:
        return 0.0

    if high_prices is not None:
        h_arr = np.asarray(high_prices, dtype=np.float64).flatten()
    else:
        h_arr = c_arr

    window_len = min(len(h_arr), lookback)
    if window_len < 1:
        return 0.0

    window_high = h_arr[-window_len:]
    max_high = float(np.max(window_high))
    latest_close = float(c_arr[-1])

    if max_high <= 0.0 or np.isnan(max_high) or latest_close <= 0.0:
        return 0.0

    return float(latest_close / max_high)


def evaluate_market_regime(
    benchmark_prices: Optional[Union[np.ndarray, pd.Series, Sequence[float]]],
    benchmark_symbol: str = "^NSEI",
    lookback_sma: int = 200,
    slope_window: int = 20,
) -> MarketRegime:
    """
    Evaluates macro market regime for a benchmark index:
        - BULLISH: Benchmark > SMA_200 and SMA_200(t) >= SMA_200(t-20) -> 100% allocation
        - NEUTRAL: Benchmark > SMA_200 and SMA_200(t) < SMA_200(t-20)  -> 50% allocation
        - BEARISH: Benchmark <= SMA_200                                 -> 0% equity allocation

    Defaults to NEUTRAL defensive mode if benchmark data is unavailable or insufficient.
    """
    if benchmark_prices is None:
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            state=MarketRegimeState.NEUTRAL,
            benchmark_price=0.0,
            sma_200=0.0,
            sma_200_slope=0.0,
            equity_allocation_pct=0.50,
            details={"warning": "Benchmark prices not provided; defaulting to NEUTRAL"},
        )

    arr = np.asarray(benchmark_prices, dtype=np.float64).flatten()
    n = len(arr)

    if n < lookback_sma:
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            state=MarketRegimeState.NEUTRAL,
            benchmark_price=float(arr[-1]) if n > 0 else 0.0,
            sma_200=0.0,
            sma_200_slope=0.0,
            equity_allocation_pct=0.50,
            details={"warning": f"Insufficient benchmark history ({n} < {lookback_sma} bars)"},
        )

    latest_price = float(arr[-1])
    sma_series = calculate_sma(arr, window=lookback_sma)
    latest_sma = float(sma_series[-1])

    if n >= lookback_sma + slope_window:
        sma_past = float(sma_series[-slope_window - 1])
    else:
        sma_past = float(sma_series[lookback_sma - 1])

    sma_slope = latest_sma - sma_past

    if latest_price > latest_sma:
        if sma_slope >= 0.0:
            state = MarketRegimeState.BULLISH
            allocation = 1.0
        else:
            state = MarketRegimeState.NEUTRAL
            allocation = 0.50
    else:
        state = MarketRegimeState.BEARISH
        allocation = 0.0

    return MarketRegime(
        benchmark_symbol=benchmark_symbol,
        state=state,
        benchmark_price=latest_price,
        sma_200=latest_sma,
        sma_200_slope=sma_slope,
        equity_allocation_pct=allocation,
        details={
            "benchmark_price": latest_price,
            "sma_200": latest_sma,
            "sma_slope_20d": sma_slope,
            "history_bars": n,
        },
    )


def evaluate_stock_eligibility(
    symbol: str,
    df: pd.DataFrame,
    min_history: int = 252,
    min_adtv: float = 0.0,
    min_dist_52w: float = 0.75,
) -> FilterResult:
    """
    Comprehensive stock eligibility evaluation against liquidity, history, and macro trend rules.
    """
    if df is None or df.empty:
        return FilterResult(
            symbol=symbol,
            is_eligible=False,
            reasons=["No price data provided"],
        )

    # Standardize column names
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "Close")
    high_col = col_map.get("high", "High")
    volume_col = col_map.get("volume", "Volume")

    if close_col not in df.columns:
        return FilterResult(
            symbol=symbol,
            is_eligible=False,
            reasons=["Missing Close column"],
        )

    close_s = df[close_col].dropna()
    n = len(close_s)
    reasons = []

    if n < 90:
        return FilterResult(
            symbol=symbol,
            is_eligible=False,
            reasons=[f"Insufficient minimum history ({n} < 90 bars)"],
            history_days=n,
        )

    if n < min_history:
        reasons.append(f"History below optimal lookback ({n} < {min_history} bars)")

    close_arr = close_s.to_numpy(dtype=np.float64)
    latest_close = float(close_arr[-1])

    if latest_close <= 0.0:
        return FilterResult(
            symbol=symbol,
            is_eligible=False,
            reasons=["Non-positive latest closing price"],
            history_days=n,
            price=latest_close,
        )

    # Moving averages
    ema_50 = float(calculate_ema(close_arr, span=50)[-1]) if n >= 50 else np.nan
    ema_100 = float(calculate_ema(close_arr, span=100)[-1]) if n >= 100 else np.nan
    sma_200 = float(calculate_sma(close_arr, window=200)[-1]) if n >= 200 else np.nan

    above_ema100 = bool(latest_close > ema_100) if not np.isnan(ema_100) else False
    above_sma200 = bool(latest_close > sma_200) if not np.isnan(sma_200) else False
    golden = bool(ema_50 > ema_100 > sma_200) if (not np.isnan(ema_50) and not np.isnan(ema_100) and not np.isnan(sma_200)) else False

    if not above_ema100:
        reasons.append(f"Close ({latest_close:.2f}) <= EMA_100 ({ema_100:.2f})")
    if not above_sma200 and not np.isnan(sma_200):
        reasons.append(f"Close ({latest_close:.2f}) <= SMA_200 ({sma_200:.2f})")

    # 52w high distance
    high_arr = df[high_col].dropna().to_numpy(dtype=np.float64) if high_col in df.columns else close_arr
    dist_52w = calculate_52w_high_distance(close_arr, high_arr, lookback=252)

    if dist_52w < min_dist_52w:
        reasons.append(f"Distance to 52W high ({dist_52w:.2%}) < {min_dist_52w:.2%}")

    # ADTV liquidity
    adtv_20 = 0.0
    if volume_col in df.columns and min_adtv > 0.0:
        vol_s = df[volume_col].dropna()
        if len(vol_s) >= 20 and len(close_s) >= 20:
            turnover_20 = (vol_s.iloc[-20:] * close_s.iloc[-20:]).mean()
            adtv_20 = float(turnover_20)
            if adtv_20 < min_adtv:
                reasons.append(f"20-day ADTV ({adtv_20:,.0f}) < minimum ({min_adtv:,.0f})")

    is_eligible = len(reasons) == 0

    return FilterResult(
        symbol=symbol,
        is_eligible=is_eligible,
        reasons=reasons,
        history_days=n,
        price=latest_close,
        ema_50=ema_50,
        ema_100=ema_100,
        sma_200=sma_200,
        price_above_ema100=above_ema100,
        price_above_sma200=above_sma200,
        golden_alignment=golden,
        dist_52w_high=dist_52w,
        adtv_20=adtv_20,
    )

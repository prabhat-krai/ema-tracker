"""
Volatility Estimation, Downside Risk & Smoothness Penalty Models.

Implements quantitative risk and volatility metrics:
    - Annualized Realized Volatility: sigma_ann = sigma_daily * sqrt(252)
    - Downside Semi-Deviation: sigma_down = sqrt(mean(min(0, r)^2)) * sqrt(252)
    - Jump Ratio & Gap Penalty: penalizes sudden single-day earnings gaps vs smooth compounding
    - 14-period Average True Range (ATR14) & Normalized ATR (NATR)
    - Volatility-adjusted Momentum (Information Momentum)

References:
    - Barroso, P. and Santa-Clara, P. (2015), 'Momentum Has Its Moments', Journal of Financial Economics.
    - Daniel, K. and Moskowitz, T. (2016), 'Momentum Crashes', Journal of Financial Economics.
    - Wilder, J. W. (1978), 'New Concepts in Technical Trading Systems'.
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def calculate_realized_volatility(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    lookback: int = 90,
    trading_days: int = 252,
) -> float:
    """
    Calculates annualized realized volatility from daily log returns over lookback window.

    Args:
        prices: 1D sequence of closing prices.
        lookback: Lookback window in trading days (default: 90).
        trading_days: Annualization factor (default: 252).

    Returns:
        Annualized volatility (float) or np.nan if insufficient data.
    """
    if prices is None:
        return np.nan

    arr = np.asarray(prices, dtype=np.float64).flatten()
    if len(arr) < 2:
        return np.nan

    window_len = min(len(arr), lookback + 1)
    if window_len < 2:
        return np.nan

    window = arr[-window_len:]
    if np.any(window <= 0.0) or np.any(np.isnan(window)):
        return np.nan

    log_returns = np.diff(np.log(window))
    if len(log_returns) < 1:
        return np.nan

    if len(log_returns) == 1:
        return 0.0

    daily_std = float(np.std(log_returns, ddof=1))
    if np.isnan(daily_std) or daily_std <= 1e-12:
        return 0.0

    return float(daily_std * np.sqrt(trading_days))


def calculate_downside_deviation(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    lookback: int = 90,
    trading_days: int = 252,
) -> float:
    """
    Calculates annualized downside semi-deviation (penalizes only negative daily log returns).

    Args:
        prices: 1D sequence of closing prices.
        lookback: Lookback window in trading days (default: 90).
        trading_days: Annualization factor (default: 252).

    Returns:
        Annualized downside volatility (float).
    """
    if prices is None:
        return np.nan

    arr = np.asarray(prices, dtype=np.float64).flatten()
    if len(arr) < 2:
        return np.nan

    window_len = min(len(arr), lookback + 1)
    window = arr[-window_len:]
    if np.any(window <= 0.0) or np.any(np.isnan(window)):
        return np.nan

    log_returns = np.diff(np.log(window))
    if len(log_returns) < 1:
        return np.nan

    neg_returns = np.minimum(0.0, log_returns)
    mean_sq_neg = np.mean(neg_returns ** 2)
    if np.isnan(mean_sq_neg) or mean_sq_neg <= 1e-16:
        return 0.0

    return float(np.sqrt(mean_sq_neg) * np.sqrt(trading_days))


def calculate_jump_ratio_penalty(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    lookback: int = 90,
    epsilon: float = 1e-6,
) -> Tuple[float, float]:
    """
    Calculates the maximum single-day jump ratio and smoothness penalty factor.

    Formula:
        MaxJump = max(|r_tau|)
        JumpRatio = MaxJump / max(sigma_daily, epsilon)
        JumpPenalty = exp(-max(0.0, (JumpRatio - 3.0) / 2.0)) in (0, 1]

    Returns:
        Tuple of (jump_ratio, jump_penalty)
    """
    if prices is None:
        return (np.nan, 1.0)

    arr = np.asarray(prices, dtype=np.float64).flatten()
    if len(arr) < 2:
        return (np.nan, 1.0)

    window_len = min(len(arr), lookback + 1)
    window = arr[-window_len:]
    if np.any(window <= 0.0) or np.any(np.isnan(window)):
        return (np.nan, 1.0)

    log_returns = np.diff(np.log(window))
    if len(log_returns) < 1:
        return (0.0, 1.0)

    daily_std = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0
    if daily_std <= epsilon:
        # Zero variance: perfectly uniform returns, no outlier jumps
        return (0.0, 1.0)

    max_jump = float(np.max(np.abs(log_returns)))
    jump_ratio = float(max_jump / daily_std)

    # Smoothness penalty factor
    excess_jump = max(0.0, (jump_ratio - 3.0) / 2.0)
    jump_penalty = float(np.exp(-excess_jump))
    jump_penalty = float(np.clip(jump_penalty, 0.0, 1.0))

    return (jump_ratio, jump_penalty)


def calculate_atr(
    high: Union[np.ndarray, pd.Series, Sequence[float]],
    low: Union[np.ndarray, pd.Series, Sequence[float]],
    close: Union[np.ndarray, pd.Series, Sequence[float]],
    period: int = 14,
) -> Tuple[float, float]:
    """
    Calculates 14-period Average True Range (ATR) and Normalized ATR (NATR %).

    Formula:
        TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        ATR_14 = Wilder's RMA / Exponential Moving Average of TR over 14 periods
        NATR_14 = (ATR_14 / Close_t) * 100%

    Returns:
        Tuple of (latest_atr, latest_natr_pct)
    """
    h = np.asarray(high, dtype=np.float64).flatten()
    l = np.asarray(low, dtype=np.float64).flatten()
    c = np.asarray(close, dtype=np.float64).flatten()

    n = len(c)
    if n < period or len(h) != n or len(l) != n:
        return (np.nan, np.nan)

    # Calculate True Range series
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )

    # Wilder's smoothing RMA (or EMA with alpha = 1 / period)
    alpha = 1.0 / period
    atr_series = np.zeros(n, dtype=np.float64)
    atr_series[period - 1] = np.mean(tr[:period])

    for i in range(period, n):
        atr_series[i] = alpha * tr[i] + (1.0 - alpha) * atr_series[i - 1]

    latest_atr = float(atr_series[-1])
    latest_close = float(c[-1])
    if latest_close > 0.0 and not np.isnan(latest_atr):
        latest_natr = float((latest_atr / latest_close) * 100.0)
    else:
        latest_natr = np.nan

    return (latest_atr, latest_natr)


def calculate_volatility_adjusted_momentum(
    momentum: float,
    realized_vol: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculates volatility-normalized momentum return (Information Momentum).
    Mom_vol_norm = Momentum / max(sigma_ann, epsilon)
    """
    if np.isnan(momentum) or np.isnan(realized_vol):
        return np.nan
    denom = max(realized_vol, epsilon)
    return float(momentum / denom)


def compute_volatility_metrics(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    high: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
    low: Optional[Union[np.ndarray, pd.Series, Sequence[float]]] = None,
    lookback: int = 90,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of volatility and risk metrics.
    """
    arr_c = np.asarray(prices, dtype=np.float64).flatten()
    vol_ann = calculate_realized_volatility(arr_c, lookback=lookback, trading_days=trading_days)
    vol_down = calculate_downside_deviation(arr_c, lookback=lookback, trading_days=trading_days)
    jump_ratio, jump_penalty = calculate_jump_ratio_penalty(arr_c, lookback=lookback)

    if high is not None and low is not None and len(high) == len(arr_c) and len(low) == len(arr_c):
        atr_14, natr_14 = calculate_atr(high, low, arr_c, period=14)
    else:
        atr_14, natr_14 = (np.nan, np.nan)

    return {
        "vol_ann": vol_ann,
        "volatility_ann": vol_ann,
        "vol_down": vol_down,
        "volatility_down": vol_down,
        "max_jump_ratio": jump_ratio,
        "jump_ratio": jump_ratio,
        "jump_penalty": jump_penalty,
        "atr_14": atr_14,
        "natr_14": natr_14,
    }

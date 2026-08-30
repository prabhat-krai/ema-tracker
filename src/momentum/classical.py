"""
Classical Jegadeesh & Titman (1993) Momentum & Multi-Timeframe Return Metrics.

Implements intermediate price momentum skipping the most recent 1 month (21 trading days)
to eliminate short-term microstructure noise and bid-ask bounce:
    Mom_12_1 = (P_{t-21} - P_{t-252}) / P_{t-252}
    Mom_6_1  = (P_{t-21} - P_{t-126}) / P_{t-126}
    Mom_3_1  = (P_{t-21} - P_{t-63})  / P_{t-63}

Reference:
    Jegadeesh, N. and Titman, S. (1993), 'Returns to Buying Winners and Selling Losers:
    Implications for Stock Market Efficiency', The Journal of Finance, 48: 65-91.
"""

from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd


def compute_jegadeesh_titman_momentum(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    lookback_total: int = 252,
    skip_recent: int = 21,
    log_return: bool = False,
) -> float:
    """
    Computes Jegadeesh & Titman 12-1 intermediate momentum return skipping the most recent 21 days.

    Args:
        prices: 1D array of historical closing prices.
        lookback_total: Total lookback window in trading days (default: 252 for 12 months).
        skip_recent: Number of most recent trading days to skip (default: 21 for 1 month).
        log_return: If True, computes ln(P_{t-21}) - ln(P_{t-252}); otherwise simple return.

    Returns:
        float momentum return or np.nan if data is insufficient or invalid.
    """
    if prices is None:
        return np.nan

    if isinstance(prices, pd.Series):
        arr = prices.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(prices, dtype=np.float64)

    if arr.ndim != 1:
        arr = arr.flatten()

    n = len(arr)
    if n < lookback_total or lookback_total <= skip_recent:
        return np.nan

    # P_{t-21} is the price 21 trading days before the latest price (index -1)
    idx_skip = -skip_recent - 1
    if abs(idx_skip) > n:
        return np.nan
    p_t_skip = arr[idx_skip]

    # P_{t-252} is the price at lookback_total bars before the latest price
    if n >= lookback_total + 1:
        p_t_start = arr[-(lookback_total + 1)]
    else:
        p_t_start = arr[0]

    if np.isnan(p_t_skip) or np.isnan(p_t_start) or p_t_start <= 0.0 or p_t_skip <= 0.0:
        return np.nan

    if log_return:
        return float(np.log(p_t_skip) - np.log(p_t_start))
    else:
        return float((p_t_skip - p_t_start) / p_t_start)


# Standard alias
calculate_jegadeesh_titman_momentum = compute_jegadeesh_titman_momentum


def calculate_intermediate_momentum(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    lookback: int = 126,
    skip_recent: int = 21,
    log_return: bool = False,
) -> float:
    """
    Computes intermediate skip momentum (e.g. 6-1 momentum with lookback=126, skip=21).
    """
    return compute_jegadeesh_titman_momentum(
        prices=prices,
        lookback_total=lookback,
        skip_recent=skip_recent,
        log_return=log_return,
    )


def calculate_multi_timeframe_returns(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    skip_recent: int = 21,
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of classical intermediate skip momentum
    and raw unskipped multi-timeframe returns.

    Returns:
        Dict with keys:
            'mom_12_1': 12-month return skipping last 21d (lookback=252, skip=21)
            'mom_6_1': 6-month return skipping last 21d (lookback=126, skip=21)
            'mom_3_1': 3-month return skipping last 21d (lookback=63, skip=21)
            'ret_1m': Raw 1-month return (21d to latest)
            'ret_3m': Raw 3-month return (63d to latest)
            'ret_6m': Raw 6-month return (126d to latest)
            'ret_12m': Raw 12-month return (252d to latest)
    """
    if prices is None:
        return {
            "mom_12_1": np.nan, "mom_6_1": np.nan, "mom_3_1": np.nan,
            "ret_1m": np.nan, "ret_3m": np.nan, "ret_6m": np.nan, "ret_12m": np.nan,
        }

    arr = np.asarray(prices, dtype=np.float64).flatten()
    n = len(arr)

    def _raw_return(window_bars: int) -> float:
        if n < window_bars + 1 or window_bars < 1:
            if n == window_bars and n > 0:
                p_start = arr[0]
            else:
                return np.nan
        else:
            p_start = arr[-(window_bars + 1)]
        p_end = arr[-1]
        if np.isnan(p_start) or np.isnan(p_end) or p_start <= 0.0 or p_end <= 0.0:
            return np.nan
        return float((p_end - p_start) / p_start)

    return {
        "mom_12_1": compute_jegadeesh_titman_momentum(arr, lookback_total=252, skip_recent=skip_recent),
        "mom_6_1": compute_jegadeesh_titman_momentum(arr, lookback_total=126, skip_recent=skip_recent),
        "mom_3_1": compute_jegadeesh_titman_momentum(arr, lookback_total=63, skip_recent=skip_recent),
        "ret_1m": _raw_return(21),
        "ret_3m": _raw_return(63),
        "ret_6m": _raw_return(126),
        "ret_12m": _raw_return(252),
    }

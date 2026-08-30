"""
Andreas Clenow Exponential Regression Momentum Model.

Implements closed-form Ordinary Least Squares (OLS) regression on log prices:
    ln(P_t) = alpha + beta * t
    Annualized Slope = exp(beta * 250) - 1.0
    R^2 = Pearson correlation squared (coefficient of determination)
    Clenow Score = Annualized Slope * R^2

Reference:
    Andreas Clenow (2015), 'Stocks on the Move: Beating the Market with Hedge Fund Momentum Strategies'
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def compute_clenow_momentum(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    lookback: Optional[int] = 90,
    trading_days: int = 250,
) -> Tuple[float, float, float]:
    """
    Computes the Andreas Clenow Exponential Regression Slope, R^2, and Momentum Score.

    Args:
        prices: 1D array-like sequence of closing prices.
        lookback: Number of trading days for the regression window. If None, uses all prices.
        trading_days: Number of trading days per year for annualization (default 250).

    Returns:
        Tuple of (annualized_slope, r_squared, clenow_score)
        Returns (np.nan, np.nan, np.nan) if data is insufficient or invalid.
    """
    if prices is None:
        return (np.nan, np.nan, np.nan)

    if isinstance(prices, pd.Series):
        arr = prices.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(prices, dtype=np.float64)

    # Remove any leading/trailing NaNs if 1D
    if arr.ndim != 1:
        arr = arr.flatten()

    n_total = len(arr)
    if n_total == 0:
        return (np.nan, np.nan, np.nan)

    if lookback is None:
        window_len = n_total
    else:
        window_len = int(lookback)

    # Minimum required bars for statistical validity is 2 (ideally >= 30)
    if window_len < 2 or n_total < window_len:
        return (np.nan, np.nan, np.nan)

    window = arr[-window_len:]

    # Check for invalid values: non-positive, NaNs, or Infs
    if np.any(window <= 0.0) or np.any(np.isnan(window)) or np.any(np.isinf(window)):
        return (np.nan, np.nan, np.nan)

    # 1. Log transformation
    y = np.log(window)
    x = np.arange(window_len, dtype=np.float64)

    # 2. Closed-form OLS calculation
    x_mean = (window_len - 1.0) / 2.0
    y_mean = np.mean(y)

    x_dev = x - x_mean
    y_dev = y - y_mean

    var_x = (window_len * (window_len * window_len - 1.0)) / 12.0
    cov_xy = np.sum(x_dev * y_dev)

    if var_x <= 0.0:
        return (0.0, 0.0, 0.0)

    beta = cov_xy / var_x

    # 3. Annualized exponential slope
    # Formula: exp(beta * trading_days) - 1.0
    # Guard against numerical overflow for extreme beta
    scaled_beta = beta * trading_days
    if scaled_beta > 700.0:
        annualized_slope = float(np.inf)
    elif scaled_beta < -700.0:
        annualized_slope = -1.0
    else:
        annualized_slope = np.exp(scaled_beta) - 1.0

    # 4. Coefficient of determination R^2
    ss_tot = np.sum(y_dev ** 2)
    if ss_tot <= 1e-12:
        # Flat series: zero variance in log price
        return (0.0, 0.0, 0.0)

    r_squared = (cov_xy ** 2) / (var_x * ss_tot)
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    # 5. Clenow Momentum Score
    clenow_score = float(annualized_slope * r_squared)

    return (float(annualized_slope), float(r_squared), float(clenow_score))


# Standard alias for compatibility with interface contracts
calculate_clenow_momentum = compute_clenow_momentum


def calculate_clenow_regression(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
) -> Tuple[float, float, float]:
    """
    Computes closed-form OLS regression on log prices: ln(P_t) = alpha + beta * t.

    Args:
        prices: 1D array-like sequence of closing prices.

    Returns:
        Tuple of (alpha, beta, r_squared).
        Returns (0.0, 0.0, 0.0) if invalid or flat.
    """
    arr = np.asarray(prices, dtype=np.float64).flatten()
    n = len(arr)
    if n < 2 or np.any(arr <= 0.0) or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return (0.0, 0.0, 0.0)

    y = np.log(arr)
    x = np.arange(n, dtype=np.float64)

    x_mean = (n - 1.0) / 2.0
    y_mean = np.mean(y)

    x_dev = x - x_mean
    y_dev = y - y_mean

    var_x = (n * (n * n - 1.0)) / 12.0
    cov_xy = np.sum(x_dev * y_dev)

    if var_x <= 0.0:
        return (float(y_mean), 0.0, 0.0)

    beta = float(cov_xy / var_x)
    alpha = float(y_mean - beta * x_mean)

    ss_tot = np.sum(y_dev ** 2)
    if ss_tot <= 1e-12:
        return (alpha, 0.0, 0.0)

    r2 = float(np.clip((cov_xy ** 2) / (var_x * ss_tot), 0.0, 1.0))
    return (alpha, beta, r2)


def calculate_multi_timeframe_clenow(
    prices: Union[np.ndarray, pd.Series, Sequence[float]],
    windows: Tuple[int, int, int] = (90, 126, 252),
    weights: Tuple[float, float, float] = (0.40, 0.30, 0.30),
    trading_days: int = 250,
) -> Dict[str, float]:
    """
    Calculates multi-timeframe Clenow scores across short (90d), medium (126d), and long (252d) windows.
    Blends valid windows using normalized weights.

    Args:
        prices: 1D array of closing prices.
        windows: Tuple of lookback windows (default: 90, 126, 252).
        weights: Tuple of weights corresponding to windows (default: 0.40, 0.30, 0.30).
        trading_days: Number of trading days per year.

    Returns:
        Dictionary containing individual window metrics and composite score.
    """
    if prices is None:
        return {
            "slope_90": np.nan, "r2_90": np.nan, "score_90": np.nan,
            "slope_126": np.nan, "r2_126": np.nan, "score_126": np.nan,
            "slope_252": np.nan, "r2_252": np.nan, "score_252": np.nan,
            "composite_score": np.nan,
            "slope_ann": np.nan, "r_squared": np.nan, "clenow_score": np.nan,
        }

    arr = np.asarray(prices, dtype=np.float64).flatten()
    n_total = len(arr)

    results: Dict[str, float] = {}
    valid_scores = []
    valid_weights = []

    for w, wt in zip(windows, weights):
        if n_total >= w:
            slope, r2, score = compute_clenow_momentum(arr, lookback=w, trading_days=trading_days)
            results[f"slope_{w}"] = slope
            results[f"r2_{w}"] = r2
            results[f"score_{w}"] = score
            if not np.isnan(score):
                valid_scores.append(score)
                valid_weights.append(wt)
        else:
            results[f"slope_{w}"] = np.nan
            results[f"r2_{w}"] = np.nan
            results[f"score_{w}"] = np.nan

    # Calculate composite score across valid windows
    if len(valid_scores) > 0 and sum(valid_weights) > 0:
        norm_weights = np.array(valid_weights) / sum(valid_weights)
        composite = float(np.sum(np.array(valid_scores) * norm_weights))
    else:
        composite = np.nan

    results["composite_score"] = composite

    # Set primary default 90d (or first valid) as main clenow_score
    primary_w = windows[0]
    results["slope_ann"] = results.get(f"slope_{primary_w}", np.nan)
    results["r_squared"] = results.get(f"r2_{primary_w}", np.nan)
    results["clenow_score"] = results.get(f"score_{primary_w}", np.nan)

    return results

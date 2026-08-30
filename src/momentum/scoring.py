"""
Cross-Sectional Standardization, Winsorized Z-Scoring & Composite Momentum Ranking.

Implements cross-sectional factor normalization and multi-factor ranking:
    - Winsorized Z-Score: Z = clip((F - mean) / std, -3.0, 3.0)
    - Percentile Ranking: RankPct in [0.0, 1.0]
    - Multi-Factor Composite Score:
        Score = w_clenow * Z(Clenow) + w_12_1 * Z(Mom12_1) + w_r2 * Z(R^2)
                + w_6_1 * Z(Mom6_1) - w_vol * Z(RealizedVol)
    - Ranking & Selection Pipeline
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.momentum.models import MomentumMetrics, MomentumScoreBreakdown, RebalanceAction


DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "clenow_score": 0.40,
    "mom_12_1": 0.25,
    "r_squared": 0.15,
    "mom_6_1": 0.10,
    "vol_ann": -0.10,  # Negative weight penalizes higher volatility
}


def winsorized_z_score(
    values: Union[np.ndarray, pd.Series, Sequence[float]],
    clip_range: Tuple[float, float] = (-3.0, 3.0),
    ddof: int = 1,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Computes cross-sectional Winsorized Z-scores clipped to [clip_min, clip_max].

    Args:
        values: 1D array of raw factor values across a stock universe.
        clip_range: Lower and upper bounds for outlier winsorization (default: (-3.0, 3.0)).
        ddof: Degrees of freedom for standard deviation (default: 1).
        epsilon: Minimum standard deviation threshold to prevent division by zero.

    Returns:
        1D array of standardized Z-scores with NaNs filled as 0.0 (neutral).
    """
    if values is None:
        return np.array([], dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64).flatten()
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)

    z_scores = np.zeros(n, dtype=np.float64)
    valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
    n_valid = int(np.sum(valid_mask))

    if n_valid <= 1:
        return z_scores

    valid_vals = arr[valid_mask]
    mean_val = float(np.mean(valid_vals))
    std_val = float(np.std(valid_vals, ddof=ddof)) if n_valid > 1 else 0.0

    if std_val < epsilon:
        # Zero variance: all valid elements are identical -> assign 0.0
        return z_scores

    z_raw = (valid_vals - mean_val) / std_val
    z_clipped = np.clip(z_raw, clip_range[0], clip_range[1])

    z_scores[valid_mask] = z_clipped
    return z_scores


def calculate_percentile_rank(
    values: Union[np.ndarray, pd.Series, Sequence[float]],
    ascending: bool = True,
) -> np.ndarray:
    """
    Calculates percentile ranks in [0.0, 1.0] across an array of values.

    Args:
        values: 1D array of factor scores.
        ascending: If True, highest value receives rank 1.0; if False, lowest receives 1.0.

    Returns:
        1D array of percentile ranks.
    """
    if values is None:
        return np.array([], dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64).flatten()
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
    n_valid = int(np.sum(valid_mask))

    pct_ranks = np.zeros(n, dtype=np.float64)
    if n_valid == 0:
        return pct_ranks
    if n_valid == 1:
        pct_ranks[valid_mask] = 1.0
        return pct_ranks

    valid_vals = arr[valid_mask]
    if not ascending:
        valid_vals = -valid_vals

    # scipy rankdata with 'average' method: 1 to n_valid
    ranks = stats.rankdata(valid_vals, method="average")
    pcts = (ranks - 1.0) / float(n_valid - 1.0)

    pct_ranks[valid_mask] = pcts
    return pct_ranks


def compute_composite_momentum_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Computes Winsorized Z-scores and weighted composite momentum score for a DataFrame.

    Args:
        df: DataFrame containing factor columns:
            ['symbol', 'clenow_score', 'mom_12_1', 'r_squared', 'mom_6_1', 'vol_ann']
        weights: Dictionary mapping factor column names to weights.

    Returns:
        DataFrame with added Z-score columns, 'composite_score', 'percentile_rank', and 'rank'.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()
    if weights is None:
        weights = DEFAULT_FACTOR_WEIGHTS.copy()

    # Calculate standardized Z-scores for each factor
    composite_scores = np.zeros(len(res_df), dtype=np.float64)

    # Normalize weights absolute sum if desired, or use explicit weights
    for factor, weight in weights.items():
        # Match column names case-insensitively or with variations
        matched_col = None
        for col in res_df.columns:
            if col.lower() == factor.lower() or (factor == "r_squared" and col.lower() in ("r2", "clenow_r2")):
                matched_col = col
                break

        if matched_col is not None:
            raw_vals = res_df[matched_col].to_numpy(dtype=np.float64)
            z = winsorized_z_score(raw_vals)
            z_col_name = f"z_{factor}"
            res_df[z_col_name] = z
            composite_scores += weight * z
        else:
            res_df[f"z_{factor}"] = 0.0

    res_df["composite_score"] = composite_scores
    res_df["percentile_rank"] = calculate_percentile_rank(composite_scores, ascending=True)

    # Sort descending by composite score
    res_df = res_df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    res_df["rank"] = res_df.index + 1

    return res_df


def rank_universe(
    metrics_list: Sequence[MomentumMetrics],
    weights: Optional[Dict[str, float]] = None,
    filter_trend: bool = True,
) -> Tuple[List[MomentumScoreBreakdown], pd.DataFrame]:
    """
    Filters and ranks a universe of stocks from a list of MomentumMetrics dataclasses.

    Args:
        metrics_list: List of MomentumMetrics objects.
        weights: Factor weights dictionary (optional).
        filter_trend: If True, only scores stocks where passes_trend is True.

    Returns:
        Tuple of (List[MomentumScoreBreakdown], ranked_pd_DataFrame)
    """
    if not metrics_list:
        return [], pd.DataFrame()

    records = []
    for m in metrics_list:
        records.append({
            "symbol": m.symbol,
            "close": m.close,
            "clenow_score": m.clenow_score,
            "clenow_slope_ann": m.clenow_slope_ann,
            "r_squared": m.clenow_r2,
            "mom_12_1": m.mom_12_1,
            "mom_6_1": m.mom_6_1,
            "vol_ann": m.volatility_ann,
            "natr_14": m.natr_14,
            "passes_trend": m.passes_trend,
        })

    df = pd.DataFrame(records)

    if filter_trend and "passes_trend" in df.columns:
        eligible_df = df[df["passes_trend"] == True].copy()
    else:
        eligible_df = df.copy()

    if eligible_df.empty:
        return [], pd.DataFrame()

    scored_df = compute_composite_momentum_score(eligible_df, weights=weights)

    breakdowns: List[MomentumScoreBreakdown] = []
    for _, row in scored_df.iterrows():
        b = MomentumScoreBreakdown(
            symbol=str(row["symbol"]),
            rank=int(row["rank"]),
            composite_score=float(row["composite_score"]),
            percentile_rank=float(row["percentile_rank"]),
            z_clenow=float(row.get("z_clenow_score", 0.0)),
            z_mom_12_1=float(row.get("z_mom_12_1", 0.0)),
            z_r2=float(row.get("z_r_squared", 0.0)),
            z_mom_6_1=float(row.get("z_mom_6_1", 0.0)),
            z_vol_ann=float(row.get("z_vol_ann", 0.0)),
            raw_clenow_score=float(row["clenow_score"]),
            raw_slope_ann=float(row["clenow_slope_ann"]),
            raw_r2=float(row["r_squared"]),
            raw_mom_12_1=float(row["mom_12_1"]),
            raw_mom_6_1=float(row["mom_6_1"]),
            raw_vol_ann=float(row["vol_ann"]),
            close=float(row["close"]),
            natr_14=float(row["natr_14"]),
            action=RebalanceAction.BUY if int(row["rank"]) <= 10 else RebalanceAction.HOLD,
        )
        breakdowns.append(b)

    return breakdowns, scored_df

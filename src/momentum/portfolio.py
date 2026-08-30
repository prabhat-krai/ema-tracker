"""
Quantitative Momentum Portfolio Selection & Weighting Engine.

Implements Top N selection, weighting algorithms (Equal Weight, Inverse-Volatility,
Bounded Risk-Parity), macro market regime cash scaling, and portfolio risk metrics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from src.momentum.models import (
    MarketRegime,
    MarketRegimeState,
    MomentumMetrics,
    MomentumScoreBreakdown,
    PortfolioConstituent,
    PortfolioRecommendation,
    RebalanceAction,
)


def calculate_equal_weights(n: int) -> np.ndarray:
    """
    Computes equal weighting across N assets: w_i = 1 / N.

    Args:
        n: Number of portfolio constituents.

    Returns:
        1D array of equal weights summing to 1.0.
    """
    if n <= 0:
        return np.array([], dtype=np.float64)
    return np.full(n, 1.0 / float(n), dtype=np.float64)


def calculate_inverse_volatility_weights(
    volatilities: Union[np.ndarray, Sequence[float]],
    min_vol: float = 0.05,
) -> np.ndarray:
    """
    Computes inverse-volatility weights:
        w_i = (1 / max(sigma_i, min_vol)) / sum(1 / max(sigma_j, min_vol))

    Lower volatility assets receive higher capital allocation to reduce portfolio drawdown.

    Args:
        volatilities: 1D array of annualized realized volatilities.
        min_vol: Minimum volatility floor to prevent division by zero or extreme sizing.

    Returns:
        1D array of inverse-volatility weights summing to 1.0.
    """
    arr = np.asarray(volatilities, dtype=np.float64).flatten()
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    floored_vols = np.maximum(arr, min_vol)
    inv_vols = 1.0 / floored_vols
    total_inv = np.sum(inv_vols)

    if total_inv <= 0 or np.isnan(total_inv):
        return np.full(n, 1.0 / float(n), dtype=np.float64)

    return inv_vols / total_inv


def calculate_bounded_risk_parity_weights(
    volatilities: Union[np.ndarray, Sequence[float]],
    w_min: float = 0.05,
    w_max: float = 0.20,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Computes bounded risk-parity weights where w_i in [w_min, w_max] and sum(w_i) == 1.0.

    Iteratively projects inverse-volatility weights into bounded space while reallocating
    excess weight across unclipped constituents.

    Args:
        volatilities: 1D array of annualized volatilities.
        w_min: Minimum weight per constituent (default: 0.05 / 5%).
        w_max: Maximum weight per constituent (default: 0.20 / 20%).
        max_iter: Maximum projection iterations.

    Returns:
        1D array of weights satisfying bounds and summing to 1.0.
    """
    arr = np.asarray(volatilities, dtype=np.float64).flatten()
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    # Infeasible bound check: fallback to equal weight
    if n * w_min > 1.0 + 1e-7 or n * w_max < 1.0 - 1e-7:
        return np.full(n, 1.0 / float(n), dtype=np.float64)

    raw_inv_vol = 1.0 / np.maximum(arr, 1e-4)
    weights = raw_inv_vol / np.sum(raw_inv_vol)

    for _ in range(max_iter):
        clipped = np.clip(weights, w_min, w_max)
        excess = 1.0 - float(np.sum(clipped))

        if abs(excess) < 1e-8:
            return clipped

        if excess > 0:
            # Need to distribute remaining positive excess to assets that haven't hit w_max
            eligible = clipped < (w_max - 1e-8)
            if not np.any(eligible):
                return clipped
            weights = clipped.copy()
            eligible_sum = np.sum(weights[eligible])
            if eligible_sum > 0:
                weights[eligible] += excess * (weights[eligible] / eligible_sum)
            else:
                weights[eligible] += excess / np.sum(eligible)
        else:
            # Need to absorb negative excess from assets that haven't hit w_min
            eligible = clipped > (w_min + 1e-8)
            if not np.any(eligible):
                return clipped
            weights = clipped.copy()
            eligible_sum = np.sum(weights[eligible])
            if eligible_sum > 0:
                weights[eligible] += excess * (weights[eligible] / eligible_sum)
            else:
                weights[eligible] += excess / np.sum(eligible)

    return np.clip(weights, w_min, w_max)


def calculate_portfolio_risk_metrics(
    constituents: Sequence[PortfolioConstituent],
) -> Dict[str, float]:
    """
    Computes portfolio-level risk and diversification metrics.

    Args:
        constituents: List of PortfolioConstituent dataclasses.

    Returns:
        Dictionary of portfolio metrics:
            - weighted_clenow_score
            - weighted_volatility
            - portfolio_volatility
            - diversification_ratio
            - total_equity_pct
            - expected_cash_pct
    """
    if not constituents:
        return {
            "weighted_clenow_score": 0.0,
            "weighted_volatility": 0.0,
            "portfolio_volatility": 0.0,
            "diversification_ratio": 1.0,
            "total_equity_pct": 0.0,
            "expected_cash_pct": 1.0,
        }

    weights = np.array([c.weight for c in constituents], dtype=np.float64)
    scores = np.array([c.clenow_score for c in constituents], dtype=np.float64)
    vols = np.array([c.volatility_ann for c in constituents], dtype=np.float64)

    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return {
            "weighted_clenow_score": 0.0,
            "weighted_volatility": 0.0,
            "portfolio_volatility": 0.0,
            "diversification_ratio": 1.0,
            "total_equity_pct": 0.0,
            "expected_cash_pct": 1.0,
        }

    # Normalized weights across equity portion
    norm_w = weights / total_weight

    weighted_clenow = float(np.sum(norm_w * scores))
    weighted_vol = float(np.sum(norm_w * vols))

    # Portfolio volatility (conservative unhedged independent risk approximation)
    portfolio_vol = float(np.sqrt(np.sum((norm_w * vols) ** 2)))

    # Diversification ratio (effective N / inverse Herfindahl index)
    herfindahl = float(np.sum(norm_w ** 2))
    diversification_ratio = float(1.0 / herfindahl) if herfindahl > 0 else 1.0

    expected_cash = float(max(0.0, 1.0 - total_weight))

    return {
        "weighted_clenow_score": weighted_clenow,
        "weighted_volatility": weighted_vol,
        "portfolio_volatility": portfolio_vol,
        "diversification_ratio": diversification_ratio,
        "total_equity_pct": total_weight,
        "expected_cash_pct": expected_cash,
    }


def build_momentum_portfolio(
    ranked_constituents: Sequence[Union[MomentumScoreBreakdown, Dict[str, Any]]],
    metrics_map: Optional[Dict[str, MomentumMetrics]] = None,
    top_n: int = 10,
    weighting_scheme: str = "inv_vol",  # 'equal', 'inv_vol', 'bounded_parity'
    current_holdings: Optional[List[str]] = None,
    rank_buffer: int = 20,
    max_weight: float = 0.20,
    min_weight: float = 0.05,
    market_regime: Optional[MarketRegime] = None,
    total_capital: float = 1_000_000.0,
    universe: str = "india",
    as_of_date: Optional[str] = None,
) -> PortfolioRecommendation:
    """
    Builds a complete Top N momentum portfolio recommendation with exact weights,
    rebalancing actions, share quantities, cash scaling, and risk metrics.

    Args:
        ranked_constituents: Ordered list of MomentumScoreBreakdown objects (or dicts).
        metrics_map: Optional map of symbol -> MomentumMetrics for extended data.
        top_n: Target portfolio size (default: 10).
        weighting_scheme: Weighting method ('equal', 'inv_vol', 'bounded_parity').
        current_holdings: List of currently held stock symbols (for BUY/HOLD/SELL tagging).
        rank_buffer: Rank hysteresis buffer threshold (default: 20).
        max_weight: Maximum single-stock allocation cap (default: 0.20).
        min_weight: Minimum single-stock allocation floor (default: 0.05).
        market_regime: Optional macro market regime evaluation for cash scaling.
        total_capital: Total portfolio capital value (default: 1,000,000).
        universe: Target market universe ("india" or "usa").
        as_of_date: Evaluation date string (defaults to today).

    Returns:
        PortfolioRecommendation dataclass with constituents, weights, and metrics.
    """
    if current_holdings is None:
        current_holdings = []
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")

    # If market regime not provided, default to BULLISH (100% equity)
    if market_regime is None:
        market_regime = MarketRegime(
            benchmark_symbol="^NSEI" if universe == "india" else "^GSPC",
            state=MarketRegimeState.BULLISH,
            benchmark_price=0.0,
            sma_200=0.0,
            sma_200_slope=0.0,
            equity_allocation_pct=1.0,
            evaluation_date=as_of_date,
        )

    equity_multiplier = float(market_regime.equity_allocation_pct)

    # Filter top candidates with hysteresis rank buffer if current holdings provided
    if current_holdings:
        current_set = set(current_holdings)
        retained = []
        new_candidates = []
        for item in ranked_constituents:
            if isinstance(item, MomentumScoreBreakdown):
                sym = item.symbol
                rank = item.rank
            elif isinstance(item, dict):
                sym = str(item.get("symbol", ""))
                rank = int(item.get("rank", 999))
            else:
                sym = str(getattr(item, "symbol", ""))
                rank = int(getattr(item, "rank", 999))

            # Check trend filter
            passes_trend = True
            if metrics_map and sym in metrics_map:
                passes_trend = metrics_map[sym].passes_trend
            elif hasattr(item, "passes_trend"):
                passes_trend = bool(getattr(item, "passes_trend"))
            elif isinstance(item, dict) and "passes_trend" in item:
                passes_trend = bool(item["passes_trend"])

            if sym in current_set:
                if rank <= rank_buffer and passes_trend:
                    retained.append(item)
            else:
                if passes_trend:
                    new_candidates.append(item)

        retained = retained[:top_n]
        slots_needed = max(0, top_n - len(retained))
        selected_new = new_candidates[:slots_needed]
        top_candidates = retained + selected_new
        top_candidates.sort(
            key=lambda x: x.rank if isinstance(x, MomentumScoreBreakdown) else (
                x.get("rank", 999) if isinstance(x, dict) else getattr(x, "rank", 999)
            )
        )
    else:
        top_candidates = list(ranked_constituents[:top_n])

    k = len(top_candidates)

    if k == 0:
        rec = PortfolioRecommendation(
            universe=universe,
            as_of_date=as_of_date,
            market_regime=market_regime,
            weighting_scheme=weighting_scheme,
            target_constituents=[],
            total_capital=total_capital,
            total_equity_pct=0.0,
            expected_cash_pct=1.0,
            allocated_cash_pct=1.0,
            weighted_clenow_score=0.0,
            weighted_volatility=0.0,
            portfolio_volatility=0.0,
            diversification_ratio=1.0,
            summary={"reason": "No eligible momentum constituents found"},
        )
        return rec

    # Extract volatilities
    vols = []
    for item in top_candidates:
        if isinstance(item, MomentumScoreBreakdown):
            vols.append(item.raw_vol_ann)
        elif isinstance(item, dict):
            vols.append(item.get("raw_vol_ann", item.get("vol_ann", 0.20)))
        else:
            vols.append(getattr(item, "raw_vol_ann", getattr(item, "vol_ann", 0.20)))
    vol_arr = np.array(vols, dtype=np.float64)

    # Compute unscaled weights based on scheme
    if weighting_scheme == "equal":
        raw_weights = calculate_equal_weights(k)
    elif weighting_scheme == "bounded_parity":
        raw_weights = calculate_bounded_risk_parity_weights(
            vol_arr, w_min=min_weight, w_max=max_weight
        )
    elif weighting_scheme == "inv_vol":
        raw_weights = calculate_inverse_volatility_weights(vol_arr)
    else:
        raise ValueError(
            f"Unknown weighting scheme: '{weighting_scheme}'. "
            f"Must be 'equal', 'inv_vol', or 'bounded_parity'."
        )

    # Scale weights by market regime equity allocation and partial qualification ratio
    partial_scale = float(k) / float(top_n) if (top_n > 0 and k < top_n) else 1.0
    final_weights = raw_weights * equity_multiplier * partial_scale

    constituents: List[PortfolioConstituent] = []
    for i, item in enumerate(top_candidates):
        w = float(final_weights[i])

        if isinstance(item, MomentumScoreBreakdown):
            sym = item.symbol
            rank = item.rank
            comp_score = item.composite_score
            clenow_score = item.raw_clenow_score
            slope = item.raw_slope_ann
            r2 = item.raw_r2
            m12_1 = item.raw_mom_12_1
            m6_1 = item.raw_mom_6_1
            vol = item.raw_vol_ann
            natr = item.natr_14
            close = item.close
        elif isinstance(item, dict):
            sym = str(item.get("symbol", f"SYM_{i+1}"))
            rank = int(item.get("rank", i + 1))
            comp_score = float(item.get("composite_score", 0.0))
            clenow_score = float(item.get("raw_clenow_score", item.get("clenow_score", 0.0)))
            slope = float(item.get("raw_slope_ann", item.get("clenow_slope_ann", 0.0)))
            r2 = float(item.get("raw_r2", item.get("r_squared", 0.0)))
            m12_1 = float(item.get("raw_mom_12_1", item.get("mom_12_1", 0.0)))
            m6_1 = float(item.get("raw_mom_6_1", item.get("mom_6_1", 0.0)))
            vol = float(item.get("raw_vol_ann", item.get("vol_ann", 0.20)))
            natr = float(item.get("natr_14", 0.0))
            close = float(item.get("close", 0.0))
        else:
            sym = getattr(item, "symbol", f"SYM_{i+1}")
            rank = getattr(item, "rank", i + 1)
            comp_score = getattr(item, "composite_score", 0.0)
            clenow_score = getattr(item, "raw_clenow_score", getattr(item, "clenow_score", 0.0))
            slope = getattr(item, "raw_slope_ann", getattr(item, "clenow_slope_ann", 0.0))
            r2 = getattr(item, "raw_r2", getattr(item, "r_squared", 0.0))
            m12_1 = getattr(item, "raw_mom_12_1", getattr(item, "mom_12_1", 0.0))
            m6_1 = getattr(item, "raw_mom_6_1", getattr(item, "mom_6_1", 0.0))
            vol = getattr(item, "raw_vol_ann", getattr(item, "vol_ann", 0.20))
            natr = getattr(item, "natr_14", 0.0)
            close = getattr(item, "close", 0.0)

        # Action assignment
        is_held = sym in current_holdings
        if not is_held:
            action = RebalanceAction.BUY if rank <= top_n else RebalanceAction.HOLD
        elif is_held and rank <= rank_buffer:
            action = RebalanceAction.HOLD
        elif is_held and rank > rank_buffer:
            action = RebalanceAction.SELL
        else:
            action = RebalanceAction.HOLD

        target_value = float(total_capital * w)
        target_shares = int(target_value // close) if close > 0 else 0

        c = PortfolioConstituent(
            symbol=sym,
            rank=rank,
            weight=w,
            composite_score=comp_score,
            clenow_score=clenow_score,
            clenow_slope_ann=slope,
            clenow_r2=r2,
            mom_12_1=m12_1,
            mom_6_1=m6_1,
            volatility_ann=vol,
            natr_14=natr,
            close=close,
            action=action,
            target_value=target_value,
            target_shares=target_shares,
        )
        constituents.append(c)

    risk_metrics = calculate_portfolio_risk_metrics(constituents)

    return PortfolioRecommendation(
        universe=universe,
        as_of_date=as_of_date,
        market_regime=market_regime,
        weighting_scheme=weighting_scheme,
        target_constituents=constituents,
        total_capital=total_capital,
        total_equity_pct=risk_metrics["total_equity_pct"],
        expected_cash_pct=risk_metrics["expected_cash_pct"],
        allocated_cash_pct=risk_metrics["expected_cash_pct"],
        weighted_clenow_score=risk_metrics["weighted_clenow_score"],
        weighted_volatility=risk_metrics["weighted_volatility"],
        portfolio_volatility=risk_metrics["portfolio_volatility"],
        diversification_ratio=risk_metrics["diversification_ratio"],
        summary={
            "top_n": top_n,
            "constituents_count": len(constituents),
            "market_regime": market_regime.state.value if hasattr(market_regime.state, "value") else str(market_regime.state),
            "equity_allocation_pct": market_regime.equity_allocation_pct,
        },
    )

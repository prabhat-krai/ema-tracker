"""
Quantitative Momentum Portfolio Rebalancing & Hysteresis Engine.

Implements rank buffer hysteresis (churn reduction), macro trend breakdown exits,
weight deadband filtering, trade share calculations, and RebalancePlan generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from src.momentum.models import (
    PortfolioConstituent,
    PortfolioRecommendation,
    RebalanceAction,
    RebalanceInstruction,
    RebalancePlan,
)


def classify_rebalance_action(
    symbol: str,
    rank: int,
    is_held: bool,
    passes_trend: bool = True,
    top_n: int = 10,
    rank_buffer: int = 20,
) -> str:
    """
    Classifies portfolio candidate action based on hysteresis rank buffer and trend state.

    Rules:
    1. New Entrant (not held):
       - If rank <= top_n AND passes_trend -> "NEW_BUY"
       - Else -> "IGNORE"
    2. Existing Holding (is_held):
       - If NOT passes_trend -> "SELL_EXIT" (immediate trend breakdown exit)
       - If rank > rank_buffer -> "SELL_EXIT" (rank deterioration exit)
       - If rank <= rank_buffer AND passes_trend -> "HOLD_RETAIN"

    Args:
        symbol: Stock ticker symbol.
        rank: Current cross-sectional rank (1 = top).
        is_held: Whether the stock is currently held in the portfolio.
        passes_trend: Whether the stock satisfies macro trend filters (Close > EMA100).
        top_n: Portfolio entry cutoff rank (default: 10).
        rank_buffer: Retention buffer rank cutoff (default: 20).

    Returns:
        Action string: "NEW_BUY", "HOLD_RETAIN", "SELL_EXIT", or "IGNORE".
    """
    if not is_held:
        if rank <= top_n and passes_trend:
            return "NEW_BUY"
        return "IGNORE"
    else:
        if not passes_trend:
            return "SELL_EXIT"
        if rank > rank_buffer:
            return "SELL_EXIT"
        return "HOLD_RETAIN"


def calculate_weight_adjustment_action(
    current_weight: float,
    target_weight: float,
    deadband: float = 0.02,
) -> str:
    """
    Classifies weight adjustment based on minimum trade deadband threshold.

    Args:
        current_weight: Current portfolio weight of asset in [0.0, 1.0].
        target_weight: Recommended target weight in [0.0, 1.0].
        deadband: Minimum weight change threshold to trigger a rebalance trade (default: 0.02 / 2%).

    Returns:
        "NEW_BUY", "SELL_EXIT", "REBALANCE_ADD", "REBALANCE_TRIM", or "NO_CHANGE".
    """
    delta = target_weight - current_weight

    if current_weight <= 1e-6 and target_weight > 1e-6:
        return "NEW_BUY"
    if current_weight > 1e-6 and target_weight <= 1e-6:
        return "SELL_EXIT"

    eps = 1e-7
    if delta > (deadband + eps):
        return "REBALANCE_ADD"
    elif delta < -(deadband + eps):
        return "REBALANCE_TRIM"
    else:
        return "NO_CHANGE"


def generate_rebalance_plan(
    current_holdings: Union[List[str], Dict[str, float]],
    target_recommendation: PortfolioRecommendation,
    deadband: float = 0.02,
    total_capital: Optional[float] = None,
    prices: Optional[Dict[str, float]] = None,
    rank_buffer: int = 20,
    top_n: int = 10,
    as_of_date: Optional[str] = None,
) -> RebalancePlan:
    """
    Generates a full rebalancing plan comparing current holdings against recommended target.

    Args:
        current_holdings: Either a list of current holding symbols,
                          or a dict of {symbol: current_weight},
                          or a dict of {symbol: current_shares}.
        target_recommendation: Target PortfolioRecommendation from portfolio builder.
        deadband: Weight adjustment deadband threshold (default: 0.02 / 2%).
        total_capital: Portfolio capital value (defaults to recommendation's total_capital).
        prices: Optional dictionary of symbol -> price for share calculations.
        rank_buffer: Hysteresis retention buffer (default: 20).
        top_n: Target portfolio size (default: 10).
        as_of_date: Evaluation date string.

    Returns:
        RebalancePlan with buys, holds, sells, and turnover analysis.
    """
    if total_capital is None:
        total_capital = target_recommendation.total_capital
    if as_of_date is None:
        as_of_date = target_recommendation.as_of_date or datetime.now().strftime("%Y-%m-%d")
    if prices is None:
        prices = {}

    # Parse current weights
    current_weights_map: Dict[str, float] = {}
    if isinstance(current_holdings, list):
        # If passed as a list of symbols with no weights, assume equal division among held
        n_held = len(current_holdings)
        if n_held > 0:
            for s in current_holdings:
                current_weights_map[s] = 1.0 / float(n_held)
    elif isinstance(current_holdings, dict):
        # Could be weights or shares
        # Check if values sum to approximately <= 1.0 or are large integers
        vals = list(current_holdings.values())
        if vals and all(isinstance(v, (int, float)) and v >= 0 for v in vals):
            total_val = sum(vals)
            if total_val <= 1.05:
                current_weights_map = {k: float(v) for k, v in current_holdings.items()}
            else:
                # Value appears to be share counts or cash amounts -> normalize
                for k, v in current_holdings.items():
                    p = prices.get(k, 1.0)
                    pos_val = float(v) * p
                    current_weights_map[k] = pos_val / max(total_capital, 1.0)

    # Build target map
    target_constituents_map: Dict[str, PortfolioConstituent] = {
        c.symbol: c for c in target_recommendation.target_constituents
    }

    # Combined universe of all active symbols (current + target)
    all_symbols = sorted(list(set(list(current_weights_map.keys()) + list(target_constituents_map.keys()))))

    buys: List[RebalanceInstruction] = []
    holds: List[RebalanceInstruction] = []
    sells: List[RebalanceInstruction] = []
    all_trades: List[RebalanceInstruction] = []

    for sym in all_symbols:
        curr_w = current_weights_map.get(sym, 0.0)
        target_c = target_constituents_map.get(sym)
        target_w = target_c.weight if target_c is not None else 0.0
        rank = target_c.rank if target_c is not None else None
        price = prices.get(sym, target_c.close if target_c is not None else 0.0)

        delta_w = target_w - curr_w

        # Sizing calculations
        curr_val = total_capital * curr_w
        target_val = total_capital * target_w
        curr_shares = int(curr_val // price) if price > 0 else 0
        target_shares = int(target_val // price) if price > 0 else 0
        trade_shares = target_shares - curr_shares
        estimated_trade_value = float(trade_shares * price)

        # Classify trade
        if curr_w <= 1e-6 and target_w > 1e-6:
            action = "NEW_BUY"
            reason = f"Rank #{rank} entrant into Top {top_n} momentum"
        elif curr_w > 1e-6 and target_w <= 1e-6:
            action = "SELL_EXIT"
            if target_c is None:
                reason = f"Rank dropped beyond retention buffer ({rank_buffer}) or failed trend"
            else:
                reason = f"Liquidated due to macro cash reallocation or trend exit"
        else:
            # Currently held and targeted
            eps = 1e-7
            if delta_w > (deadband + eps):
                action = "REBALANCE_ADD"
                reason = f"Weight increased (+{delta_w*100:.1f}%) exceeding {deadband*100:.1f}% deadband"
            elif delta_w < -(deadband + eps):
                action = "REBALANCE_TRIM"
                reason = f"Weight decreased ({delta_w*100:.1f}%) exceeding {deadband*100:.1f}% deadband"
            else:
                action = "HOLD_RETAIN"
                reason = f"Weight change ({delta_w*100:.1f}%) within {deadband*100:.1f}% deadband (retained)"

        instr = RebalanceInstruction(
            symbol=sym,
            action=action,
            current_weight=curr_w,
            target_weight=target_w,
            weight_delta=delta_w,
            current_shares=curr_shares,
            target_shares=target_shares,
            trade_shares=trade_shares,
            price=price,
            estimated_trade_value=estimated_trade_value,
            rank=rank,
            reason=reason,
        )

        all_trades.append(instr)

        if action in ("NEW_BUY", "REBALANCE_ADD"):
            buys.append(instr)
        elif action in ("SELL_EXIT", "REBALANCE_TRIM"):
            sells.append(instr)
        else:
            holds.append(instr)

    # Calculate portfolio turnover (one-way turnover including cash change)
    cash_curr = max(0.0, 1.0 - sum(current_weights_map.values()))
    cash_target = max(0.0, 1.0 - sum(t.target_weight for t in all_trades))
    cash_delta = cash_target - cash_curr
    turnover_pct = float(0.5 * (sum(abs(t.weight_delta) for t in all_trades) + abs(cash_delta)))
    estimated_cash_change = float(-sum(t.estimated_trade_value for t in all_trades))

    return RebalancePlan(
        evaluation_date=as_of_date,
        universe=target_recommendation.universe,
        total_capital=total_capital,
        current_holdings=list(current_weights_map.keys()),
        target_holdings=[c.symbol for c in target_recommendation.target_constituents if c.weight > 0],
        buys=buys,
        holds=holds,
        sells=sells,
        all_trades=all_trades,
        turnover_pct=turnover_pct,
        estimated_cash_change=estimated_cash_change,
        summary={
            "total_trades": len([t for t in all_trades if t.action != "HOLD_RETAIN"]),
            "new_buys_count": len([b for b in buys if b.action == "NEW_BUY"]),
            "sells_count": len([s for s in sells if s.action == "SELL_EXIT"]),
            "rebalance_adds_count": len([b for b in buys if b.action == "REBALANCE_ADD"]),
            "rebalance_trims_count": len([s for s in sells if s.action == "REBALANCE_TRIM"]),
            "holds_retained_count": len(holds),
            "turnover_pct": turnover_pct,
            "deadband": deadband,
        },
    )

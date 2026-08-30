"""
Quantitative Momentum Portfolio Multi-Format Exporter.

Provides structured export functionality for PortfolioRecommendation and RebalancePlan
into CSV, JSON, and executive Markdown formats.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.momentum.models import (
    PortfolioRecommendation,
    RebalancePlan,
)


def _ensure_dir(path: Union[str, Path]) -> Path:
    """Ensures parent directory exists and returns Path object."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def export_portfolio_csv(
    recommendation: PortfolioRecommendation,
    rebalance_plan: Optional[RebalancePlan] = None,
    filepath: Union[str, Path] = "reports/momentum/portfolio.csv",
) -> Path:
    """
    Exports recommended portfolio constituents and rebalance actions to CSV.

    Args:
        recommendation: Target PortfolioRecommendation.
        rebalance_plan: Optional RebalancePlan with trade instructions.
        filepath: Target CSV output path.

    Returns:
        Path to generated CSV file.
    """
    p = _ensure_dir(filepath)

    rows: List[Dict[str, Any]] = []
    trade_map = (
        {t.symbol: t for t in rebalance_plan.all_trades}
        if rebalance_plan is not None
        else {}
    )

    for c in recommendation.target_constituents:
        trade = trade_map.get(c.symbol)
        row = {
            "Symbol": c.symbol,
            "Rank": c.rank,
            "Weight_Pct": round(c.weight * 100.0, 3),
            "Target_Shares": c.target_shares,
            "Target_Value": round(c.target_value, 2),
            "Close_Price": round(c.close, 2),
            "Action": c.action.value if hasattr(c.action, "value") else str(c.action),
            "Composite_Score": round(c.composite_score, 4),
            "Clenow_Score": round(c.clenow_score, 4),
            "Clenow_Slope_Ann": round(c.clenow_slope_ann, 4),
            "Clenow_R2": round(c.clenow_r2, 4),
            "Mom_12_1_Pct": round(c.mom_12_1 * 100.0, 2),
            "Mom_6_1_Pct": round(c.mom_6_1 * 100.0, 2),
            "Volatility_Ann_Pct": round(c.volatility_ann * 100.0, 2),
            "NATR_14_Pct": round(c.natr_14, 2),
            "Current_Shares": trade.current_shares if trade else 0,
            "Trade_Shares": trade.trade_shares if trade else c.target_shares,
            "Estimated_Trade_Value": round(trade.estimated_trade_value, 2) if trade else round(c.target_value, 2),
            "Trade_Reason": trade.reason if trade else "Initial portfolio allocation",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return p


def export_portfolio_json(
    recommendation: PortfolioRecommendation,
    rebalance_plan: Optional[RebalancePlan] = None,
    filepath: Union[str, Path] = "reports/momentum/portfolio.json",
    indent: int = 2,
) -> Path:
    """
    Exports portfolio recommendation and rebalancing plan to structured JSON.

    Args:
        recommendation: Target PortfolioRecommendation.
        rebalance_plan: Optional RebalancePlan.
        filepath: Target JSON output path.
        indent: JSON indentation spaces.

    Returns:
        Path to generated JSON file.
    """
    p = _ensure_dir(filepath)

    data: Dict[str, Any] = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "generator": "Quantitative Momentum Portfolio Engine",
            "version": "1.0.0",
        },
        "recommendation": recommendation.to_dict(),
    }

    if rebalance_plan is not None:
        data["rebalance_plan"] = rebalance_plan.to_dict()

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

    return p


def export_portfolio_markdown(
    recommendation: PortfolioRecommendation,
    rebalance_plan: Optional[RebalancePlan] = None,
    filepath: Union[str, Path] = "reports/momentum/portfolio_summary.md",
) -> Path:
    """
    Generates an executive Markdown summary report of the momentum portfolio.

    Args:
        recommendation: Target PortfolioRecommendation.
        rebalance_plan: Optional RebalancePlan.
        filepath: Target Markdown output path.

    Returns:
        Path to generated Markdown file.
    """
    p = _ensure_dir(filepath)

    currency = "₹" if recommendation.universe.lower() == "india" else "$"
    regime_state = (
        recommendation.market_regime.state.value
        if hasattr(recommendation.market_regime.state, "value")
        else str(recommendation.market_regime.state)
    )

    lines: List[str] = []
    lines.append(f"# 🏆 Quantitative Momentum Portfolio Summary")
    lines.append(f"")
    lines.append(f"**Universe**: `{recommendation.universe.upper()}` | **Evaluation Date**: `{recommendation.as_of_date}` | **Weighting**: `{recommendation.weighting_scheme}`")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 1. Executive Key Performance Indicators")
    lines.append(f"")
    lines.append(f"| Metric | Value | Description |")
    lines.append(f"|---|---|---|")
    lines.append(f"| **Market Regime** | `{regime_state}` | Benchmark trend filter status |")
    lines.append(f"| **Total Capital** | `{currency}{recommendation.total_capital:,.2f}` | Allocated portfolio capital |")
    lines.append(f"| **Total Equity Allocation** | `{recommendation.total_equity_pct*100.0:.1f}%` | Total invested capital in equities |")
    lines.append(f"| **Cash Allocation** | `{recommendation.expected_cash_pct*100.0:.1f}%` | Defensive / unallocated cash balance |")
    lines.append(f"| **Weighted Clenow Score** | `{recommendation.weighted_clenow_score:.4f}` | Annualized exponential regression slope × R² |")
    lines.append(f"| **Weighted Volatility (Ann.)** | `{recommendation.weighted_volatility*100.0:.1f}%` | Capital-weighted annualized volatility |")
    lines.append(f"| **Portfolio Volatility (Ann.)** | `{recommendation.portfolio_volatility*100.0:.1f}%` | Unhedged combined asset volatility |")
    lines.append(f"| **Diversification Ratio ($N_{{eff}}$)** | `{recommendation.diversification_ratio:.2f}` | Inverse Herfindahl index (effective constituent count) |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 2. Recommended Top {len(recommendation.target_constituents)} Portfolio Constituents")
    lines.append(f"")
    lines.append(f"| Rank | Symbol | Action | Weight | Target Value | Shares | Close Price | Clenow Score | Slope (Ann.) | $R^2$ | 12-1 Mom | Vol (Ann.) |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|---|---|---|")

    for c in recommendation.target_constituents:
        action_badge = c.action.value if hasattr(c.action, "value") else str(c.action)
        lines.append(
            f"| **#{c.rank}** | **`{c.symbol}`** | `{action_badge}` | "
            f"`{c.weight*100.0:.2f}%` | {currency}{c.target_value:,.2f} | "
            f"`{c.target_shares}` | {currency}{c.close:,.2f} | "
            f"`{c.clenow_score:.3f}` | `{c.clenow_slope_ann*100.0:+.1f}%` | "
            f"`{c.clenow_r2:.2f}` | `{c.mom_12_1*100.0:+.1f}%` | "
            f"`{c.volatility_ann*100.0:.1f}%` |"
        )

    if rebalance_plan is not None:
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## 3. Rebalancing & Execution Plan")
        lines.append(f"")
        lines.append(f"- **Total One-Way Turnover**: `{rebalance_plan.turnover_pct*100.0:.2f}%`")
        lines.append(f"- **Estimated Cash Impact**: `{currency}{rebalance_plan.estimated_cash_change:+,.2f}`")
        lines.append(f"- **New Buys**: `{len(rebalance_plan.buys)}` | **Holds/Retained**: `{len(rebalance_plan.holds)}` | **Sells/Exits**: `{len(rebalance_plan.sells)}`")
        lines.append(f"")
        lines.append(f"### Trade Instructions")
        lines.append(f"")
        lines.append(f"| Symbol | Action | Curr Wt | Target Wt | Delta | Trade Shares | Est. Trade Value | Reason |")
        lines.append(f"|---|---|---|---|---|---|---|---|")
        for t in rebalance_plan.all_trades:
            lines.append(
                f"| **`{t.symbol}`** | `{t.action}` | `{t.current_weight*100.0:.1f}%` | "
                f"`{t.target_weight*100.0:.1f}%` | `{t.weight_delta*100.0:+.1f}%` | "
                f"`{t.trade_shares:+d}` | {currency}{t.estimated_trade_value:+,.2f} | "
                f"{t.reason} |"
            )

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Generated automatically by Quantitative Momentum Engine on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.*")
    lines.append(f"")

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return p


def export_all_reports(
    recommendation: PortfolioRecommendation,
    rebalance_plan: Optional[RebalancePlan] = None,
    output_dir: Union[str, Path] = "reports/momentum",
    base_name: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Exports portfolio recommendation and plan in all formats (CSV, JSON, Markdown).

    Args:
        recommendation: PortfolioRecommendation dataclass.
        rebalance_plan: Optional RebalancePlan dataclass.
        output_dir: Output directory.
        base_name: Optional base filename prefix (default: '{universe}_portfolio_{date}').

    Returns:
        Dictionary mapping format name ('csv', 'json', 'markdown') to Path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if base_name is None:
        date_str = recommendation.as_of_date or datetime.now().strftime("%Y%m%d")
        base_name = f"{recommendation.universe}_momentum_{date_str}"

    csv_path = out_path / f"{base_name}.csv"
    json_path = out_path / f"{base_name}.json"
    md_path = out_path / f"{base_name}.md"

    res_csv = export_portfolio_csv(recommendation, rebalance_plan, filepath=csv_path)
    res_json = export_portfolio_json(recommendation, rebalance_plan, filepath=json_path)
    res_md = export_portfolio_markdown(recommendation, rebalance_plan, filepath=md_path)

    # Also maintain latest default copies
    default_csv = out_path / "portfolio.csv"
    default_json = out_path / "portfolio.json"
    default_md = out_path / "portfolio_summary.md"

    export_portfolio_csv(recommendation, rebalance_plan, filepath=default_csv)
    export_portfolio_json(recommendation, rebalance_plan, filepath=default_json)
    export_portfolio_markdown(recommendation, rebalance_plan, filepath=default_md)

    return {
        "csv": res_csv,
        "json": res_json,
        "markdown": res_md,
    }

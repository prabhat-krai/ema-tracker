"""
Quantitative Momentum Portfolio Screener CLI Runner.

Command-line entry point to execute multi-universe momentum screening,
portfolio construction, rebalancing recommendations, and report exports.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.momentum.engine import MomentumPipeline, MomentumPipelineResult
from src.momentum.exporter import (
    export_all_reports,
    export_portfolio_csv,
    export_portfolio_json,
    export_portfolio_markdown,
)

logger = logging.getLogger("momentum.cli")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds command-line argument parser for momentum screener."""
    parser = argparse.ArgumentParser(
        prog="python -m src.momentum",
        description="Quantitative Momentum Portfolio Tracker & Screener CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--universe",
        "-u",
        type=str,
        choices=["india", "usa"],
        default="india",
        help="Target stock universe to screen ('india' for NSE Nifty 500, 'usa' for S&P 500)",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=10,
        help="Target number of top momentum constituents to select",
    )
    parser.add_argument(
        "--weighting",
        "-w",
        type=str,
        choices=["inv_vol", "equal", "bounded_parity"],
        default="inv_vol",
        help="Portfolio weighting scheme",
    )
    parser.add_argument(
        "--capital",
        "-c",
        type=float,
        default=1_000_000.0,
        help="Total portfolio capital value in local currency (₹ for India, $ for US)",
    )
    parser.add_argument(
        "--holdings",
        "-H",
        type=str,
        default=None,
        help="Comma-separated list of currently held tickers (e.g. 'RELIANCE.NS,TCS.NS' or 'AAPL,MSFT')",
    )
    parser.add_argument(
        "--rank-buffer",
        "-b",
        type=int,
        default=20,
        help="Hysteresis retention buffer rank cutoff (prevents turnover churn)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help="Minimum single-stock weight bound for bounded risk parity",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.20,
        help="Maximum single-stock weight bound for bounded risk parity",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.02,
        help="Weight change deadband threshold (e.g. 0.02 for 2%%)",
    )
    parser.add_argument(
        "--period",
        "-p",
        type=str,
        default="2y",
        help="Lookback historical period for price data (e.g. '1y', '2y')",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=90,
        help="Minimum daily trading bars required for stock eligibility",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        choices=["all", "csv", "json", "md", "none"],
        default="all",
        help="Report export format",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="reports/momentum",
        help="Directory to save exported reports",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline deterministic mode using synthetic test universe",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed verbose debug logging",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress banner and console summary tables",
    )

    return parser


def format_console_summary(result: MomentumPipelineResult) -> str:
    """Formats human-readable ASCII console summary of momentum screening result."""
    rec = result.portfolio
    plan = result.rebalance_plan
    currency = "₹" if result.universe == "india" else "$"
    regime_str = (
        result.market_regime.state.value
        if hasattr(result.market_regime.state, "value")
        else str(result.market_regime.state)
    )

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"🏆 QUANTITATIVE MOMENTUM PORTFOLIO REPORT — {result.universe.upper()}")
    lines.append(f"Evaluation Date: {rec.as_of_date} | Total Screened: {result.total_screened} | Qualified: {result.total_eligible}")
    lines.append("=" * 80)
    lines.append(f"Market Regime: {regime_str} (Benchmark: {result.market_regime.benchmark_symbol})")
    lines.append(f"Total Capital: {currency}{rec.total_capital:,.2f} | Total Equity: {rec.total_equity_pct*100.0:.1f}% | Cash: {rec.expected_cash_pct*100.0:.1f}%")
    lines.append(f"Weighted Clenow Score: {rec.weighted_clenow_score:.4f} | Port. Volatility (Ann.): {rec.portfolio_volatility*100.0:.1f}% | N_eff: {rec.diversification_ratio:.2f}")
    lines.append("-" * 80)
    lines.append(f"TOP {len(rec.target_constituents)} MOMENTUM CONSTITUENTS (Weighting: {rec.weighting_scheme}):")
    lines.append(f"{'Rank':<5} {'Symbol':<14} {'Action':<6} {'Weight':<8} {'Target Shares':<14} {'Close Price':<12} {'Clenow':<8} {'Slope':<8} {'R2':<6} {'12-1 M':<8} {'Vol':<6}")
    lines.append("-" * 80)

    for c in rec.target_constituents:
        action_str = c.action.value if hasattr(c.action, "value") else str(c.action)
        lines.append(
            f"#{c.rank:<4} {c.symbol:<14} {action_str:<6} {c.weight*100.0:>6.2f}% "
            f"{c.target_shares:>13d} {currency}{c.close:>10.2f} "
            f"{c.clenow_score:>7.3f} {c.clenow_slope_ann*100.0:>+6.1f}% "
            f"{c.clenow_r2:>5.2f} {c.mom_12_1*100.0:>+6.1f}% {c.volatility_ann*100.0:>5.1f}%"
        )

    if plan is not None and plan.all_trades:
        lines.append("-" * 80)
        lines.append(f"REBALANCING EXECUTION PLAN (One-Way Turnover: {plan.turnover_pct*100.0:.2f}% | Est. Cash Impact: {currency}{plan.estimated_cash_change:+,.2f}):")
        lines.append(f"{'Symbol':<14} {'Action':<15} {'Curr Wt':<9} {'Target Wt':<10} {'Trade Shares':<14} {'Trade Value':<14} {'Reason'}")
        lines.append("-" * 80)
        for t in plan.all_trades:
            lines.append(
                f"{t.symbol:<14} {t.action:<15} {t.current_weight*100.0:>6.1f}%   {t.target_weight*100.0:>7.1f}%   "
                f"{t.trade_shares:>+12d}   {currency}{t.estimated_trade_value:>+12.2f}   {t.reason}"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


def _generate_synthetic_offline_universe(universe: str, n_stocks: int = 50) -> tuple:
    """Generates synthetic deterministic universe and benchmark for offline execution."""
    rng = np.random.RandomState(42)
    universe_data = {}
    dates = pd.date_range(end=pd.Timestamp.now(), periods=300, freq="B")

    suffix = ".NS" if universe == "india" else ""
    symbols = [
        ("RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN", "LICI", "ITC", "HINDUNILVR",
         "LT", "BAJFINANCE", "MARUTI", "HCLTECH", "SUNPHARMA", "ONGC", "NTPC", "TATAMOTORS", "AXISBANK", "TITAN")
        if universe == "india"
        else
        ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "AVGO", "LLY",
         "JPM", "V", "UNH", "XOM", "MA", "PG", "COST", "JNJ", "HD", "MRK")
    ][0]

    for idx, base_sym in enumerate(symbols):
        sym = f"{base_sym}{suffix}"
        drift = 0.0008 + (20 - idx) * 0.0001
        vol = 0.012 + (idx % 5) * 0.002
        shocks = rng.normal(drift, vol, size=len(dates))
        cum_ret = np.exp(np.cumsum(shocks))
        p0 = 100.0 + idx * 25.0
        close = p0 * cum_ret
        high = close * (1.0 + rng.uniform(0.001, 0.015, size=len(dates)))
        low = close * (1.0 - rng.uniform(0.001, 0.015, size=len(dates)))
        open_p = (high + low) / 2.0
        vol_arr = rng.randint(100_000, 2_000_000, size=len(dates)).astype(float)

        df = pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol_arr,
        }, index=dates)
        universe_data[sym] = df

    # Benchmark
    bench_shocks = rng.normal(0.0004, 0.008, size=len(dates))
    bench_close = 20000.0 * np.exp(np.cumsum(bench_shocks))
    benchmark_df = pd.DataFrame({
        "open": bench_close * 0.999,
        "high": bench_close * 1.005,
        "low": bench_close * 0.995,
        "close": bench_close,
        "volume": np.full(len(dates), 10_000_000.0),
    }, index=dates)

    bench_symbol = "^NSEI" if universe == "india" else "^GSPC"
    return universe_data, benchmark_df, bench_symbol


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point function."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        # Parse current holdings
        holdings_list = []
        if args.holdings:
            holdings_list = [h.strip().upper() for h in args.holdings.split(",") if h.strip()]

        pipeline = MomentumPipeline(
            universe=args.universe,
            top_n=args.top,
            weighting_scheme=args.weighting,
            rank_buffer=args.rank_buffer,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
            deadband=args.deadband,
            total_capital=args.capital,
            min_history_bars=args.min_history,
        )

        if args.offline:
            if not args.quiet:
                logger.info(f"Running Momentum Pipeline in OFFLINE mode for universe '{args.universe}'")
            universe_data, benchmark_df, bench_sym = _generate_synthetic_offline_universe(args.universe)
            result = pipeline.run_with_data(
                universe_data=universe_data,
                benchmark_df=benchmark_df,
                benchmark_symbol=bench_sym,
                current_holdings=holdings_list,
            )
        else:
            if not args.quiet:
                logger.info(f"Executing live momentum screening for universe '{args.universe}' (period: {args.period})")
            result = pipeline.run(
                period=args.period,
                current_holdings=holdings_list,
            )

        # Print console table
        if not args.quiet:
            print("\n" + format_console_summary(result) + "\n")

        # Report exports
        if args.export != "none":
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.export == "all":
                paths = export_all_reports(
                    recommendation=result.portfolio,
                    rebalance_plan=result.rebalance_plan,
                    output_dir=out_dir,
                )
                if not args.quiet:
                    for fmt, p in paths.items():
                        logger.info(f"Exported {fmt.upper()} report -> {p}")
            elif args.export == "csv":
                csv_p = export_portfolio_csv(
                    result.portfolio,
                    result.rebalance_plan,
                    filepath=out_dir / f"{result.universe}_momentum.csv",
                )
                if not args.quiet:
                    logger.info(f"Exported CSV report -> {csv_p}")
            elif args.export == "json":
                json_p = export_portfolio_json(
                    result.portfolio,
                    result.rebalance_plan,
                    filepath=out_dir / f"{result.universe}_momentum.json",
                )
                if not args.quiet:
                    logger.info(f"Exported JSON report -> {json_p}")
            elif args.export == "md":
                md_p = export_portfolio_markdown(
                    result.portfolio,
                    result.rebalance_plan,
                    filepath=out_dir / f"{result.universe}_momentum_summary.md",
                )
                if not args.quiet:
                    logger.info(f"Exported Markdown report -> {md_p}")

        return 0

    except Exception as e:
        logger.exception(f"Momentum screener encountered a fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

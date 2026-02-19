#!/usr/bin/env python3
"""
EMA TA Rules Screener - Main Entry Point

Analyzes stocks and generates buy/sell signals
based on EMA Technical Analysis rules flowchart.

Usage:
    python -m src.main [--stocks N] [--delay SECONDS]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from . import config
    from .data_fetcher import fetch_weekly_data
    from .technical import analyze_stock
    from .ta_rules_engine import (
        Signal,
        SignalResult,
        analyze_with_ta_rules,
        format_signal_line,
        get_signal_emoji,
    )
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\nDid you forget to activate the virtual environment?")
    print("Try running: source venv/bin/activate\n")
    sys.exit(1)

# Set up logging
def setup_logging(log_dir: Path, market_prefix: str) -> Tuple[logging.Logger, Path]:
    """Configure logging to file and console."""
    log_dir.mkdir(exist_ok=True)
    
    today_str = datetime.now().strftime('%d-%m-%Y')
    
    # We prefix with the market, USA or INDIA
    log_path = log_dir / f"{market_prefix}_{today_str}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter("%(message)s")
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    logger = logging.getLogger()
    # Remove existing handlers to avoid duplicates if logging setup is called multiple times (though unlikely here)
    if logger.handlers:
        logger.handlers = []
        
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger, log_path


def print_header():
    """Print the screener header."""
    print("\n" + "=" * 70)
    print("  EMA TA Rules Screener")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def print_progress(current: int, total: int, symbol: str):
    """Print progress update."""
    pct = (current / total) * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}% | {current}/{total} | {symbol:15}", end="", flush=True)


def print_summary(results: Dict[Signal, List[SignalResult]], errors: int, currency_symbol: str = "₹"):
    """Print the final summary grouped by signal type."""
    print("\n\n" + "=" * 70)
    print("  ANALYSIS RESULTS")
    print("=" * 70)
    
    # Order of signals for display
    signal_order = [
        (Signal.BULLISH, "BULLISH SIGNALS (Buy candidates)"),
        (Signal.EXIT, "EXIT SIGNALS (Sell candidates)"),
        (Signal.CAUTIOUS, "CAUTIOUS (Reduce exposure)"),
        (Signal.FADING, "MOMENTUM FADING"),
        (Signal.HOLD_ADD, "MAINTAIN / ADD (Strong positions)"),
        (Signal.WAIT, "WAIT / WATCH (Consolidating)"),
    ]
    
    for signal, title in signal_order:
        if signal in results and results[signal]:
            emoji = get_signal_emoji(signal)
            print(f"\n{emoji} {title}:")
            print("-" * 50)
            for r in sorted(results[signal], key=lambda x: x.symbol):
                print(f"  {r.symbol:15} {currency_symbol}{r.current_price:>10.2f}  {r.reason}")
    
    # Print counts
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    total_analyzed = sum(len(v) for v in results.values())
    
    counts = []
    for signal, _ in signal_order:
        count = len(results.get(signal, []))
        if count > 0:
            emoji = get_signal_emoji(signal)
            counts.append(f"{emoji} {signal.value}: {count}")
    
    print(f"\n  Total Analyzed: {total_analyzed}")
    print(f"  Errors/Skipped: {errors}")
    print(f"\n  {' | '.join(counts) if counts else 'No signals generated'}")
    print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EMA TA Rules Stock Screener")
    parser.add_argument(
        "--stocks", "-n",
        type=int,
        default=None,
        help="Process top N stocks from the list (default: all)"
    )
    parser.add_argument(
        "--tickers", "-t",
        type=str,
        default=None,
        help="Comma-separated list of specific stock symbols to analyze (e.g. RELIANCE,TCS)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=config.API_DELAY_SECONDS,
        help=f"Delay between API calls in seconds (default: {config.API_DELAY_SECONDS})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="Run backtest on historical data (last 1 year)"
    )
    parser.add_argument(
        "--usa", "-USA",
        action="store_true",
        help="Use top 100 US stocks universe instead of India universe"
    )
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=1,
        help="Number of years for backtest (default: 1)"
    )
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be >= 0")
    if args.stocks is not None and args.stocks < 1:
        parser.error("--stocks must be >= 1")
    if args.years < 1:
        parser.error("--years must be >= 1")
    
    # Setup
    log_dir = Path(__file__).parent.parent / "logs"
    market_prefix_log = "USA" if args.usa else "INDIA"
    logger, log_path = setup_logging(log_dir, market_prefix_log)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get stock list
    market = "usa" if args.usa else "india"
    currency_symbol = "$" if args.usa else "₹"
    if args.tickers:
        all_stocks = [s.strip().upper() for s in args.tickers.split(",") if s.strip()]
    elif args.usa:
        all_stocks = config.get_usa_top_100_stocks()
    else:
        all_stocks = config.get_all_stocks()
        
    if args.stocks is not None:
        all_stocks = all_stocks[:args.stocks]

    if not all_stocks:
        print_header()
        print("  No stocks selected. Check --tickers/--stocks arguments.\n")
        return
    
    if args.backtest:
        from .backtester import run_backtest_for_symbol
        print_header()
        print(f"  RUNNING BACKTEST on {len(all_stocks)} stocks (Last {args.years} Year(s))")
        print(f"  Market: {'USA Top 100' if args.usa else 'India (Nifty 100 + Midcap 150)'}")
        print(f"  Analyzing...")
        
        total_trades = 0
        winning_trades = 0
        sum_trade_returns = 0.0
        
        for i, symbol in enumerate(all_stocks, 1):
            print_progress(i, len(all_stocks), symbol)
            try:
                # Fetch data (history needs to be enough for backtest + EMA warm up of ~1 year)
                fetch_years = args.years + 1
                df = fetch_weekly_data(symbol, years=fetch_years, delay=args.delay, market=market)
                if df is None:
                    logger.error(f"{symbol}: No data available, aborting backtest.")
                    print("\n\n" + "=" * 50)
                    print("  BACKTEST FAILED")
                    print("=" * 50)
                    print(f"  Reason: No data available for {symbol}")
                    print("  The strategy cannot run without historical candles.")
                    print("=" * 50 + "\n")
                    raise SystemExit(2)
                
                # Run backtest
                portfolio = run_backtest_for_symbol(symbol, df, lookback_weeks=args.years * 52)
                
                # Stats
                final_price = float(df.iloc[-1]["close"])
                res = portfolio.get_performance(current_prices={symbol: final_price})
                
                total_trades += res.total_trades
                winning_trades += res.winning_trades
                sum_trade_returns += sum(t.return_pct for t in res.trades if t.return_pct is not None)
                
                if res.total_trades > 0:
                    logger.info(
                        f"{symbol}: {res.total_trades} trades, Win Rate: {res.win_rate:.1%}, "
                        f"Avg Return: {res.total_return:.1%}"
                    )
                    # Print detailed trade log
                    print(f"\n  --- Trade Log for {symbol} ---")
                    print("  Date       | Action | Symbol     | Price   | Details")
                    print("  " + "-"*55)
                    for log_entry in portfolio.log:
                        print(f"  {log_entry}")
                    print("  " + "-"*55 + "\n")
                    
            except Exception as e:
                logger.error(f"{symbol}: Backtest failed - {e}")

        print("\n\n" + "=" * 50)
        print("  BACKTEST RESULTS (Summary)")
        print("=" * 50)
        print(f"  Stocks Tested: {len(all_stocks)}")
        print(f"  Total Trades: {total_trades}")
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_return = sum_trade_returns / total_trades
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Avg Return / Trade: {avg_return:.1%}")
        else:
            print("  No trades were generated in the test window.")
        print("=" * 50 + "\n")
        return

    # Normal mode (current analysis)
    print_header()
    print(f"  Log file: {log_path}")
    print(f"  Market: {'USA Top 100' if args.usa else 'India (Nifty 100 + Midcap 150)'}")
    print(f"  Analyzing {len(all_stocks)} stocks with {args.delay}s delay between requests")
    print(f"  Estimated time: {len(all_stocks) * args.delay / 60:.1f} minutes\n")
    
    # Results storage
    results: Dict[Signal, List[SignalResult]] = defaultdict(list)
    errors = 0
    
    # Process each stock
    for i, symbol in enumerate(all_stocks, 1):
        print_progress(i, len(all_stocks), symbol)
        
        try:
            # Fetch data
            df = fetch_weekly_data(symbol, delay=args.delay, market=market)
            
            if df is None:
                logger.debug(f"{symbol}: No data available")
                errors += 1
                continue
            
            # Technical analysis
            indicators = analyze_stock(symbol, df)
            
            if indicators is None:
                logger.debug(f"{symbol}: Analysis failed")
                errors += 1
                continue
            
            # Apply TA rules
            signal_result = analyze_with_ta_rules(indicators)
            results[signal_result.signal].append(signal_result)
            
            # Log the result
            logger.info(format_signal_line(signal_result, currency_symbol=currency_symbol))
            
        except Exception as e:
            logger.error(f"{symbol}: Unexpected error - {e}")
            errors += 1
    
    # Print summary
    print_summary(results, errors, currency_symbol=currency_symbol)
    
    # Log summary to file
    logger.info("=" * 50)
    logger.info("SCAN COMPLETE")
    logger.info(f"Total: {sum(len(v) for v in results.values())} | Errors: {errors}")
    for signal in Signal:
        if signal in results:
            logger.info(f"{signal.value}: {len(results[signal])}")


if __name__ == "__main__":
    main()

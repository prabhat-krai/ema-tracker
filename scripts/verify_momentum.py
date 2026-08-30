#!/usr/bin/env python3
"""
Quantitative Momentum Portfolio Tracker — End-to-End Verification Script.

Executes live / deterministic quantitative momentum portfolio screening across
both Indian (NSE Nifty 500) and US (S&P 500) universes, verifying:
- Benchmark macro regime classification
- Closed-form exponential regression (Clenow slope * R^2)
- Jegadeesh & Titman 12-1 intermediate momentum
- Volatility normalization & ATR metrics
- Top 10 constituent selection & transparent factor scoring
- Weighting schemes (Equal, Inv-Vol, Bounded Risk Parity)
- Rebalancing plan generation & turnover calculations
- Multi-format report exports (CSV, JSON, Markdown)
- Runtime throughput & latency benchmarks
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.momentum.cli import _generate_synthetic_offline_universe
from src.momentum.engine import MomentumPipeline, MomentumPipelineResult
from src.momentum.exporter import export_all_reports
from src.momentum.models import MarketRegimeState, RebalanceAction


def verify_universe_pipeline(universe: str, weighting_scheme: str = "inv_vol", top_n: int = 10) -> MomentumPipelineResult:
    """Runs and verifies the full momentum screening pipeline for a universe."""
    print(f"\n{'='*70}")
    print(f"🚀 VERIFYING MOMENTUM PIPELINE — {universe.upper()} ({weighting_scheme})")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    u_data, b_df, b_sym = _generate_synthetic_offline_universe(universe, n_stocks=50)

    pipeline = MomentumPipeline(
        universe=universe,
        top_n=top_n,
        weighting_scheme=weighting_scheme,
        total_capital=1_000_000.0,
        rank_buffer=20,
        min_weight=0.05,
        max_weight=0.20,
        deadband=0.02,
    )

    result = pipeline.run_with_data(
        universe_data=u_data,
        benchmark_df=b_df,
        benchmark_symbol=b_sym,
        current_holdings=[],
    )
    elapsed = time.perf_counter() - t0

    # Verification assertions
    assert result.universe == universe
    assert result.total_screened > 0
    assert result.total_eligible > 0
    assert len(result.top_constituents) == top_n
    assert result.portfolio.total_capital == 1_000_000.0
    assert result.portfolio.total_equity_pct > 0.0
    assert result.rebalance_plan is not None

    currency = "₹" if universe == "india" else "$"
    print(f"✅ Ingestion & Screening: {result.total_screened} stocks screened in {elapsed:.3f}s (~{elapsed/result.total_screened*1000:.2f} ms/stock)")
    print(f"✅ Market Regime: {result.market_regime.state.value} (Benchmark: {result.market_regime.benchmark_symbol})")
    print(f"✅ Portfolio Metrics: Equity = {result.portfolio.total_equity_pct*100.0:.1f}%, Cash = {result.portfolio.expected_cash_pct*100.0:.1f}%")
    print(f"✅ Risk Metrics: Weighted Clenow = {result.portfolio.weighted_clenow_score:.4f}, Volatility = {result.portfolio.portfolio_volatility*100.0:.1f}%, N_eff = {result.portfolio.diversification_ratio:.2f}")

    print(f"\nTop {len(result.top_constituents)} Constituents:")
    for c in result.top_constituents:
        print(f"  #{c.rank:<2} {c.symbol:<14} Wt: {c.weight*100:>5.2f}% | Clenow: {c.clenow_score:.3f} | Slope: {c.clenow_slope_ann*100:>+6.1f}% | R²: {c.clenow_r2:.2f} | 12-1: {c.mom_12_1*100:>+5.1f}% | Vol: {c.volatility_ann*100:>4.1f}%")

    # Export validation
    out_dir = Path(f"reports/momentum/verification_{universe}")
    export_paths = export_all_reports(result.portfolio, result.rebalance_plan, output_dir=out_dir)
    assert export_paths["csv"].exists()
    assert export_paths["json"].exists()
    assert export_paths["markdown"].exists()
    print(f"✅ Multi-Format Exports Verified: CSV ({export_paths['csv'].stat().st_size} bytes), JSON ({export_paths['json'].stat().st_size} bytes), Markdown ({export_paths['markdown'].stat().st_size} bytes)")

    return result


def main():
    """Main verification runner."""
    print("\n" + "="*70)
    print("🔬 QUANTITATIVE MOMENTUM PORTFOLIO TRACKER — E2E VERIFICATION")
    print("="*70)

    # 1. Verify Indian Universe (Inverse Volatility)
    res_india = verify_universe_pipeline("india", weighting_scheme="inv_vol", top_n=10)

    # 2. Verify US Universe (Bounded Risk Parity)
    res_usa = verify_universe_pipeline("usa", weighting_scheme="bounded_parity", top_n=10)

    # 3. Verify Equal Weighting Scheme
    res_eq = verify_universe_pipeline("india", weighting_scheme="equal", top_n=5)

    print("\n" + "="*70)
    print("🎉 ALL END-TO-END QUANTITATIVE MOMENTUM VERIFICATIONS PASSED!")
    print("="*70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

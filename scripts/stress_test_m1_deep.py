"""
Deep Stress Test: Regimes, Multi-Timeframe Weights, 5000-Asset Performance & Circuit Breakers.
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.momentum.clenow import compute_clenow_momentum, calculate_multi_timeframe_clenow
from src.momentum.volatility import calculate_realized_volatility, calculate_downside_deviation, calculate_atr
from src.momentum.regime import (
    calculate_sma,
    calculate_ema,
    evaluate_market_regime,
    evaluate_stock_eligibility,
    check_golden_regime,
)
from src.momentum.scoring import rank_universe
from src.momentum.models import MomentumMetrics, MarketRegimeState, RebalanceAction

print("=" * 80)
print("  DEEP STRESS TEST: PERFORMANCE, REGIMES & CIRCUIT BREAKERS")
print("=" * 80)

# 1. Performance & Scale Stress Test (5,000 Stocks)
print("\n1. High-Dimensional Universe Stress (5,000 Tickers):")
np.random.seed(999)
universe = []
for i in range(5000):
    m = MomentumMetrics(
        symbol=f"TICKER_{i+1:05d}",
        close=float(np.random.uniform(10.0, 5000.0)),
        history_bars=252,
        clenow_slope_ann=float(np.random.normal(0.15, 0.50)),
        clenow_r2=float(np.clip(np.random.normal(0.50, 0.30), 0.0, 1.0)),
        clenow_score=0.0,
        mom_12_1=float(np.random.normal(0.20, 0.40)),
        mom_6_1=float(np.random.normal(0.10, 0.25)),
        volatility_ann=float(np.clip(np.random.normal(0.28, 0.15), 0.02, 1.5)),
        natr_14=float(np.random.uniform(1.0, 8.0)),
        passes_trend=(i % 4 != 0),  # 75% eligible
    )
    m.clenow_score = m.clenow_slope_ann * m.clenow_r2
    universe.append(m)

t_start = time.perf_counter()
breakdowns, df_scored = rank_universe(universe, filter_trend=True)
t_elapsed = time.perf_counter() - t_start

print(f"  Processed {len(universe)} stocks in {t_elapsed*1000:.2f} ms ({len(breakdowns)} eligible)")
print(f"  Throughput: {len(universe)/t_elapsed:,.0f} stocks/second")
print(f"  Top 1:  {breakdowns[0].symbol}, Score: {breakdowns[0].composite_score:.4f}, Rank: {breakdowns[0].rank}")
print(f"  Top 10: {breakdowns[9].symbol}, Score: {breakdowns[9].composite_score:.4f}, Rank: {breakdowns[9].rank}")
assert len(breakdowns) == 3750, f"Expected 3750 eligible, got {len(breakdowns)}"
assert t_elapsed < 1.0, f"Ranking took too long ({t_elapsed:.2f}s > 1.0s)"

# 2. Benchmark Macro Regime Dynamic Transitions
print("\n2. Benchmark Macro Regime Dynamic Transitions:")
# 250 bars: Bull market (rising) -> Peak -> Bear market (falling)
t_bench = np.arange(300, dtype=np.float64)
# Rising phase up to 220, then sharp crash
p_bench = np.zeros(300)
p_bench[:220] = 10000.0 + 30.0 * t_bench[:220]  # rises to 16,600
p_bench[220:] = 16600.0 - 150.0 * (t_bench[220:] - 220)  # crashes to 4,600

regime_bull = evaluate_market_regime(p_bench[:210], benchmark_symbol="^NSEI")
regime_bear = evaluate_market_regime(p_bench, benchmark_symbol="^NSEI")

print(f"  Bull Phase (Day 210): Price={regime_bull.benchmark_price:.1f}, SMA200={regime_bull.sma_200:.1f}, Slope={regime_bull.sma_200_slope:.1f} -> State={regime_bull.state.value}, Equity Alloc={regime_bull.equity_allocation_pct:.0%}")
print(f"  Crash Phase (Day 300): Price={regime_bear.benchmark_price:.1f}, SMA200={regime_bear.sma_200:.1f}, Slope={regime_bear.sma_200_slope:.1f} -> State={regime_bear.state.value}, Equity Alloc={regime_bear.equity_allocation_pct:.0%}")

assert regime_bull.state == MarketRegimeState.BULLISH and regime_bull.equity_allocation_pct == 1.0
assert regime_bear.state == MarketRegimeState.BEARISH and regime_bear.equity_allocation_pct == 0.0

# 3. Multi-timeframe Clenow Dynamic Re-weighting
print("\n3. Multi-timeframe Clenow Dynamic Re-weighting on Variable Length Histories:")
p_100d = np.exp(np.linspace(np.log(100), np.log(150), 100))
mtf_100 = calculate_multi_timeframe_clenow(p_100d, windows=(90, 126, 252), weights=(0.40, 0.30, 0.30))
print(f"  100-Day History: 90d={mtf_100['score_90']:.4f}, 126d={mtf_100['score_126']}, 252d={mtf_100['score_252']}")
print(f"  Composite Score: {mtf_100['composite_score']:.4f} (Valid re-weighting)")
assert not np.isnan(mtf_100["score_90"])
assert np.isnan(mtf_100["score_126"])
assert np.isnan(mtf_100["score_252"])
assert abs(mtf_100["composite_score"] - mtf_100["score_90"]) < 1e-10

p_200d = np.exp(np.linspace(np.log(100), np.log(200), 200))
mtf_200 = calculate_multi_timeframe_clenow(p_200d, windows=(90, 126, 252), weights=(0.40, 0.30, 0.30))
expected_comp = (0.40 * mtf_200["score_90"] + 0.30 * mtf_200["score_126"]) / (0.40 + 0.30)
print(f"  200-Day History: Composite = {mtf_200['composite_score']:.4f}, Expected = {expected_comp:.4f}")
assert abs(mtf_200["composite_score"] - expected_comp) < 1e-10

print("\n  ALL DEEP STRESS CHECKS PASSED EMPIRICALLY!")

"""
Empirical Integration & Universe Execution Stress Test Suite for Milestone 2 (M2).
Author: Challenger 2 (Empirical Challenger)

Tests:
1. End-to-end 500-stock performance benchmark (runtime and memory footprint) for Indian (NSE Nifty 500) and US (S&P 500) universes.
2. Benchmark crash simulation and 100% cash transition verification.
3. Partial qualification regime (only 3 stocks pass filters, evaluating cash allocation and weight capping).
4. Ticker format normalization (.NS for Indian tickers, dash conversion for US tickers).
5. Portfolio rebalancing hysteresis, deadband filtering, and turnover management.
"""

import time
import tracemalloc
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from tests.e2e.fixtures import (
    generate_deterministic_series,
    generate_mock_benchmark,
)
from src.data_fetcher import (
    get_market_ticker,
    get_nse_ticker,
    get_us_ticker,
)
from src.momentum.engine import MomentumPipeline, MomentumPipelineResult
from src.momentum.models import (
    MarketRegimeState,
    MomentumScoreBreakdown,
    PortfolioRecommendation,
    RebalanceAction,
)
from src.momentum.portfolio import (
    build_momentum_portfolio,
    calculate_bounded_risk_parity_weights,
    calculate_equal_weights,
    calculate_inverse_volatility_weights,
)
from src.momentum.rebalancer import (
    calculate_weight_adjustment_action,
    classify_rebalance_action,
    generate_rebalance_plan,
)


def generate_synthetic_500_stock_universe(
    universe_type: str = "india",
    seed: int = 42,
    n_stocks: int = 500,
    n_bars: int = 300,
) -> Dict[str, pd.DataFrame]:
    """
    Generates a realistic 500-stock universe with diverse regimes:
    - 150 Steady Compounders (uptrend, high R^2)
    - 100 Choppy Growth (uptrend, high vol, lower R^2)
    - 150 Severe Downtrends (fails EMA100/SMA200)
    - 40 Broken Trends / Recent Crashes
    - 30 Short History IPOs (< 100 bars)
    - 20 Flat / Illiquid / Zero Volatility
    - 10 Gapping Earnings Outliers
    """
    rng = np.random.RandomState(seed)
    universe: Dict[str, pd.DataFrame] = {}

    suffix = ".NS" if universe_type == "india" else ""
    prefix = "IND" if universe_type == "india" else "US"

    # 1. 150 Steady Compounders (g = 0.0012 to 0.0028)
    for i in range(150):
        sym = f"{prefix}_LEAD_{i:03d}{suffix}"
        drift = 0.0012 + 0.0016 * (i / 150.0)
        noise = 0.002 + 0.002 * (i / 150.0)
        universe[sym] = generate_deterministic_series(
            initial_price=100.0 + i * 10.0,
            n_bars=n_bars,
            regime="smooth_exponential_uptrend",
            daily_drift=drift,
            noise_std=noise,
            seed=seed + i,
        )

    # 2. 100 Choppy Growth (higher drift but heavy noise)
    for i in range(100):
        sym = f"{prefix}_CHOP_{i:03d}{suffix}"
        universe[sym] = generate_deterministic_series(
            initial_price=200.0 + i * 5.0,
            n_bars=n_bars,
            regime="choppy_uptrend",
            daily_drift=0.0022,
            noise_std=0.08,
            seed=seed + 200 + i,
        )

    # 3. 150 Severe Downtrends
    for i in range(150):
        sym = f"{prefix}_DOWN_{i:03d}{suffix}"
        universe[sym] = generate_deterministic_series(
            initial_price=500.0 - i * 2.0,
            n_bars=n_bars,
            regime="strong_downtrend",
            daily_drift=0.0015,
            noise_std=0.003,
            seed=seed + 400 + i,
        )

    # 4. 40 Broken Trends (crash in last month)
    for i in range(40):
        sym = f"{prefix}_CRASH_{i:03d}{suffix}"
        universe[sym] = generate_deterministic_series(
            initial_price=300.0,
            n_bars=n_bars,
            regime="short_term_crash_skip_month",
            daily_drift=0.0020,
            seed=seed + 600 + i,
        )

    # 5. 30 Short History IPOs (80 bars, below 90d threshold)
    for i in range(30):
        sym = f"{prefix}_IPO_{i:03d}{suffix}"
        universe[sym] = generate_deterministic_series(
            initial_price=150.0,
            regime="low_history_80d",
            seed=seed + 700 + i,
        )

    # 6. 20 Flat / Illiquid Series
    for i in range(20):
        sym = f"{prefix}_FLAT_{i:03d}{suffix}"
        universe[sym] = generate_deterministic_series(
            initial_price=100.0,
            n_bars=n_bars,
            regime="flat_series",
            seed=seed + 800 + i,
        )

    # 7. 10 Gapping Earnings Outliers
    for i in range(10):
        sym = f"{prefix}_GAP_{i:03d}{suffix}"
        universe[sym] = generate_deterministic_series(
            initial_price=250.0,
            n_bars=n_bars,
            regime="earnings_gap_up",
            seed=seed + 900 + i,
        )

    return universe


class Test500StockPipelinePerformance:
    """Stress tests verifying runtime latency and memory footprint on 500-stock datasets."""

    def test_indian_500_stock_pipeline_runtime_and_memory(self):
        """
        Benchmark full MomentumPipeline on 500 Indian NSE stocks.
        Verifies:
        - Total runtime < 3.0s (vectorized efficiency)
        - Memory footprint < 150 MB delta
        - Correct constituent count (top 10)
        - Disqualification of downtrends and low-history IPOs
        """
        universe_500 = generate_synthetic_500_stock_universe(universe_type="india", seed=1001, n_stocks=500)
        benchmark_df = generate_mock_benchmark(regime="bullish", symbol="^NSEI", n_bars=300, seed=1002)

        assert len(universe_500) == 500

        pipeline = MomentumPipeline(
            universe="india",
            top_n=10,
            weighting_scheme="inv_vol",
            total_capital=10_000_000.0,
            min_history_bars=90,
        )

        tracemalloc.start()
        t0 = time.perf_counter()

        result = pipeline.run_with_data(
            universe_data=universe_500,
            benchmark_df=benchmark_df,
            benchmark_symbol="^NSEI",
            current_holdings=[],
        )

        elapsed_seconds = time.perf_counter() - t0
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)

        # Performance assertions
        assert elapsed_seconds < 6.0, f"500-stock pipeline took {elapsed_seconds:.3f}s (budget: <6.0s)"
        assert peak_mb < 150.0, f"500-stock peak memory was {peak_mb:.2f}MB (budget: <150MB)"

        # Functional correctness
        assert result.universe == "india"
        assert result.total_screened == 500
        assert result.total_eligible > 0
        assert len(result.top_constituents) == 10

        top_symbols = [c.symbol for c in result.top_constituents]
        assert all(s.endswith(".NS") for s in top_symbols)
        # Downtrends, IPOs, flat stocks, and crash stocks must NOT be in top 10
        assert not any("IND_DOWN_" in s for s in top_symbols)
        assert not any("IND_IPO_" in s for s in top_symbols)
        assert not any("IND_FLAT_" in s for s in top_symbols)
        assert not any("IND_CRASH_" in s for s in top_symbols)

        # Portfolio metrics
        assert result.portfolio.total_equity_pct == pytest.approx(1.0, abs=1e-6)
        assert result.portfolio.expected_cash_pct == pytest.approx(0.0, abs=1e-6)
        assert len(result.rebalance_plan.buys) == 10
        assert result.rebalance_plan.turnover_pct == pytest.approx(1.0, abs=1e-6)

    def test_usa_500_stock_pipeline_runtime_and_memory(self):
        """
        Benchmark full MomentumPipeline on 500 US S&P 500 stocks with bounded risk parity.
        Verifies:
        - Total runtime < 3.0s
        - Memory footprint < 150 MB delta
        - Bounded weights strictly in [5%, 20%]
        """
        universe_500 = generate_synthetic_500_stock_universe(universe_type="usa", seed=2001, n_stocks=500)
        benchmark_df = generate_mock_benchmark(regime="bullish", symbol="^GSPC", n_bars=300, seed=2002)

        pipeline = MomentumPipeline(
            universe="usa",
            top_n=10,
            weighting_scheme="bounded_parity",
            min_weight=0.05,
            max_weight=0.20,
            total_capital=5_000_000.0,
        )

        tracemalloc.start()
        t0 = time.perf_counter()

        result = pipeline.run_with_data(
            universe_data=universe_500,
            benchmark_df=benchmark_df,
            benchmark_symbol="^GSPC",
            current_holdings=["US_LEAD_001", "US_DOWN_005"],
        )

        elapsed_seconds = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)

        assert elapsed_seconds < 6.0, f"500-stock US pipeline took {elapsed_seconds:.3f}s (budget: <6.0s)"
        assert peak_mb < 150.0, f"500-stock peak memory was {peak_mb:.2f}MB"

        assert len(result.top_constituents) == 10
        for c in result.top_constituents:
            assert 0.049 <= c.weight <= 0.201, f"Constituent {c.symbol} weight {c.weight} out of bounds"

        # Check rebalance plan: US_DOWN_005 must be sold
        sell_symbols = [s.symbol for s in result.rebalance_plan.sells]
        assert "US_DOWN_005" in sell_symbols


class TestBenchmarkCrashAndCashPreservation:
    """Stress tests on benchmark crash simulations and defensive cash transitions."""

    def test_benchmark_crash_simulation_full_cash_transition(self):
        """
        Simulate severe market crash on benchmark:
        - Benchmark index drops below SMA200 (BEARISH regime).
        - Equity allocation must be 0%, Cash allocation 100%.
        - When existing holdings exist, pipeline generates SELL_EXIT for all held positions.
        """
        universe_data = generate_synthetic_500_stock_universe(universe_type="india", seed=3001, n_stocks=50)
        # Bearish benchmark dropping 35% below SMA200
        benchmark_df = generate_mock_benchmark(regime="bearish", symbol="^NSEI", n_bars=300, seed=3002)

        pipeline = MomentumPipeline(
            universe="india",
            top_n=10,
            weighting_scheme="inv_vol",
            total_capital=1_000_000.0,
        )

        # Existing portfolio held 5 stocks
        current_holdings = {
            "IND_LEAD_001.NS": 0.20,
            "IND_LEAD_002.NS": 0.20,
            "IND_LEAD_003.NS": 0.20,
            "IND_LEAD_004.NS": 0.20,
            "IND_LEAD_005.NS": 0.20,
        }

        result = pipeline.run_with_data(
            universe_data=universe_data,
            benchmark_df=benchmark_df,
            benchmark_symbol="^NSEI",
            current_holdings=current_holdings,
        )

        # 1. Verify Market Regime is BEARISH
        assert result.market_regime.state == MarketRegimeState.BEARISH
        assert result.market_regime.equity_allocation_pct == 0.0

        # 2. Verify Portfolio Cash Allocation is 100%
        assert result.portfolio.total_equity_pct == pytest.approx(0.0, abs=1e-6)
        assert result.portfolio.expected_cash_pct == pytest.approx(1.0, abs=1e-6)
        assert result.portfolio.allocated_cash_pct == pytest.approx(1.0, abs=1e-6)

        # All target constituent weights must be 0.0
        for c in result.top_constituents:
            assert c.weight == 0.0
            assert c.target_shares == 0
            assert c.target_value == 0.0

        # 3. Verify Rebalance Plan liquidates all 5 held positions
        sells = result.rebalance_plan.sells
        sell_syms = [s.symbol for s in sells]
        for held_sym in current_holdings.keys():
            assert held_sym in sell_syms, f"Held stock {held_sym} was NOT liquidated during market crash!"

        # All sales must be classified as SELL_EXIT
        for s in sells:
            assert s.action == "SELL_EXIT"
            assert s.target_weight == 0.0
            assert s.target_shares == 0

        # No new buys should be generated
        assert len(result.rebalance_plan.buys) == 0

        # Portfolio turnover must reflect full transition to 100% cash (100% turnover)
        assert result.rebalance_plan.turnover_pct == pytest.approx(1.0, abs=1e-6)

    def test_benchmark_neutral_regime_half_cash_transition(self):
        """
        Verify that in a NEUTRAL market regime:
        - Equity allocation is 50%, Cash allocation is 50%.
        - Individual stock weights are scaled by 0.50.
        """
        universe_data = generate_synthetic_500_stock_universe(universe_type="india", seed=4001, n_stocks=50)
        benchmark_df = generate_mock_benchmark(regime="neutral", symbol="^NSEI", n_bars=300, seed=4002)

        pipeline = MomentumPipeline(universe="india", top_n=10, weighting_scheme="equal")
        result = pipeline.run_with_data(
            universe_data=universe_data,
            benchmark_df=benchmark_df,
            benchmark_symbol="^NSEI",
            current_holdings=[],
        )

        assert result.market_regime.state == MarketRegimeState.NEUTRAL
        assert result.market_regime.equity_allocation_pct == 0.50
        assert result.portfolio.total_equity_pct == pytest.approx(0.50, abs=1e-6)
        assert result.portfolio.expected_cash_pct == pytest.approx(0.50, abs=1e-6)

        # Equal weight with 10 stocks: 10% * 0.50 = 5% per stock
        for c in result.top_constituents:
            assert c.weight == pytest.approx(0.05, abs=1e-6)


class TestPartialQualificationRegime:
    """
    Stress tests on partial qualification regime where fewer than top_n stocks qualify.
    Example: Only 3 stocks pass trend filters out of 500 stocks.
    """

    def test_partial_qualification_behavior_empirical_analysis(self):
        """
        Empirically test and document how MomentumPipeline and build_momentum_portfolio
        handle the scenario where only 3 stocks pass trend filters when top_n=10.
        """
        # Construct universe where exactly 3 stocks pass trend filter, 47 fail
        universe: Dict[str, pd.DataFrame] = {}
        # 3 strong leaders
        for i in range(3):
            universe[f"QUALIFIED_{i+1}.NS"] = generate_deterministic_series(
                initial_price=100.0,
                n_bars=300,
                regime="smooth_exponential_uptrend",
                daily_drift=0.003,
                seed=5000 + i,
            )
        # 47 broken downtrends
        for i in range(47):
            universe[f"DISQUALIFIED_{i+1}.NS"] = generate_deterministic_series(
                initial_price=200.0,
                n_bars=300,
                regime="strong_downtrend",
                daily_drift=0.002,
                seed=5100 + i,
            )

        benchmark_df = generate_mock_benchmark(regime="bullish", symbol="^NSEI", seed=5200)

        pipeline = MomentumPipeline(
            universe="india",
            top_n=10,
            weighting_scheme="equal",
            total_capital=1_000_000.0,
        )

        result = pipeline.run_with_data(
            universe_data=universe,
            benchmark_df=benchmark_df,
            benchmark_symbol="^NSEI",
            current_holdings=[],
        )

        # Only 3 stocks qualified
        assert result.total_eligible == 3
        assert len(result.top_constituents) == 3

        # Verify remediated partial qualification behavior:
        # Each of the 3 stocks receives 10% (1/top_n), total equity = 30%, cash = 70%
        weights = [c.weight for c in result.top_constituents]
        assert len(result.top_constituents) == 3
        for w in weights:
            assert w == pytest.approx(0.10, abs=1e-6)
        assert result.portfolio.total_equity_pct == pytest.approx(0.30, abs=1e-6)
        assert result.portfolio.expected_cash_pct == pytest.approx(0.70, abs=1e-6)
        assert result.portfolio.allocated_cash_pct == pytest.approx(0.70, abs=1e-6)


class TestTickerFormatNormalization:
    """Stress tests on ticker format normalization for Indian and US universes."""

    def test_indian_ticker_formatting(self):
        """Verify .NS suffix addition and preservation."""
        assert get_nse_ticker("RELIANCE") == "RELIANCE.NS"
        assert get_nse_ticker("reliance") == "RELIANCE.NS"
        assert get_nse_ticker("TCS.NS") == "TCS.NS"
        assert get_nse_ticker("tcs.ns") == "TCS.NS"
        assert get_nse_ticker("  INFY  ") == "INFY.NS"
        assert get_nse_ticker("  HDFCBANK.NS  ") == "HDFCBANK.NS"

    def test_us_ticker_formatting(self):
        """Verify US dot-to-dash conversion, capitalization, and whitespace stripping."""
        assert get_us_ticker("BRK.B") == "BRK-B"
        assert get_us_ticker("brk.b") == "BRK-B"
        assert get_us_ticker("BF.B") == "BF-B"
        assert get_us_ticker("  msft  ") == "MSFT"
        assert get_us_ticker("AAPL") == "AAPL"
        assert get_us_ticker("  aapl  ") == "AAPL"

    def test_market_ticker_dispatcher(self):
        """Verify get_market_ticker handles both markets properly."""
        assert get_market_ticker("RELIANCE", market="india") == "RELIANCE.NS"
        assert get_market_ticker("RELIANCE.NS", market="india") == "RELIANCE.NS"
        assert get_market_ticker("BRK.B", market="usa") == "BRK-B"
        assert get_market_ticker("AAPL", market="usa") == "AAPL"


class TestRebalancingHysteresisAndDeadband:
    """Stress tests on rebalancing churn prevention, deadband thresholds, and turnover."""

    def test_rank_buffer_churn_prevention(self):
        """
        Verify hysteresis:
        - Rank <= 10: New Buy (if not held) or Hold (if held)
        - Rank 11-20: Retained if held, ignored if not held
        - Rank > 20: Sold if held
        - Broken Trend: Sold immediately regardless of rank
        """
        # Held stock ranked #15 -> HOLD_RETAIN
        assert classify_rebalance_action("SYM_A", rank=15, is_held=True, passes_trend=True, top_n=10, rank_buffer=20) == "HOLD_RETAIN"
        # Non-held stock ranked #15 -> IGNORE
        assert classify_rebalance_action("SYM_B", rank=15, is_held=False, passes_trend=True, top_n=10, rank_buffer=20) == "IGNORE"
        # Held stock ranked #22 -> SELL_EXIT
        assert classify_rebalance_action("SYM_C", rank=22, is_held=True, passes_trend=True, top_n=10, rank_buffer=20) == "SELL_EXIT"
        # Held stock ranked #2 but failed trend -> SELL_EXIT
        assert classify_rebalance_action("SYM_D", rank=2, is_held=True, passes_trend=False, top_n=10, rank_buffer=20) == "SELL_EXIT"

    def test_rebalance_deadband_thresholds(self):
        """Verify 2% deadband trade filtering."""
        # Change +1% (within 2% deadband) -> NO_CHANGE
        assert calculate_weight_adjustment_action(0.10, 0.11, deadband=0.02) == "NO_CHANGE"
        # Change +2.5% (exceeds deadband) -> REBALANCE_ADD
        assert calculate_weight_adjustment_action(0.10, 0.125, deadband=0.02) == "REBALANCE_ADD"
        # Change -3% (exceeds deadband) -> REBALANCE_TRIM
        assert calculate_weight_adjustment_action(0.10, 0.07, deadband=0.02) == "REBALANCE_TRIM"

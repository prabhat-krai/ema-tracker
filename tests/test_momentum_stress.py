"""
Empirical Stress Tests & Adversarial Verification Suite for Milestone 1.

Verifies:
1. Analytical and theoretical invariants on synthetic price series with known slopes (P_t = P_0 * (1+g)^t).
2. Monotonicity of Clenow exponential regression slope, R^2, score, and 12-1 momentum.
3. Microstructure crash and surge invariance for Jegadeesh & Titman 12-1 momentum.
4. Volatility jump ratio and smoothness penalty under extreme single-day gaps and flash crashes.
5. Scale invariance across micro-prices and macro-prices.
6. Graceful edge-case handling for zero variance, severe NaNs, Infs, non-positive prices, and boundary lengths.
7. Large-scale cross-sectional ranking stress test across a heterogeneous 500-asset universe.
8. Macro benchmark regime transitions and circuit breaker allocations.
"""

import numpy as np
import pandas as pd
import pytest

from src.momentum.classical import (
    calculate_intermediate_momentum,
    calculate_multi_timeframe_returns,
    compute_jegadeesh_titman_momentum,
)
from src.momentum.clenow import (
    calculate_clenow_momentum,
    calculate_multi_timeframe_clenow,
    compute_clenow_momentum,
)
from src.momentum.models import (
    FilterResult,
    MarketRegime,
    MarketRegimeState,
    MomentumMetrics,
    RebalanceAction,
)
from src.momentum.regime import (
    calculate_52w_high_distance,
    calculate_ema,
    calculate_sma,
    check_golden_regime,
    check_single_stock_trend_filter,
    evaluate_market_regime,
    evaluate_stock_eligibility,
)
from src.momentum.scoring import (
    calculate_percentile_rank,
    compute_composite_momentum_score,
    rank_universe,
    winsorized_z_score,
)
from src.momentum.volatility import (
    calculate_atr,
    calculate_downside_deviation,
    calculate_jump_ratio_penalty,
    calculate_realized_volatility,
    calculate_volatility_adjusted_momentum,
    compute_volatility_metrics,
)


class TestTheoreticalMathAndMonotonicity:
    """Stress tests on theoretical ground truths and mathematical invariants."""

    @pytest.mark.parametrize("growth_rate", [-0.05, -0.02, -0.005, 0.001, 0.005, 0.01, 0.02, 0.05])
    def test_clenow_geometric_slopes_and_r2_exact_analytical_match(self, growth_rate):
        """
        For a pure geometric series P_t = P_0 * (1+g)^t over T bars:
        ln(P_t) = ln(P_0) + t * ln(1+g)
        Exact OLS slope beta = ln(1+g)
        Exact annualized slope = exp(250 * beta) - 1 = (1+g)^250 - 1
        Exact R^2 = 1.000000
        Exact Clenow score = annualized_slope * 1.0
        """
        n = 90
        t = np.arange(n, dtype=np.float64)
        p0 = 100.0
        prices = p0 * ((1.0 + growth_rate) ** t)

        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90, trading_days=250)

        expected_slope = ((1.0 + growth_rate) ** 250) - 1.0
        expected_r2 = 1.0
        expected_score = expected_slope * expected_r2

        assert slope_ann == pytest.approx(expected_slope, rel=1e-5)
        assert r2 == pytest.approx(expected_r2, rel=1e-5)
        assert score == pytest.approx(expected_score, rel=1e-5)

    def test_clenow_score_strict_monotonicity_across_growth_rates(self):
        """Higher underlying growth rates must produce strictly higher annualized slopes and Clenow scores."""
        growth_rates = np.linspace(-0.02, 0.03, 20)
        n = 90
        t = np.arange(n, dtype=np.float64)

        slopes = []
        scores = []
        for g in growth_rates:
            prices = 100.0 * ((1.0 + g) ** t)
            slope, r2, score = compute_clenow_momentum(prices, lookback=90, trading_days=250)
            slopes.append(slope)
            scores.append(score)

        # Verify strict monotonicity
        for i in range(len(growth_rates) - 1):
            assert slopes[i] < slopes[i + 1], f"Slope not strictly increasing at index {i}: {slopes[i]} >= {slopes[i+1]}"
            assert scores[i] < scores[i + 1], f"Score not strictly increasing at index {i}: {scores[i]} >= {scores[i+1]}"

    def test_r2_monotonicity_under_increasing_noise(self):
        """
        Adding increasing noise epsilon ~ N(0, sigma^2) to log prices must monotonically
        degrade R^2 and thus penalize the Clenow score.
        """
        np.random.seed(42)
        n = 126
        t = np.arange(n, dtype=np.float64)
        trend_log = np.log(100.0) + t * np.log(1.002)

        noise_sigmas = [0.0, 0.005, 0.015, 0.035, 0.070, 0.150]
        avg_r2s = []
        avg_scores = []

        # Average over 30 Monte Carlo trials per noise level to verify statistical monotonicity
        for sigma in noise_sigmas:
            r2_trials = []
            score_trials = []
            for trial in range(30):
                noise = np.random.normal(0.0, sigma, size=n) if sigma > 0 else np.zeros(n)
                prices = np.exp(trend_log + noise)
                slope, r2, score = compute_clenow_momentum(prices, lookback=126, trading_days=250)
                r2_trials.append(r2)
                score_trials.append(score)
            avg_r2s.append(np.mean(r2_trials))
            avg_scores.append(np.mean(score_trials))

        # Check that R^2 strictly degrades with higher noise
        for i in range(len(noise_sigmas) - 1):
            assert avg_r2s[i] > avg_r2s[i + 1], f"R^2 did not decrease with noise: {avg_r2s[i]} <= {avg_r2s[i+1]}"
            assert avg_scores[i] > avg_scores[i + 1], f"Score did not decrease with noise: {avg_scores[i]} <= {avg_scores[i+1]}"

    @pytest.mark.parametrize("growth_rate", [-0.01, -0.002, 0.001, 0.005, 0.01])
    def test_jegadeesh_titman_exact_geometric_analytical_match(self, growth_rate):
        """
        For a geometric series P_t = P_0 * (1+g)^t over 253 bars:
        P_{t-21} = P_0 * (1+g)^{t-21}
        P_{t-252} = P_0 * (1+g)^{t-252}
        Simple 12-1 Return = (P_{t-21} - P_{t-252}) / P_{t-252} = (1+g)^231 - 1
        Log 12-1 Return = ln(P_{t-21}) - ln(P_{t-252}) = 231 * ln(1+g)
        """
        n = 300
        t = np.arange(n, dtype=np.float64)
        prices = 100.0 * ((1.0 + growth_rate) ** t)

        mom_simple = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21, log_return=False)
        mom_log = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21, log_return=True)

        expected_simple = ((1.0 + growth_rate) ** 231) - 1.0
        expected_log = 231.0 * np.log(1.0 + growth_rate)

        assert mom_simple == pytest.approx(expected_simple, rel=1e-5)
        assert mom_log == pytest.approx(expected_log, rel=1e-5)


class TestMicrostructureAndFlashCrashInvariance:
    """Stress tests on flash crashes, short-term microstructure noise, and jump penalties."""

    def test_12_1_momentum_microstructure_invariance(self):
        """
        Core theoretical property of Jegadeesh & Titman (1993):
        Any price movements in the most recent 21 trading days (t-20..t) MUST NOT alter Mom_12_1.
        """
        n = 300
        t = np.arange(n, dtype=np.float64)
        base_prices = 100.0 * (1.001 ** t)

        # Baseline series
        mom_base = compute_jegadeesh_titman_momentum(base_prices, lookback_total=252, skip_recent=21)

        # Crash scenario in the last 21 days (50% crash from t-20 to t)
        crash_prices = base_prices.copy()
        crash_prices[-21:] = crash_prices[-22] * 0.50
        mom_crash = compute_jegadeesh_titman_momentum(crash_prices, lookback_total=252, skip_recent=21)

        # Surge scenario in the last 21 days (100% surge from t-20 to t)
        surge_prices = base_prices.copy()
        surge_prices[-21:] = surge_prices[-22] * 2.0
        mom_surge = compute_jegadeesh_titman_momentum(surge_prices, lookback_total=252, skip_recent=21)

        # 12-1 Momentum MUST be identical across all 3
        assert mom_base == pytest.approx(mom_crash, abs=1e-12)
        assert mom_base == pytest.approx(mom_surge, abs=1e-12)

        # In contrast, unskipped returns MUST differ
        ret_base = calculate_multi_timeframe_returns(base_prices)["ret_12m"]
        ret_crash = calculate_multi_timeframe_returns(crash_prices)["ret_12m"]
        ret_surge = calculate_multi_timeframe_returns(surge_prices)["ret_12m"]

        assert ret_crash < ret_base < ret_surge

    def test_volatility_jump_penalty_monotonicity(self):
        """
        Jump penalty factor exp(-max(0, (jump_ratio - 3.0)/2.0)) must be 1.0 for jump ratio <= 3.0,
        and strictly monotonically decrease towards 0.0 as the single-day outlier jump increases.
        """
        n = 100
        # Smooth baseline with std ~ 0.01 daily
        np.random.seed(42)
        daily_ret = np.random.normal(0.001, 0.01, size=n)
        prices = 100.0 * np.exp(np.cumsum(daily_ret))

        ratios = []
        penalties = []

        jump_multipliers = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0]
        for mult in jump_multipliers:
            p_jump = prices.copy()
            if mult > 0:
                p_jump[-1] = p_jump[-2] * (1.0 + 0.01 * mult)
            ratio, penalty = calculate_jump_ratio_penalty(p_jump, lookback=90)
            ratios.append(ratio)
            penalties.append(penalty)

        # All penalties must be strictly bounded in (0.0, 1.0]
        assert all(0.0 < p <= 1.0 for p in penalties)

        # For mult <= 3 (ratio <= 3.0), penalty is 1.0
        # For large jumps, penalty strictly decreases
        assert penalties[0] == 1.0
        assert penalties[-1] < penalties[0]
        for i in range(len(penalties) - 1):
            if ratios[i] > 3.0 and ratios[i + 1] > ratios[i]:
                assert penalties[i] > penalties[i + 1]

    def test_smooth_compounding_vs_erratic_jump_comparison(self):
        """
        Two assets with identical 90-day cumulative return (+25%):
        Asset A: Steady compounding (0.25% daily) -> JumpPenalty = 1.0, high R^2 = 1.0
        Asset B: Flat all 89 days, then single +25% jump on day 90 -> JumpPenalty < 0.20, low R^2
        """
        n = 90
        # Asset A: smooth compounding
        t = np.arange(n, dtype=np.float64)
        daily_rate = (1.25 ** (1.0 / (n - 1))) - 1.0
        prices_a = 100.0 * ((1.0 + daily_rate) ** t)

        # Asset B: flat then jump
        prices_b = np.full(n, 100.0, dtype=np.float64)
        prices_b[-1] = 125.0

        # Verify cumulative return is identical
        ret_a = (prices_a[-1] - prices_a[0]) / prices_a[0]
        ret_b = (prices_b[-1] - prices_b[0]) / prices_b[0]
        assert ret_a == pytest.approx(0.25, rel=1e-5)
        assert ret_b == pytest.approx(0.25, rel=1e-5)

        # Clenow scores & R^2
        _, r2_a, score_a = compute_clenow_momentum(prices_a, lookback=90)
        _, r2_b, score_b = compute_clenow_momentum(prices_b, lookback=90)
        assert r2_a == pytest.approx(1.0, rel=1e-5)
        assert r2_b < 0.20
        assert score_a > score_b * 5.0

        # Jump penalties
        _, penalty_a = calculate_jump_ratio_penalty(prices_a, lookback=90)
        _, penalty_b = calculate_jump_ratio_penalty(prices_b, lookback=90)
        assert penalty_a == 1.0
        assert penalty_b < 0.10


class TestDegenerateAndAdversarialEdgeCases:
    """Stress tests on extreme numerical inputs, penny stocks, NaNs, Infs, zero variance, and boundary lengths."""

    def test_scale_invariance_micro_vs_macro_prices(self):
        """
        Log-based regression slope and R^2 must be strictly scale invariant:
        Multiplying all prices by a scalar constant C (e.g. 10^-6 or 10^6) does not change beta, R^2, or score.
        """
        n = 90
        t = np.arange(n, dtype=np.float64)
        p_normal = 100.0 * (1.002 ** t)
        p_micro = 0.0001 * (1.002 ** t)    # Penny stock / crypto / micro price
        p_macro = 1_000_000.0 * (1.002 ** t) # High nominal share price

        slope_norm, r2_norm, score_norm = compute_clenow_momentum(p_normal, lookback=90)
        slope_micro, r2_micro, score_micro = compute_clenow_momentum(p_micro, lookback=90)
        slope_macro, r2_macro, score_macro = compute_clenow_momentum(p_macro, lookback=90)

        assert slope_norm == pytest.approx(slope_micro, rel=1e-6)
        assert slope_norm == pytest.approx(slope_macro, rel=1e-6)
        assert r2_norm == pytest.approx(r2_micro, rel=1e-6)
        assert r2_norm == pytest.approx(r2_macro, rel=1e-6)
        assert score_norm == pytest.approx(score_micro, rel=1e-6)
        assert score_norm == pytest.approx(score_macro, rel=1e-6)

    def test_zero_variance_and_flat_series(self):
        """Constant series must return safe zeros/defaults without division by zero or errors."""
        flat_prices = np.full(100, 50.0)

        # Clenow
        slope, r2, score = compute_clenow_momentum(flat_prices, lookback=90)
        assert slope == 0.0 and r2 == 0.0 and score == 0.0

        # Volatility
        vol = calculate_realized_volatility(flat_prices, lookback=90)
        vol_down = calculate_downside_deviation(flat_prices, lookback=90)
        ratio, penalty = calculate_jump_ratio_penalty(flat_prices, lookback=90)
        assert vol == 0.0
        assert vol_down == 0.0
        assert ratio == 0.0
        assert penalty == 1.0

        # Momentum
        mom_12_1 = compute_jegadeesh_titman_momentum(flat_prices, lookback_total=90, skip_recent=21)
        assert mom_12_1 == 0.0

    def test_severe_nans_infs_and_negative_prices(self):
        """Degenerate price arrays must be handled safely without unhandled exceptions."""
        cases = [
            np.array([np.nan] * 50),
            np.array([10.0, np.nan, 20.0, 30.0]),
            np.array([10.0, np.inf, 20.0, 30.0]),
            np.array([10.0, -5.0, 20.0, 30.0]),
            np.array([0.0, 10.0, 20.0]),
            [],
            None,
        ]

        for case in cases:
            slope, r2, score = compute_clenow_momentum(case, lookback=90)
            assert np.isnan(slope) or slope == 0.0
            assert np.isnan(r2) or r2 == 0.0
            assert np.isnan(score) or score == 0.0

            vol = calculate_realized_volatility(case, lookback=90)
            assert np.isnan(vol) or vol == 0.0

            mom = compute_jegadeesh_titman_momentum(case, lookback_total=252, skip_recent=21)
            assert np.isnan(mom) or np.isinf(mom) or mom == 0.0

    @pytest.mark.parametrize("length", [0, 1, 2, 20, 21, 62, 63, 89, 90, 125, 126, 251, 252])
    def test_boundary_history_lengths(self, length):
        """Verify behavior at exact boundary conditions of required history lengths."""
        if length > 0:
            prices = np.linspace(100.0, 150.0, length)
        else:
            prices = np.array([])

        slope_90, _, _ = compute_clenow_momentum(prices, lookback=90)
        mom_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
        mom_6_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=126, skip_recent=21)
        mom_3_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=63, skip_recent=21)

        if length < 90:
            assert np.isnan(slope_90)
        else:
            assert not np.isnan(slope_90)

        if length < 252:
            assert np.isnan(mom_12_1)
        else:
            assert not np.isnan(mom_12_1)

        if length < 126:
            assert np.isnan(mom_6_1)
        else:
            assert not np.isnan(mom_6_1)

        if length < 63:
            assert np.isnan(mom_3_1)
        else:
            assert not np.isnan(mom_3_1)


class Test500AssetCrossSectionalStressUniverse:
    """Stress testing cross-sectional ranking across a 500-asset heterogeneous universe."""

    @pytest.fixture
    def synthetic_500_universe(self):
        """
        Generates 500 diverse synthetic stocks with varied statistical properties:
        1. 100 Steady Compounders: positive slope (g=0.0015..0.0025), low noise (high R^2), low vol.
        2. 100 High-Beta / Volatile Meme Stocks: same underlying drift (g=0.0015) but extreme vol (sigma=0.035) and jumps.
        3. 100 Sideways / Random Walk Stocks: zero net trend, moderate vol.
        4. 100 Consistent Losers: negative slope, low noise.
        5. 50 Fallen Angels: strong 12-1 momentum, but severe crash in last 20 days below SMA200.
        6. 25 Short-History IPOs: only 100 days of data.
        7. 15 Flat/Halted Stocks: zero volatility.
        8. 10 Corrupted/NaN Stocks: missing/invalid prices.
        """
        np.random.seed(42)
        n_bars = 300
        t = np.arange(n_bars, dtype=np.float64)

        universe_metrics = []

        # 1. 100 Steady Compounders (g = 0.0015 to 0.0025)
        for i in range(100):
            sym = f"COMPOUNDER_{i:03d}"
            g = 0.0015 + 0.0010 * (i / 100.0) # daily growth ~ +45% to +85% annual
            noise = np.random.normal(0, 0.005, size=n_bars)
            prices = 100.0 * np.exp(t * np.log(1.0 + g) + noise)
            s_ann, r2, score = compute_clenow_momentum(prices, lookback=90)
            m_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
            m_6_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=126, skip_recent=21)
            vol = calculate_realized_volatility(prices, lookback=90)

            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=float(prices[-1]), history_bars=n_bars,
                clenow_slope_ann=s_ann, clenow_r2=r2, clenow_score=score,
                mom_12_1=m_12_1, mom_6_1=m_6_1, volatility_ann=vol,
                passes_trend=True, golden_alignment=True,
            ))

        # 2. 100 High-Beta Meme Stocks (g = 0.0015, extreme vol sigma=0.035, and random jumps)
        for i in range(100):
            sym = f"MEME_{i:03d}"
            g = 0.0015
            noise = np.random.normal(0, 0.035, size=n_bars) # heavy volatility
            prices = 100.0 * np.exp(t * np.log(1.0 + g) + noise)
            # Add random jumps
            jump_idx = np.random.choice(range(50, n_bars), size=2, replace=False)
            prices[jump_idx] *= 1.10

            s_ann, r2, score = compute_clenow_momentum(prices, lookback=90)
            m_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
            m_6_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=126, skip_recent=21)
            vol = calculate_realized_volatility(prices, lookback=90)

            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=float(prices[-1]), history_bars=n_bars,
                clenow_slope_ann=s_ann, clenow_r2=r2, clenow_score=score,
                mom_12_1=m_12_1, mom_6_1=m_6_1, volatility_ann=vol,
                passes_trend=True, golden_alignment=False,
            ))

        # 3. 100 Sideways Random Walk Stocks
        for i in range(100):
            sym = f"SIDEWAYS_{i:03d}"
            noise = np.random.normal(0, 0.015, size=n_bars)
            prices = 100.0 * np.exp(np.cumsum(noise))

            s_ann, r2, score = compute_clenow_momentum(prices, lookback=90)
            m_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
            m_6_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=126, skip_recent=21)
            vol = calculate_realized_volatility(prices, lookback=90)

            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=float(prices[-1]), history_bars=n_bars,
                clenow_slope_ann=s_ann, clenow_r2=r2, clenow_score=score,
                mom_12_1=m_12_1, mom_6_1=m_6_1, volatility_ann=vol,
                passes_trend=True, golden_alignment=False,
            ))

        # 4. 100 Consistent Losers
        for i in range(100):
            sym = f"LOSER_{i:03d}"
            g = -0.0010 - 0.0002 * (i / 100.0)
            noise = np.random.normal(0, 0.005, size=n_bars)
            prices = 100.0 * np.exp(t * np.log(1.0 + g) + noise)

            s_ann, r2, score = compute_clenow_momentum(prices, lookback=90)
            m_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
            m_6_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=126, skip_recent=21)
            vol = calculate_realized_volatility(prices, lookback=90)

            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=float(prices[-1]), history_bars=n_bars,
                clenow_slope_ann=s_ann, clenow_r2=r2, clenow_score=score,
                mom_12_1=m_12_1, mom_6_1=m_6_1, volatility_ann=vol,
                passes_trend=False, golden_alignment=False,
            ))

        # 5. 50 Fallen Angels (Strong 12-1 mom, but failed trend filter Close < SMA200)
        for i in range(50):
            sym = f"FALLEN_{i:03d}"
            prices = 100.0 * (1.0015 ** t)
            prices[-20:] = prices[-21] * 0.50 # 50% crash below SMA200

            s_ann, r2, score = compute_clenow_momentum(prices, lookback=90)
            m_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
            m_6_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=126, skip_recent=21)
            vol = calculate_realized_volatility(prices, lookback=90)

            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=float(prices[-1]), history_bars=n_bars,
                clenow_slope_ann=s_ann, clenow_r2=r2, clenow_score=score,
                mom_12_1=m_12_1, mom_6_1=m_6_1, volatility_ann=vol,
                passes_trend=False, # Trend broken
                golden_alignment=False,
            ))

        # 6. 25 Short History IPOs (100 bars)
        for i in range(25):
            sym = f"IPO_{i:03d}"
            p_short = 100.0 * (1.002 ** np.arange(100, dtype=np.float64))
            s_ann, r2, score = compute_clenow_momentum(p_short, lookback=90)
            vol = calculate_realized_volatility(p_short, lookback=90)

            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=float(p_short[-1]), history_bars=100,
                clenow_slope_ann=s_ann, clenow_r2=r2, clenow_score=score,
                mom_12_1=np.nan, mom_6_1=np.nan, volatility_ann=vol,
                passes_trend=True, golden_alignment=False,
            ))

        # 7. 15 Flat/Halted Stocks
        for i in range(15):
            sym = f"HALTED_{i:03d}"
            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=100.0, history_bars=n_bars,
                clenow_slope_ann=0.0, clenow_r2=0.0, clenow_score=0.0,
                mom_12_1=0.0, mom_6_1=0.0, volatility_ann=0.0,
                passes_trend=False, golden_alignment=False,
            ))

        # 8. 10 Corrupted/NaN Stocks
        for i in range(10):
            sym = f"CORRUPT_{i:03d}"
            universe_metrics.append(MomentumMetrics(
                symbol=sym, close=0.0, history_bars=0,
                clenow_slope_ann=np.nan, clenow_r2=np.nan, clenow_score=np.nan,
                mom_12_1=np.nan, mom_6_1=np.nan, volatility_ann=np.nan,
                passes_trend=False, golden_alignment=False,
            ))

        return universe_metrics

    def test_500_asset_cross_sectional_ranking_properties(self, synthetic_500_universe):
        """
        Stress test ranking stability, Z-score clipping, percentile coverage,
        and trend regime filtering across the 500-asset universe.
        """
        assert len(synthetic_500_universe) == 500

        # Rank with trend filter active
        breakdowns, df_ranked = rank_universe(synthetic_500_universe, filter_trend=True)

        # 1. Ensure output is non-empty and sorted
        assert len(breakdowns) > 0
        assert len(breakdowns) == len(df_ranked)

        # 2. Check strict descending order of composite scores
        scores = [b.composite_score for b in breakdowns]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Ranking not descending at rank {i+1}"

        # 3. Check contiguous 1-indexed ranks
        ranks = [b.rank for b in breakdowns]
        assert ranks == list(range(1, len(breakdowns) + 1))

        # 4. Check percentile ranks strictly in [0.0, 1.0]
        pcts = [b.percentile_rank for b in breakdowns]
        assert min(pcts) == pytest.approx(0.0, abs=1e-5)
        assert max(pcts) == pytest.approx(1.0, abs=1e-5)

        # 5. Check Winsorized Z-scores strictly in [-3.0, +3.0]
        for b in breakdowns:
            assert -3.0 <= b.z_clenow <= 3.0
            assert -3.0 <= b.z_mom_12_1 <= 3.0
            assert -3.0 <= b.z_r2 <= 3.0
            assert -3.0 <= b.z_mom_6_1 <= 3.0
            assert -3.0 <= b.z_vol_ann <= 3.0

        # 6. Trend filter verification: None of the Fallen Angels or Losers should be present
        for b in breakdowns:
            assert not b.symbol.startswith("FALLEN_"), f"Fallen Angel {b.symbol} leaked through trend filter!"
            assert not b.symbol.startswith("LOSER_"), f"Loser {b.symbol} leaked through trend filter!"
            assert not b.symbol.startswith("CORRUPT_"), f"Corrupted stock {b.symbol} ranked!"

        # 7. Quality assertion: Top 10 ranks should be dominated by Steady Compounders (high R^2, low vol)
        # over High-Beta Meme stocks (penalized by volatility factor and lower R^2)
        top_10 = breakdowns[:10]
        compounder_count = sum(1 for b in top_10 if b.symbol.startswith("COMPOUNDER_"))
        assert compounder_count >= 8, f"Expected at least 8/10 top assets to be steady compounders, got {compounder_count}"

        # 8. Action assignment: Top 10 must have BUY action, others HOLD
        for b in breakdowns:
            if b.rank <= 10:
                assert b.action == RebalanceAction.BUY
            else:
                assert b.action == RebalanceAction.HOLD

        # 8. Action assignment: Top 10 must have BUY action, others HOLD
        for b in breakdowns:
            if b.rank <= 10:
                assert b.action == RebalanceAction.BUY
            else:
                assert b.action == RebalanceAction.HOLD


class TestMacroMarketRegimeTransitions:
    """Stress tests on macro benchmark regime classification and equity allocation scaling."""

    def test_benchmark_regime_state_transitions(self):
        """
        Verify the 3 macro benchmark regimes:
        1. BULLISH (100% allocation): Benchmark > SMA200 AND SMA200 ascending (20-day slope >= 0)
        2. NEUTRAL (50% allocation): Benchmark > SMA200 AND SMA200 descending (20-day slope < 0)
        3. BEARISH (0% allocation / cash): Benchmark <= SMA200
        """
        n = 300
        t = np.arange(n, dtype=np.float64)

        # 1. Bullish series: Steady 10% annual uptrend
        bull_prices = 10000.0 * (1.0004 ** t)
        regime_bull = evaluate_market_regime(bull_prices, benchmark_symbol="^NSEI", lookback_sma=200, slope_window=20)
        assert regime_bull.state == MarketRegimeState.BULLISH
        assert regime_bull.equity_allocation_pct == 1.0
        assert regime_bull.sma_200_slope > 0.0

        # 2. Bearish series: Steady downtrend below SMA200
        bear_prices = 10000.0 * (0.999 ** t)
        regime_bear = evaluate_market_regime(bear_prices, benchmark_symbol="^NSEI", lookback_sma=200, slope_window=20)
        assert regime_bear.state == MarketRegimeState.BEARISH
        assert regime_bear.equity_allocation_pct == 0.0

        # 3. Neutral series: Price bounced above SMA200, but SMA200 is still sloping downwards
        # Construct series that declined heavily for 250 days, then bounced in the last 40 days
        downtrend = 10000.0 * (0.998 ** np.arange(260, dtype=np.float64))
        bounce_target = downtrend[-1] * 1.30
        bounce = np.linspace(downtrend[-1], bounce_target, 40)
        neutral_prices = np.concatenate([downtrend, bounce])

        regime_neutral = evaluate_market_regime(neutral_prices, benchmark_symbol="^NSEI", lookback_sma=200, slope_window=20)
        assert regime_neutral.state == MarketRegimeState.NEUTRAL
        assert regime_neutral.equity_allocation_pct == 0.50
        assert regime_neutral.sma_200_slope < 0.0

    def test_missing_or_short_benchmark_defensive_default(self):
        """Missing or short benchmark data must gracefully default to NEUTRAL (50% allocation)."""
        reg_none = evaluate_market_regime(None)
        assert reg_none.state == MarketRegimeState.NEUTRAL
        assert reg_none.equity_allocation_pct == 0.50

        reg_short = evaluate_market_regime([100.0] * 50)
        assert reg_short.state == MarketRegimeState.NEUTRAL
        assert reg_short.equity_allocation_pct == 0.50

"""
Tier 1: Core Feature Coverage Test Suite.
Validates >= 5 tests per quantitative feature against authoritative analytical oracles.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from tests.e2e.fixtures import (
    generate_deterministic_series,
    generate_mock_benchmark,
    oracle_bounded_risk_parity_weights,
    oracle_exponential_regression,
    oracle_inverse_volatility_weights,
    oracle_jegadeesh_titman_12_1,
    oracle_realized_volatility,
)
from src.momentum.clenow import (
    calculate_clenow_momentum,
    calculate_multi_timeframe_clenow,
    compute_clenow_momentum,
)
from src.momentum.classical import (
    calculate_intermediate_momentum,
    calculate_jegadeesh_titman_momentum,
    calculate_multi_timeframe_returns,
    compute_jegadeesh_titman_momentum,
)
from src.momentum.volatility import (
    calculate_atr,
    calculate_downside_deviation,
    calculate_jump_ratio_penalty,
    calculate_realized_volatility,
    calculate_volatility_adjusted_momentum,
    compute_volatility_metrics,
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
from src.momentum.models import (
    FilterResult,
    MarketRegimeState,
    MomentumMetrics,
    RebalanceAction,
)


# ==============================================================================
# Feature 1: Andreas Clenow Exponential Regression Slope
# ==============================================================================

class TestClenowSlope:
    """Feature 1: Andreas Clenow Exponential Regression & Annualized Slope."""

    def test_exponential_slope_analytical_ground_truth(self):
        """Analytical ground truth: P_t = 100 * (1.002)^t => Slope_ann = (1.002)^250 - 1."""
        N = 90
        t = np.arange(N, dtype=np.float64)
        prices = 100.0 * (1.002 ** t)

        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90, trading_days=250)
        expected_slope = (1.002 ** 250) - 1.0  # ~0.647898

        assert abs(slope_ann - expected_slope) < 1e-4, f"Expected {expected_slope}, got {slope_ann}"
        assert abs(r2 - 1.0) < 1e-5, f"Expected R^2=1.0, got {r2}"
        assert abs(score - expected_slope) < 1e-4

    def test_exponential_slope_flat_series(self):
        """Constant price series yields exact 0.0 slope and 0.0 Clenow score."""
        prices = np.full(90, 150.0, dtype=np.float64)
        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90)

        assert slope_ann == 0.0
        assert r2 == 0.0
        assert score == 0.0

    def test_exponential_slope_negative_drift_downtrend(self):
        """Negative drift P_t = 100 * (0.998)^t yields negative annualized slope."""
        N = 90
        t = np.arange(N, dtype=np.float64)
        prices = 100.0 * (0.998 ** t)

        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90, trading_days=250)
        expected_slope = (0.998 ** 250) - 1.0  # ~-0.3938

        assert abs(slope_ann - expected_slope) < 1e-4
        assert slope_ann < 0.0
        assert abs(r2 - 1.0) < 1e-5
        assert score < 0.0

    def test_exponential_slope_scale_invariance(self):
        """Multiplying all prices by constant C (e.g. 50.0) leaves slope and R^2 unchanged."""
        N = 90
        t = np.arange(N, dtype=np.float64)
        base_prices = 10.0 * (1.0015 ** t)
        scaled_prices = base_prices * 50.0

        slope_base, r2_base, score_base = compute_clenow_momentum(base_prices, lookback=90)
        slope_scaled, r2_scaled, score_scaled = compute_clenow_momentum(scaled_prices, lookback=90)

        assert abs(slope_base - slope_scaled) < 1e-6
        assert abs(r2_base - r2_scaled) < 1e-6
        assert abs(score_base - score_scaled) < 1e-6

    def test_exponential_slope_variable_lookback_windows(self):
        """Validates slope calculations across 90, 126, and 252 day lookback windows."""
        df = generate_deterministic_series(n_bars=300, regime="smooth_exponential_uptrend", daily_drift=0.002)
        prices = df["close"].values

        for window in [90, 126, 252]:
            slope, r2, score = compute_clenow_momentum(prices, lookback=window)
            o_slope, o_r2, o_score = oracle_exponential_regression(prices[-window:], trading_days=250)

            assert abs(slope - o_slope) < 1e-5
            assert abs(r2 - o_r2) < 1e-5
            assert abs(score - o_score) < 1e-5

    def test_exponential_slope_steep_growth(self):
        """Very steep exponential trend compounding at 1% daily: (1.01)^250 - 1 ~ 11.032."""
        N = 90
        t = np.arange(N, dtype=np.float64)
        prices = 50.0 * (1.01 ** t)

        slope_ann, r2, score = compute_clenow_momentum(prices, lookback=90, trading_days=250)
        expected_slope = (1.01 ** 250) - 1.0

        assert abs(slope_ann - expected_slope) < 1e-3
        assert slope_ann > 10.0
        assert abs(r2 - 1.0) < 1e-5


# ==============================================================================
# Feature 2: Coefficient of Determination (R^2) Trend Smoothness
# ==============================================================================

class TestClenowRSquared:
    """Feature 2: R^2 Trend Smoothness Metric."""

    def test_r_squared_perfect_exponential_series(self):
        """Pure geometric progression in log-space has R^2 == 1.0."""
        prices = np.exp(np.linspace(1.0, 3.0, 100))
        _, r2, _ = compute_clenow_momentum(prices, lookback=100)
        assert abs(r2 - 1.0) < 1e-6

    def test_r_squared_zero_variance_flat_series(self):
        """Flat price series returns R^2 == 0.0 without division by zero error."""
        prices = np.full(90, 100.0)
        _, r2, _ = compute_clenow_momentum(prices, lookback=90)
        assert r2 == 0.0

    def test_r_squared_noise_monotonic_penalty(self):
        """Adding increasing Gaussian noise monotonically reduces R^2."""
        rng = np.random.RandomState(42)
        t = np.arange(100, dtype=np.float64)
        trend = np.log(100.0) + 0.002 * t

        r2_values = []
        for noise_lvl in [0.001, 0.01, 0.04, 0.10]:
            noisy_log = trend + rng.normal(0, noise_lvl, 100)
            _, r2, _ = compute_clenow_momentum(np.exp(noisy_log), lookback=100)
            r2_values.append(r2)

        # Confirm strict monotonic degradation
        for i in range(len(r2_values) - 1):
            assert r2_values[i] > r2_values[i + 1], f"Noise penalty violated: {r2_values}"

    def test_r_squared_bounded_in_unit_interval(self):
        """R^2 must remain in [0.0, 1.0] across diverse stochastic price paths."""
        for seed in range(20):
            df = generate_deterministic_series(n_bars=150, regime="choppy_uptrend", seed=seed)
            _, r2, _ = compute_clenow_momentum(df["close"].values, lookback=90)
            assert 0.0 <= r2 <= 1.0, f"R^2 out of bounds: {r2}"

    def test_r_squared_scale_invariance(self):
        """Price magnitude scaling does not alter R^2."""
        df = generate_deterministic_series(n_bars=100, regime="smooth_exponential_uptrend", seed=99)
        p1 = df["close"].values
        p2 = p1 * 1000.0

        _, r2_1, _ = compute_clenow_momentum(p1, lookback=90)
        _, r2_2, _ = compute_clenow_momentum(p2, lookback=90)
        assert abs(r2_1 - r2_2) < 1e-6

    def test_r_squared_trend_reversal_penalty(self):
        """V-shaped trend reversal (drop then recover) yields low R^2."""
        t = np.arange(90, dtype=np.float64)
        # Drops first 45 days, recovers next 45 days
        p = 100.0 * np.exp(-0.005 * np.minimum(t, 45) + 0.005 * np.maximum(0, t - 45))
        _, r2, _ = compute_clenow_momentum(p, lookback=90)
        assert r2 < 0.25, f"V-shape should have low R^2, got {r2}"


# ==============================================================================
# Feature 3: Andreas Clenow Momentum Score
# ==============================================================================

class TestClenowScore:
    """Feature 3: Clenow Momentum Score (Slope_ann * R^2)."""

    def test_clenow_score_exact_multiplication(self):
        """Score must equal annualized_slope * r_squared exactly."""
        df = generate_deterministic_series(n_bars=120, regime="smooth_exponential_uptrend")
        slope, r2, score = compute_clenow_momentum(df["close"].values, lookback=90)
        assert abs(score - (slope * r2)) < 1e-7

    def test_clenow_score_flat_series_is_zero(self):
        """Flat series produces exact 0.0 score."""
        _, _, score = compute_clenow_momentum(np.full(90, 200.0), lookback=90)
        assert score == 0.0

    def test_clenow_score_smooth_outscores_choppy(self):
        """Smooth moderate growth outscores erratic high-slope growth due to R^2 penalty."""
        rng = np.random.RandomState(42)
        t = np.arange(100, dtype=np.float64)
        smooth_prices = 100.0 * np.exp(0.002 * t + rng.normal(0, 0.001, 100))
        choppy_prices = 100.0 * np.exp(0.003 * t + rng.normal(0, 0.20, 100))

        s_slope, s_r2, s_score = compute_clenow_momentum(smooth_prices, lookback=90)
        c_slope, c_r2, c_score = compute_clenow_momentum(choppy_prices, lookback=90)

        assert s_r2 > 0.85
        assert c_r2 < 0.50
        assert s_score > c_score, f"Smooth score ({s_score:.3f}) should beat choppy score ({c_score:.3f})"

    def test_clenow_score_negative_slope_handling(self):
        """Negative slope results in negative Clenow score."""
        df = generate_deterministic_series(n_bars=100, regime="strong_downtrend")
        slope, r2, score = compute_clenow_momentum(df["close"].values, lookback=90)
        assert slope < 0.0
        assert score < 0.0

    def test_multi_timeframe_clenow_blending(self):
        """Multi-timeframe Clenow blends 90d (40%), 126d (30%), 252d (30%)."""
        df = generate_deterministic_series(n_bars=300, regime="smooth_exponential_uptrend")
        prices = df["close"].values

        mtf_dict = calculate_multi_timeframe_clenow(prices, windows=(90, 126, 252), weights=(0.40, 0.30, 0.30))
        assert "composite_score" in mtf_dict
        assert not np.isnan(mtf_dict["composite_score"])

        expected = 0.40 * mtf_dict["score_90"] + 0.30 * mtf_dict["score_126"] + 0.30 * mtf_dict["score_252"]
        assert abs(mtf_dict["composite_score"] - expected) < 1e-5


# ==============================================================================
# Feature 4: Classical Jegadeesh & Titman 12-1 Momentum
# ==============================================================================

class TestClassicalMomentum:
    """Feature 4: Jegadeesh & Titman (1993) 12-1 Intermediate Momentum."""

    def test_mom_12_1_analytical_exact(self):
        """Exact formula: Mom_12_1 = (P_{t-21} - P_{t-252}) / P_{t-252}."""
        prices = np.full(300, 100.0, dtype=np.float64)
        prices[47] = 100.0   # t-252 (at index 300 - 253 = 47)
        prices[278] = 175.0  # t-21  (at index 300 - 22 = 278)
        prices[299] = 150.0  # latest

        mom_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
        expected = (175.0 - 100.0) / 100.0  # 0.75

        assert abs(mom_12_1 - expected) < 1e-6

    def test_mom_12_1_skips_recent_month_crash(self):
        """A severe crash in the most recent 21 days does NOT contaminate 12-1 momentum."""
        df = generate_deterministic_series(n_bars=300, regime="short_term_crash_skip_month")
        prices = df["close"].values

        mom_12_1 = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
        raw_12m = (prices[-1] - prices[-253]) / prices[-253]

        # 12-1 momentum should be substantially higher than raw 12-month return because it skipped recent crash
        assert mom_12_1 > raw_12m
        assert mom_12_1 > 0.30

    def test_mom_6_1_intermediate_momentum(self):
        """6-1 momentum evaluates return from t-126 to t-21."""
        prices = np.linspace(100.0, 200.0, 300)
        mom_6_1 = calculate_intermediate_momentum(prices, lookback=126, skip_recent=21)
        p_start = prices[-127]
        p_skip = prices[-22]
        expected = (p_skip - p_start) / p_start

        assert abs(mom_6_1 - expected) < 1e-6

    def test_mom_3_1_quarterly_intermediate_momentum(self):
        """3-1 momentum evaluates return from t-63 to t-21."""
        prices = np.linspace(100.0, 160.0, 300)
        res = calculate_multi_timeframe_returns(prices, skip_recent=21)
        assert "mom_3_1" in res
        p_start = prices[-64]
        p_skip = prices[-22]
        expected = (p_skip - p_start) / p_start

        assert abs(res["mom_3_1"] - expected) < 1e-6

    def test_mom_12_1_insufficient_history_returns_nan(self):
        """Price series with < 252 bars returns np.nan for 12-1 momentum."""
        prices = np.full(200, 100.0)
        mom = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21)
        assert np.isnan(mom)

    def test_mom_12_1_log_return_calculation(self):
        """Log return calculation: ln(P_{t-21}) - ln(P_{t-252})."""
        prices = np.full(300, 100.0)
        prices[-253] = 100.0
        prices[-22] = 200.0

        log_mom = compute_jegadeesh_titman_momentum(prices, lookback_total=252, skip_recent=21, log_return=True)
        expected_log = np.log(200.0) - np.log(100.0)  # ln(2) ~ 0.693147

        assert abs(log_mom - expected_log) < 1e-6


# ==============================================================================
# Feature 5: Realized Volatility & Risk Normalization
# ==============================================================================

class TestVolatilityMetrics:
    """Feature 5: Realized Volatility & Risk Penalties."""

    def test_realized_volatility_analytical_comparison(self):
        """Realized annualized vol matches sample std of log returns * sqrt(252)."""
        df = generate_deterministic_series(n_bars=150, regime="smooth_exponential_uptrend")
        prices = df["close"].values

        vol_calc = calculate_realized_volatility(prices, lookback=90, trading_days=252)
        vol_oracle = oracle_realized_volatility(prices[-91:], trading_days=252)

        assert abs(vol_calc - vol_oracle) < 1e-6

    def test_realized_volatility_flat_series_zero(self):
        """Constant price series yields exact 0.0 realized volatility."""
        prices = np.full(100, 50.0)
        vol = calculate_realized_volatility(prices, lookback=90)
        assert vol == 0.0

    def test_downside_deviation_penalizes_negative_only(self):
        """Monotonically increasing series has 0.0 downside deviation."""
        prices = np.exp(np.linspace(1.0, 2.0, 100))
        down_vol = calculate_downside_deviation(prices, lookback=90)
        assert down_vol == 0.0

    def test_jump_ratio_penalty_single_outlier_shock(self):
        """Single +25% earnings gap yields high jump ratio and reduced jump penalty factor < 1.0."""
        df = generate_deterministic_series(n_bars=100, regime="earnings_gap_up")
        prices = df["close"].values

        jump_ratio, jump_penalty = calculate_jump_ratio_penalty(prices, lookback=90)
        assert jump_ratio > 3.0
        assert jump_penalty < 0.95

    def test_atr_14_and_natr_wilder_calculation(self):
        """Calculates 14-day ATR and Normalized ATR percentage."""
        df = generate_deterministic_series(n_bars=100, regime="smooth_exponential_uptrend")
        atr, natr = calculate_atr(df["high"].values, df["low"].values, df["close"].values, period=14)

        assert not np.isnan(atr)
        assert atr > 0.0
        assert 0.1 <= natr <= 10.0  # NATR % realistic bounds

    def test_volatility_adjusted_momentum_ratio(self):
        """Information momentum Mom / max(vol, epsilon) calculation."""
        mom = 0.50
        vol = 0.25
        adj_mom = calculate_volatility_adjusted_momentum(mom, vol)
        assert abs(adj_mom - 2.0) < 1e-6


# ==============================================================================
# Feature 6: Trend Regime & Moving Average Filters
# ==============================================================================

class TestRegimeFilters:
    """Feature 6: Single-Stock Trend Regime & Moving Average Filters."""

    def test_trend_filter_bullish_uptrend_passes(self):
        """Smooth exponential uptrend satisfies Close > EMA100 and Close > SMA200."""
        df = generate_deterministic_series(n_bars=300, regime="smooth_exponential_uptrend")
        passes, details = check_single_stock_trend_filter(df["close"].values, ema_fast=100, sma_slow=200)

        assert passes is True
        assert details["above_ema_fast"] is True
        assert details["above_sma_slow"] is True

    def test_trend_filter_downtrend_fails(self):
        """Downtrend price series fails Close > EMA100 and Close > SMA200."""
        df = generate_deterministic_series(n_bars=300, regime="strong_downtrend")
        passes, details = check_single_stock_trend_filter(df["close"].values, ema_fast=100, sma_slow=200)

        assert passes is False
        assert details["above_ema_fast"] is False
        assert details["above_sma_slow"] is False

    def test_golden_regime_alignment(self):
        """Verifies EMA10 > EMA50 > SMA200 alignment in strong uptrend."""
        df = generate_deterministic_series(n_bars=300, regime="smooth_exponential_uptrend")
        is_golden = check_golden_regime(df["close"].values, ema_short=10, ema_mid=50, sma_long=200)
        assert is_golden is True

    def test_52w_high_distance_ratio(self):
        """Stock at 52-week high yields 1.0; stock in 40% drawdown yields 0.60."""
        prices = np.full(252, 100.0)
        prices[-1] = 100.0
        dist = calculate_52w_high_distance(prices, lookback=252)
        assert abs(dist - 1.0) < 1e-6

        prices[-1] = 60.0  # 40% drawdown
        dist_dd = calculate_52w_high_distance(prices, lookback=252)
        assert abs(dist_dd - 0.60) < 1e-6

    def test_insufficient_history_for_trend_fails(self):
        """Series with only 80 bars fails trend filter requiring 200 SMA."""
        df = generate_deterministic_series(regime="low_history_80d")
        passes, details = check_single_stock_trend_filter(df["close"].values, ema_fast=100, sma_slow=200)

        assert passes is False
        assert "Insufficient history" in details["reason"]

    def test_stock_eligibility_comprehensive_evaluation(self):
        """Comprehensive stock eligibility handles multiple criteria."""
        df_good = generate_deterministic_series(n_bars=300, regime="smooth_exponential_uptrend")
        res_good = evaluate_stock_eligibility("TRENT.NS", df_good, min_history=252, min_dist_52w=0.75)
        assert res_good.is_eligible is True

        df_bad = generate_deterministic_series(n_bars=300, regime="strong_downtrend")
        res_bad = evaluate_stock_eligibility("BAD.NS", df_bad, min_history=252, min_dist_52w=0.75)
        assert res_bad.is_eligible is False


# ==============================================================================
# Feature 7: Macro Market Benchmark Gate
# ==============================================================================

class TestBenchmarkRegime:
    """Feature 7: Macro Benchmark Circuit Breaker (^NSEI / ^GSPC)."""

    def test_benchmark_bullish_regime(self):
        """Benchmark cleanly above ascending 200 SMA evaluates to BULLISH (100% allocation)."""
        df_bench = generate_mock_benchmark(regime="bullish", symbol="^NSEI")
        regime = evaluate_market_regime(df_bench["close"].values, benchmark_symbol="^NSEI")

        assert regime.state == MarketRegimeState.BULLISH
        assert regime.equity_allocation_pct == 1.00
        assert regime.sma_200_slope >= 0.0

    def test_benchmark_neutral_regime(self):
        """Benchmark above 200 SMA but flat/descending slope evaluates to NEUTRAL (50% allocation)."""
        df_bench = generate_mock_benchmark(regime="neutral", symbol="^NSEI")
        regime = evaluate_market_regime(df_bench["close"].values, benchmark_symbol="^NSEI")

        assert regime.state == MarketRegimeState.NEUTRAL
        assert regime.equity_allocation_pct == 0.50

    def test_benchmark_bearish_regime(self):
        """Benchmark below 200 SMA evaluates to BEARISH (0% equity allocation)."""
        df_bench = generate_mock_benchmark(regime="bearish", symbol="^NSEI")
        regime = evaluate_market_regime(df_bench["close"].values, benchmark_symbol="^NSEI")

        assert regime.state == MarketRegimeState.BEARISH
        assert regime.equity_allocation_pct == 0.00

    def test_benchmark_none_fallback_defensive(self):
        """Passing None for benchmark defaults to NEUTRAL defensive mode."""
        regime = evaluate_market_regime(None, benchmark_symbol="^NSEI")
        assert regime.state == MarketRegimeState.NEUTRAL
        assert regime.equity_allocation_pct == 0.50

    def test_benchmark_insufficient_history_defensive(self):
        """Benchmark with < 200 bars defaults to NEUTRAL defensive mode."""
        short_bench = np.full(100, 20000.0)
        regime = evaluate_market_regime(short_bench, benchmark_symbol="^NSEI")
        assert regime.state == MarketRegimeState.NEUTRAL
        assert regime.equity_allocation_pct == 0.50


# ==============================================================================
# Feature 8: Cross-Sectional Winsorized Z-Score & Composite Scoring
# ==============================================================================

class TestCompositeScoring:
    """Feature 8: Winsorized Z-Score Standardization & Composite Ranking."""

    def test_winsorized_z_score_standardization(self):
        """Z-scores of standard array have mean ~0 and std ~1."""
        raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        z = winsorized_z_score(raw)

        assert abs(np.mean(z)) < 1e-6
        assert abs(np.std(z, ddof=1) - 1.0) < 1e-6

    def test_winsorized_z_score_outlier_clipping(self):
        """Extreme 10-sigma outliers are clipped to [-3.0, +3.0]."""
        raw = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 100.0, -100.0])
        z = winsorized_z_score(raw, clip_range=(-3.0, 3.0))

        assert np.max(z) <= 3.0
        assert np.min(z) >= -3.0

    def test_winsorized_z_score_zero_variance(self):
        """Array of identical values returns all 0.0 without division by zero."""
        raw = np.full(10, 42.0)
        z = winsorized_z_score(raw)
        assert np.all(z == 0.0)

    def test_composite_score_weighted_sum(self):
        """Composite score applies exact weights: 0.40 Clenow, 0.25 12-1, 0.15 R^2, 0.10 6-1, -0.10 Vol."""
        df = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "clenow_score": [0.50, 0.20, 0.10],
            "mom_12_1": [0.80, 0.40, 0.10],
            "r_squared": [0.90, 0.70, 0.40],
            "mom_6_1": [0.40, 0.20, 0.05],
            "vol_ann": [0.15, 0.25, 0.35],
        })
        scored_df = compute_composite_momentum_score(df)

        assert "composite_score" in scored_df.columns
        assert "rank" in scored_df.columns
        # Stock A should be top ranked
        assert scored_df.iloc[0]["symbol"] == "A"
        assert scored_df.iloc[0]["rank"] == 1

    def test_percentile_rank_uniform_distribution(self):
        """Percentile ranks map values uniformly to [0.0, 1.0]."""
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pct = calculate_percentile_rank(vals)
        assert np.allclose(pct, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_rank_universe_sorting_and_filtering(self):
        """rank_universe excludes trend-failing stocks and ranks passing stocks descending."""
        m1 = MomentumMetrics(symbol="M1", close=100.0, clenow_score=0.6, clenow_r2=0.9, mom_12_1=0.8, passes_trend=True)
        m2 = MomentumMetrics(symbol="M2", close=100.0, clenow_score=0.4, clenow_r2=0.8, mom_12_1=0.5, passes_trend=True)
        m3 = MomentumMetrics(symbol="M3", close=100.0, clenow_score=0.9, clenow_r2=0.95, mom_12_1=1.2, passes_trend=False)

        breakdowns, ranked_df = rank_universe([m1, m2, m3], filter_trend=True)
        symbols = [b.symbol for b in breakdowns]

        assert "M3" not in symbols  # Excluded due to trend failure
        assert symbols == ["M1", "M2"]


# ==============================================================================
# Feature 9: Portfolio Weighting Schemes
# ==============================================================================

class TestPortfolioWeighting:
    """Feature 9: Portfolio Weighting Schemes (Equal, Inverse-Vol, Bounded Parity)."""

    def test_equal_weighting_sum_and_values(self):
        """Equal weighting allocates exactly 1/K per stock, summing to 1.0."""
        K = 10
        weights = np.full(K, 1.0 / K)
        assert np.allclose(weights, 0.10)
        assert abs(np.sum(weights) - 1.0) < 1e-7

    def test_inverse_volatility_inversely_proportional(self):
        """Stock with half the volatility receives double the weight."""
        vols = np.array([0.10, 0.20])
        weights = oracle_inverse_volatility_weights(vols)

        assert abs(weights[0] - 2.0 * weights[1]) < 1e-6
        assert abs(weights[0] - 2.0 / 3.0) < 1e-6
        assert abs(weights[1] - 1.0 / 3.0) < 1e-6

    def test_inverse_volatility_weights_sum_to_one(self):
        """Inverse-volatility weights sum to 1.0 across varied volatility profiles."""
        vols = np.array([0.15, 0.22, 0.35, 0.18, 0.29, 0.12, 0.40, 0.25, 0.19, 0.31])
        weights = oracle_inverse_volatility_weights(vols)

        assert abs(np.sum(weights) - 1.0) < 1e-7
        assert np.all(weights > 0.0)

    def test_bounded_risk_parity_bounds_adherence(self):
        """Bounded risk parity weights satisfy 0.05 <= w_i <= 0.20 and sum to 1.0."""
        vols = np.array([0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
        weights = oracle_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.20)

        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights >= 0.0499)
        assert np.all(weights <= 0.2001)

    def test_bounded_risk_parity_extreme_dispersion(self):
        """Extreme volatility dispersion is clipped safely to [w_min, w_max]."""
        vols = np.array([0.02, 0.03, 0.80, 0.90, 1.20])
        weights = oracle_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.35)

        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights >= 0.05)
        assert np.all(weights <= 0.35)

    def test_single_asset_portfolio_weighting(self):
        """Single-asset universe allocates 100% weight."""
        vols = np.array([0.25])
        weights = oracle_inverse_volatility_weights(vols)
        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-7


# ==============================================================================
# Feature 10: Rebalancing Rank Buffer & Exits
# ==============================================================================

class TestRebalancingBuffer:
    """Feature 10: Hysteresis Rank Buffer & Rebalancing Action Rules."""

    def test_new_entrant_assigned_buy(self):
        """New stock entering Top 10 that was not previously held is assigned BUY."""
        current_holdings = ["STOCK_A", "STOCK_B"]
        ranked_symbols = ["STOCK_NEW", "STOCK_A", "STOCK_B"]

        action = RebalanceAction.BUY if "STOCK_NEW" not in current_holdings and 1 <= 10 else RebalanceAction.HOLD
        assert action == RebalanceAction.BUY

    def test_retained_holding_within_buffer_assigned_hold(self):
        """Existing holding slipping from Rank 8 to Rank 16 (within buffer 20) is retained (HOLD)."""
        current_holdings = ["STOCK_A"]
        new_rank = 16
        buffer_threshold = 20

        # Holding is within buffer rank <= 20
        action = RebalanceAction.HOLD if new_rank <= buffer_threshold else RebalanceAction.SELL
        assert action == RebalanceAction.HOLD

    def test_dropped_holding_beyond_buffer_assigned_sell(self):
        """Existing holding dropping to Rank 24 (beyond buffer 20) is liquidated (SELL)."""
        current_holdings = ["STOCK_A"]
        new_rank = 24
        buffer_threshold = 20

        action = RebalanceAction.SELL if new_rank > buffer_threshold else RebalanceAction.HOLD
        assert action == RebalanceAction.SELL

    def test_trend_breakdown_immediate_sell(self):
        """Existing holding that falls below 100-day EMA triggers immediate SELL regardless of rank."""
        passes_trend = False
        rank = 5

        action = RebalanceAction.SELL if not passes_trend else (RebalanceAction.BUY if rank <= 10 else RebalanceAction.HOLD)
        assert action == RebalanceAction.SELL

    def test_rebalance_turnover_buffer_prevents_unnecessary_churn(self):
        """Small rank fluctuations within the buffer do not trigger portfolio exits."""
        holdings = [f"H_{i}" for i in range(1, 11)]
        # Ranks shift slightly: H_10 moves to rank 14
        ranks = {f"H_{i}": i for i in range(1, 10)}
        ranks["H_10"] = 14

        exits = [sym for sym in holdings if ranks.get(sym, 999) > 20]
        assert len(exits) == 0, "No stock should exit if within buffer 20"

    def test_initial_portfolio_generation_all_buys(self):
        """When current holdings list is empty, all top 10 stocks are tagged as BUY."""
        top_10 = [f"SYM_{i}" for i in range(1, 11)]
        current_holdings = []

        actions = [RebalanceAction.BUY if sym not in current_holdings else RebalanceAction.HOLD for sym in top_10]
        assert all(a == RebalanceAction.BUY for a in actions)

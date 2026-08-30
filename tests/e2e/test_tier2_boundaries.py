"""
Tier 2: Boundary, Edge & Numerical Corner Cases Test Suite.
Validates exact boundaries, missing/corrupted data, extreme outliers, and degenerate states.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from tests.e2e.fixtures import (
    generate_deterministic_series,
    oracle_bounded_risk_parity_weights,
    oracle_inverse_volatility_weights,
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


class TestBoundaryCutoffs:
    """Exact data length boundaries and cutoffs."""

    def test_exact_90_day_clenow_boundary(self):
        """Length 89 returns NaN; length 90 computes valid Clenow metrics."""
        p_89 = np.linspace(100.0, 150.0, 89)
        slope_89, r2_89, score_89 = compute_clenow_momentum(p_89, lookback=90)
        assert np.isnan(slope_89)
        assert np.isnan(r2_89)
        assert np.isnan(score_89)

        p_90 = np.linspace(100.0, 150.0, 90)
        slope_90, r2_90, score_90 = compute_clenow_momentum(p_90, lookback=90)
        assert not np.isnan(slope_90)
        assert not np.isnan(r2_90)
        assert not np.isnan(score_90)
        assert r2_90 > 0.95

    def test_exact_252_day_classical_momentum_boundary(self):
        """Length 251 returns NaN; length 252 computes valid 12-1 momentum."""
        p_251 = np.linspace(100.0, 200.0, 251)
        mom_251 = compute_jegadeesh_titman_momentum(p_251, lookback_total=252, skip_recent=21)
        assert np.isnan(mom_251)

        p_252 = np.linspace(100.0, 200.0, 252)
        mom_252 = compute_jegadeesh_titman_momentum(p_252, lookback_total=252, skip_recent=21)
        assert not np.isnan(mom_252)
        assert mom_252 > 0.50

    def test_exact_126_day_intermediate_momentum_boundary(self):
        """Length 125 returns NaN; length 126 computes valid 6-1 momentum."""
        p_125 = np.linspace(100.0, 150.0, 125)
        mom_125 = calculate_intermediate_momentum(p_125, lookback=126, skip_recent=21)
        assert np.isnan(mom_125)

        p_126 = np.linspace(100.0, 150.0, 126)
        mom_126 = calculate_intermediate_momentum(p_126, lookback=126, skip_recent=21)
        assert not np.isnan(mom_126)


class TestEmptyAndCorruptedData:
    """Resilience against empty, None, NaN, Inf, and non-positive price series."""

    def test_none_and_empty_inputs_across_all_modules(self):
        """All functions return graceful NaNs/empty on None or empty arrays without uncaught exceptions."""
        assert np.isnan(compute_clenow_momentum(None)[0])
        assert np.isnan(compute_clenow_momentum([])[0])
        assert np.isnan(compute_jegadeesh_titman_momentum(None))
        assert np.isnan(compute_jegadeesh_titman_momentum([]))
        assert np.isnan(calculate_realized_volatility(None))
        assert np.isnan(calculate_realized_volatility([]))
        assert len(winsorized_z_score(None)) == 0
        assert len(winsorized_z_score([])) == 0
        assert len(calculate_percentile_rank(None)) == 0
        assert len(calculate_percentile_rank([])) == 0
        assert compute_composite_momentum_score(None).empty
        assert compute_composite_momentum_score(pd.DataFrame()).empty

    def test_all_nan_series_handling(self):
        """Price series consisting entirely of NaNs returns NaN without crashing."""
        nan_prices = np.full(100, np.nan)
        slope, r2, score = compute_clenow_momentum(nan_prices, lookback=90)
        assert np.isnan(slope)
        assert np.isnan(r2)

        vol = calculate_realized_volatility(nan_prices, lookback=90)
        assert np.isnan(vol)

        mom = compute_jegadeesh_titman_momentum(nan_prices, lookback_total=252)
        assert np.isnan(mom)

    def test_negative_and_zero_price_protection(self):
        """Non-positive prices (<= 0.0) invalid for log transformation return NaN."""
        invalid_prices = np.linspace(-10.0, 50.0, 100)
        slope, r2, score = compute_clenow_momentum(invalid_prices, lookback=90)
        assert np.isnan(slope)

        zero_prices = np.zeros(100)
        slope_z, r2_z, _ = compute_clenow_momentum(zero_prices, lookback=90)
        assert np.isnan(slope_z)

    def test_missing_and_corrupted_dataframe_columns(self):
        """evaluate_stock_eligibility handles missing OHLCV columns gracefully."""
        df_no_close = pd.DataFrame({"open": [100.0, 105.0], "volume": [1000, 2000]})
        res = evaluate_stock_eligibility("NO_CLOSE", df_no_close)
        assert res.is_eligible is False
        assert "Missing Close column" in res.reasons[0]


class TestNumericalCornersAndDegenerateUniverses:
    """Safe clamping on zero volatility, single asset universes, and extreme outliers."""

    def test_zero_volatility_safe_clamping_in_weighting(self):
        """Zero volatility series clamped at epsilon / floor without ZeroDivisionError."""
        vols = np.array([0.0, 0.0, 0.0])
        weights = oracle_inverse_volatility_weights(vols, min_vol=0.05)
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert np.allclose(weights, 1.0 / 3.0)

    def test_single_asset_universe_pipeline(self):
        """Universe with a single stock allocates 100% weight and rank 1."""
        m = MomentumMetrics(
            symbol="SOLO",
            close=100.0,
            clenow_score=0.5,
            clenow_r2=0.9,
            mom_12_1=0.7,
            volatility_ann=0.20,
            passes_trend=True,
        )
        breakdowns, ranked_df = rank_universe([m], filter_trend=True)
        assert len(breakdowns) == 1
        assert breakdowns[0].rank == 1
        assert breakdowns[0].symbol == "SOLO"

    def test_small_universe_fewer_than_target_portfolio_size(self):
        """Universe with 4 qualifying stocks when top 10 requested allocates all 4 properly."""
        metrics = [
            MomentumMetrics(symbol=f"S_{i}", close=100.0, clenow_score=0.5 - 0.1 * i, clenow_r2=0.85, mom_12_1=0.6 - 0.1 * i, passes_trend=True)
            for i in range(4)
        ]
        breakdowns, ranked_df = rank_universe(metrics, filter_trend=True)
        assert len(breakdowns) == 4
        assert [b.rank for b in breakdowns] == [1, 2, 3, 4]

    def test_extreme_outlier_winsorization_insulates_universe(self):
        """Single stock with +1000% jump winsorized at Z=+3.0, preserving relative order of other stocks."""
        # 20 stocks with regular returns, 1 with extreme jump
        scores = np.linspace(0.10, 0.50, 20)
        scores = np.append(scores, 50.0)  # Extreme outlier

        z = winsorized_z_score(scores, clip_range=(-3.0, 3.0))
        assert z[-1] == 3.0  # Outlier capped at +3.0

        # Ranks among non-outlier stocks remain strictly preserved
        for i in range(19):
            assert z[i] < z[i + 1]

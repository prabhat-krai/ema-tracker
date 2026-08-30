"""
Unit Tests for Cross-Sectional Standardization, Winsorized Z-Scoring & Composite Momentum Ranking.
"""

import numpy as np
import pandas as pd
import pytest

from src.momentum.models import MomentumMetrics, RebalanceAction
from src.momentum.scoring import (
    calculate_percentile_rank,
    compute_composite_momentum_score,
    rank_universe,
    winsorized_z_score,
)


class TestZScoreAndPercentiles:
    """Test standardization and non-parametric percentile ranking."""

    def test_winsorized_z_score_standard_normal(self):
        """Standard array transforms to mean ~0.0 and std ~1.0."""
        np.random.seed(42)
        raw_vals = np.random.normal(50.0, 10.0, size=100)
        z = winsorized_z_score(raw_vals)

        assert np.mean(z) == pytest.approx(0.0, abs=1e-5)
        assert np.std(z, ddof=1) == pytest.approx(1.0, abs=1e-2)
        assert np.all(z >= -3.0) and np.all(z <= 3.0)

    def test_winsorized_z_score_clipping_extreme_outliers(self):
        """Values exceeding +/- 3.0 sigma are capped at +/- 3.0."""
        # 50 points centered at 10.0 with std ~1.0, plus extreme outliers at +1000 and -1000
        baseline = np.random.RandomState(42).normal(10.0, 1.0, size=50)
        raw_vals = np.concatenate([baseline, [1000.0, -1000.0]])
        z = winsorized_z_score(raw_vals)

        assert np.max(z) == 3.0
        assert np.min(z) == -3.0

    def test_winsorized_z_score_zero_variance(self):
        """Constant array returns array of 0.0."""
        raw_vals = np.full(50, 42.0)
        z = winsorized_z_score(raw_vals)
        assert np.all(z == 0.0)

    def test_winsorized_z_score_with_nans(self):
        """NaN values are filled with 0.0 neutral score."""
        raw_vals = np.array([10.0, 20.0, 30.0, np.nan, 50.0])
        z = winsorized_z_score(raw_vals)
        assert z[3] == 0.0
        assert len(z) == 5

    def test_percentile_rank(self):
        """Values are mapped to [0.0, 1.0] range."""
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pcts = calculate_percentile_rank(vals, ascending=True)

        assert pcts[0] == pytest.approx(0.0)
        assert pcts[-1] == pytest.approx(1.0)
        assert pcts[2] == pytest.approx(0.50)


class TestCompositeScoringAndRanking:
    """Test multi-factor composite scoring and universe ranking."""

    def test_compute_composite_momentum_score_dataframe(self):
        """DataFrame with multiple factors is correctly standardized, scored, and ranked."""
        data = {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "clenow_score": [0.80, 0.40, -0.10, 0.50],
            "mom_12_1": [1.20, 0.60, -0.20, 0.70],
            "r_squared": [0.95, 0.70, 0.20, 0.85],
            "mom_6_1": [0.50, 0.25, -0.10, 0.30],
            "vol_ann": [0.15, 0.25, 0.40, 0.20],  # Lower vol is better
        }
        df = pd.DataFrame(data)
        scored_df = compute_composite_momentum_score(df)

        assert "composite_score" in scored_df.columns
        assert "rank" in scored_df.columns
        assert "percentile_rank" in scored_df.columns

        # AAA should be top ranked (highest momentum and lowest volatility)
        assert scored_df.iloc[0]["symbol"] == "AAA"
        assert scored_df.iloc[0]["rank"] == 1
        # CCC should be lowest ranked
        assert scored_df.iloc[-1]["symbol"] == "CCC"
        assert scored_df.iloc[-1]["rank"] == 4

    def test_rank_universe_with_trend_filtering(self):
        """rank_universe filters out stocks failing macro trend filter."""
        metrics_list = [
            MomentumMetrics(
                symbol="LEADER",
                close=200.0,
                clenow_slope_ann=0.50,
                clenow_r2=0.90,
                clenow_score=0.45,
                mom_12_1=0.80,
                mom_6_1=0.40,
                volatility_ann=0.18,
                passes_trend=True,
            ),
            MomentumMetrics(
                symbol="RUNNER_UP",
                close=100.0,
                clenow_slope_ann=0.30,
                clenow_r2=0.80,
                clenow_score=0.24,
                mom_12_1=0.50,
                mom_6_1=0.25,
                volatility_ann=0.22,
                passes_trend=True,
            ),
            MomentumMetrics(
                symbol="FAILED_TREND",
                close=50.0,
                clenow_slope_ann=0.90,  # High slope but fails trend
                clenow_r2=0.95,
                clenow_score=0.85,
                mom_12_1=1.50,
                mom_6_1=0.60,
                volatility_ann=0.20,
                passes_trend=False,     # Should be excluded
            ),
        ]

        breakdowns, scored_df = rank_universe(metrics_list, filter_trend=True)

        symbols = [b.symbol for b in breakdowns]
        assert "LEADER" in symbols
        assert "RUNNER_UP" in symbols
        assert "FAILED_TREND" not in symbols  # Excluded by trend filter

        assert breakdowns[0].symbol == "LEADER"
        assert breakdowns[0].rank == 1
        assert breakdowns[0].action == RebalanceAction.BUY

    def test_rank_universe_empty_or_no_eligible(self):
        """Returns empty list and empty DataFrame when no stocks are eligible."""
        breakdowns, df = rank_universe([])
        assert len(breakdowns) == 0
        assert df.empty

        metrics_all_failed = [
            MomentumMetrics(symbol="FAIL1", close=10.0, passes_trend=False),
            MomentumMetrics(symbol="FAIL2", close=20.0, passes_trend=False),
        ]
        breakdowns, df = rank_universe(metrics_all_failed, filter_trend=True)
        assert len(breakdowns) == 0
        assert df.empty

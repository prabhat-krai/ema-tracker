"""
Unit Tests for Quantitative Momentum Portfolio Selection & Weighting Engine.
"""

import numpy as np
import pytest

from src.momentum.models import (
    MarketRegime,
    MarketRegimeState,
    MomentumScoreBreakdown,
    PortfolioConstituent,
    RebalanceAction,
)
from src.momentum.portfolio import (
    build_momentum_portfolio,
    calculate_bounded_risk_parity_weights,
    calculate_equal_weights,
    calculate_inverse_volatility_weights,
    calculate_portfolio_risk_metrics,
)


class TestWeightingAlgorithms:
    """Tests for Equal, Inverse-Volatility, and Bounded Risk-Parity Weighting."""

    def test_calculate_equal_weights_standard(self):
        w = calculate_equal_weights(10)
        assert len(w) == 10
        assert np.allclose(w, 0.10)
        assert abs(np.sum(w) - 1.0) < 1e-7

    def test_calculate_equal_weights_edge_cases(self):
        w_single = calculate_equal_weights(1)
        assert len(w_single) == 1 and w_single[0] == 1.0

        w_zero = calculate_equal_weights(0)
        assert len(w_zero) == 0

    def test_calculate_inverse_volatility_weights_standard(self):
        vols = np.array([0.10, 0.20, 0.40])
        w = calculate_inverse_volatility_weights(vols)

        assert abs(np.sum(w) - 1.0) < 1e-7
        # Lower volatility asset must have strictly higher weight
        assert w[0] > w[1] > w[2]
        # Ratio of inv vol 0.10 vs 0.20 should be 2:1
        assert np.isclose(w[0] / w[1], 2.0)

    def test_calculate_inverse_volatility_weights_flooring(self):
        # Extremely low / zero volatility is floored at min_vol (0.05)
        vols = np.array([0.0, 0.05, 0.10])
        w = calculate_inverse_volatility_weights(vols, min_vol=0.05)

        assert abs(np.sum(w) - 1.0) < 1e-7
        assert np.isclose(w[0], w[1])  # Both floored to 0.05

    def test_calculate_inverse_volatility_single_asset(self):
        w = calculate_inverse_volatility_weights(np.array([0.25]))
        assert len(w) == 1 and w[0] == 1.0

    def test_calculate_bounded_risk_parity_weights_bounds(self):
        vols = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.20)

        assert abs(np.sum(w) - 1.0) < 1e-6
        assert np.all(w >= 0.0499)
        assert np.all(w <= 0.2001)

    def test_calculate_bounded_risk_parity_infeasible_fallback(self):
        # 10 assets with w_min = 0.20 (sum = 2.0 > 1.0) -> fallback to equal weights (0.10)
        vols = np.full(10, 0.20)
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.20, w_max=0.30)
        assert abs(np.sum(w) - 1.0) < 1e-7
        assert np.allclose(w, 0.10)


class TestPortfolioRiskMetrics:
    """Tests for portfolio risk metrics calculation."""

    def test_risk_metrics_calculation(self):
        constituents = [
            PortfolioConstituent(
                symbol="A",
                rank=1,
                weight=0.60,
                composite_score=1.0,
                clenow_score=0.50,
                clenow_slope_ann=0.60,
                clenow_r2=0.83,
                mom_12_1=0.80,
                mom_6_1=0.40,
                volatility_ann=0.15,
                natr_14=2.0,
                close=100.0,
            ),
            PortfolioConstituent(
                symbol="B",
                rank=2,
                weight=0.40,
                composite_score=0.8,
                clenow_score=0.30,
                clenow_slope_ann=0.40,
                clenow_r2=0.75,
                mom_12_1=0.50,
                mom_6_1=0.25,
                volatility_ann=0.25,
                natr_14=3.0,
                close=50.0,
            ),
        ]

        metrics = calculate_portfolio_risk_metrics(constituents)
        # Weighted score: 0.60*0.50 + 0.40*0.30 = 0.30 + 0.12 = 0.42
        assert metrics["weighted_clenow_score"] == pytest.approx(0.42)
        # Weighted vol: 0.60*0.15 + 0.40*0.25 = 0.09 + 0.10 = 0.19
        assert metrics["weighted_volatility"] == pytest.approx(0.19)
        assert metrics["total_equity_pct"] == pytest.approx(1.0)
        assert metrics["expected_cash_pct"] == pytest.approx(0.0)

    def test_risk_metrics_empty(self):
        metrics = calculate_portfolio_risk_metrics([])
        assert metrics["total_equity_pct"] == 0.0
        assert metrics["expected_cash_pct"] == 1.0


class TestBuildMomentumPortfolio:
    """Tests for build_momentum_portfolio end-to-end recommendation generator."""

    @pytest.fixture
    def mock_breakdowns(self):
        breakdowns = []
        for i in range(1, 15):
            b = MomentumScoreBreakdown(
                symbol=f"SYM_{i}",
                rank=i,
                composite_score=2.0 - i * 0.1,
                percentile_rank=1.0 - (i - 1) / 14.0,
                raw_clenow_score=0.60 - i * 0.03,
                raw_slope_ann=0.70 - i * 0.03,
                raw_r2=0.85,
                raw_mom_12_1=0.80 - i * 0.04,
                raw_mom_6_1=0.40 - i * 0.02,
                raw_vol_ann=0.15 + i * 0.01,
                close=100.0 * i,
                natr_14=2.5,
            )
            breakdowns.append(b)
        return breakdowns

    def test_build_portfolio_bullish_inv_vol(self, mock_breakdowns):
        regime = MarketRegime(
            benchmark_symbol="^NSEI",
            state=MarketRegimeState.BULLISH,
            benchmark_price=24000.0,
            sma_200=22000.0,
            sma_200_slope=0.01,
            equity_allocation_pct=1.0,
        )

        rec = build_momentum_portfolio(
            ranked_constituents=mock_breakdowns,
            top_n=10,
            weighting_scheme="inv_vol",
            current_holdings=["SYM_2", "SYM_12"],
            market_regime=regime,
            total_capital=1_000_000.0,
        )

        assert len(rec.target_constituents) == 10
        assert rec.total_equity_pct == pytest.approx(1.0, abs=1e-6)
        assert rec.expected_cash_pct == pytest.approx(0.0, abs=1e-6)

        # Action checking: SYM_1 is not held -> BUY, SYM_2 is held -> HOLD
        actions = {c.symbol: c.action for c in rec.target_constituents}
        assert actions["SYM_1"] == RebalanceAction.BUY
        assert actions["SYM_2"] == RebalanceAction.HOLD

        # Target shares and value checking
        for c in rec.target_constituents:
            assert c.target_value > 0
            assert c.target_shares > 0
            assert c.target_shares * c.close <= c.target_value

    def test_build_portfolio_neutral_half_cash(self, mock_breakdowns):
        regime = MarketRegime(
            benchmark_symbol="^NSEI",
            state=MarketRegimeState.NEUTRAL,
            benchmark_price=23000.0,
            sma_200=22500.0,
            sma_200_slope=-0.005,
            equity_allocation_pct=0.50,
        )

        rec = build_momentum_portfolio(
            ranked_constituents=mock_breakdowns,
            top_n=10,
            weighting_scheme="equal",
            market_regime=regime,
        )

        assert len(rec.target_constituents) == 10
        assert rec.total_equity_pct == pytest.approx(0.50, abs=1e-6)
        assert rec.expected_cash_pct == pytest.approx(0.50, abs=1e-6)
        for c in rec.target_constituents:
            assert c.weight == pytest.approx(0.05)  # 0.10 * 0.50 = 0.05

    def test_build_portfolio_bearish_full_cash(self, mock_breakdowns):
        regime = MarketRegime(
            benchmark_symbol="^NSEI",
            state=MarketRegimeState.BEARISH,
            benchmark_price=21000.0,
            sma_200=22500.0,
            sma_200_slope=-0.01,
            equity_allocation_pct=0.0,
        )

        rec = build_momentum_portfolio(
            ranked_constituents=mock_breakdowns,
            top_n=10,
            weighting_scheme="bounded_parity",
            market_regime=regime,
        )

        assert rec.total_equity_pct == pytest.approx(0.0, abs=1e-6)
        assert rec.expected_cash_pct == pytest.approx(1.0, abs=1e-6)
        for c in rec.target_constituents:
            assert c.weight == 0.0

    def test_build_portfolio_empty_ranked_constituents(self):
        rec = build_momentum_portfolio(ranked_constituents=[])
        assert len(rec.target_constituents) == 0
        assert rec.expected_cash_pct == 1.0

    def test_build_portfolio_invalid_scheme_raises(self, mock_breakdowns):
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            build_momentum_portfolio(
                ranked_constituents=mock_breakdowns,
                weighting_scheme="martingale",
            )

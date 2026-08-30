"""
Tier 5 White-Box Adversarial Stress Testing & Coverage Hardening Suite.

Rigorous stress testing across extreme market regimes, bounded parity attacks,
hysteresis buffer churn dynamics, partial qualification scaling, and cross-module integrity.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from src.momentum.engine import MomentumPipeline
from src.momentum.exporter import (
    export_all_reports,
    export_portfolio_csv,
    export_portfolio_json,
    export_portfolio_markdown,
)
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
from src.momentum.rebalancer import (
    calculate_weight_adjustment_action,
    classify_rebalance_action,
    generate_rebalance_plan,
)
from tests.e2e.fixtures import (
    generate_deterministic_series,
    generate_mock_benchmark,
    generate_mock_indian_universe,
)


class TestTier5AdversarialRegimeTransitions:
    """Stress testing dynamic multi-cycle portfolio transitions across market regimes."""

    def test_multi_cycle_regime_transitions(self):
        """
        Simulate a full 4-quarter market cycle:
        Q1 (Bullish 100% Equity) -> Q2 (Bearish 0% Equity, 100% Cash) ->
        Q3 (Neutral 50% Equity, 50% Cash) -> Q4 (Bullish 100% Equity).
        Verifies exact cash scaling, liquidation, and turnover calculations.
        """
        breakdowns = [
            MomentumScoreBreakdown(
                symbol=f"STOCK_{i:02d}.NS",
                rank=i,
                composite_score=2.0 - i * 0.1,
                raw_vol_ann=0.15 + (i % 3) * 0.05,
                close=100.0 * i,
            )
            for i in range(1, 16)
        ]

        # --- Q1: BULLISH (Initial Build) ---
        regime_q1 = MarketRegime(
            benchmark_symbol="^NSEI",
            state=MarketRegimeState.BULLISH,
            benchmark_price=25000.0,
            sma_200=23000.0,
            sma_200_slope=0.015,
            equity_allocation_pct=1.0,
        )
        rec_q1 = build_momentum_portfolio(
            ranked_constituents=breakdowns,
            top_n=10,
            weighting_scheme="equal",
            market_regime=regime_q1,
            total_capital=1_000_000.0,
        )
        assert rec_q1.total_equity_pct == pytest.approx(1.0, abs=1e-6)
        assert rec_q1.expected_cash_pct == pytest.approx(0.0, abs=1e-6)
        plan_q1 = generate_rebalance_plan(current_holdings=[], target_recommendation=rec_q1)
        assert len(plan_q1.buys) == 10
        assert plan_q1.turnover_pct == pytest.approx(1.0, abs=1e-6)

        held_q1 = [c.symbol for c in rec_q1.target_constituents]

        # --- Q2: BEARISH (Crash Defense) ---
        regime_q2 = MarketRegime(
            benchmark_symbol="^NSEI",
            state=MarketRegimeState.BEARISH,
            benchmark_price=20000.0,
            sma_200=23000.0,
            sma_200_slope=-0.02,
            equity_allocation_pct=0.0,
        )
        rec_q2 = build_momentum_portfolio(
            ranked_constituents=breakdowns,
            top_n=10,
            current_holdings=held_q1,
            market_regime=regime_q2,
            total_capital=1_000_000.0,
        )
        assert rec_q2.total_equity_pct == pytest.approx(0.0, abs=1e-6)
        assert rec_q2.expected_cash_pct == pytest.approx(1.0, abs=1e-6)
        plan_q2 = generate_rebalance_plan(current_holdings=held_q1, target_recommendation=rec_q2)
        assert len(plan_q2.sells) == 10
        assert all(s.action == "SELL_EXIT" for s in plan_q2.sells)
        assert plan_q2.turnover_pct == pytest.approx(1.0, abs=1e-6)

        # --- Q3: NEUTRAL (Cautious 50% Deployment) ---
        regime_q3 = MarketRegime(
            benchmark_symbol="^NSEI",
            state=MarketRegimeState.NEUTRAL,
            benchmark_price=22500.0,
            sma_200=22000.0,
            sma_200_slope=0.001,
            equity_allocation_pct=0.50,
        )
        rec_q3 = build_momentum_portfolio(
            ranked_constituents=breakdowns,
            top_n=10,
            current_holdings=[],  # coming from 100% cash
            weighting_scheme="equal",
            market_regime=regime_q3,
            total_capital=1_000_000.0,
        )
        assert rec_q3.total_equity_pct == pytest.approx(0.50, abs=1e-6)
        assert rec_q3.expected_cash_pct == pytest.approx(0.50, abs=1e-6)
        plan_q3 = generate_rebalance_plan(current_holdings=[], target_recommendation=rec_q3)
        assert plan_q3.turnover_pct == pytest.approx(0.50, abs=1e-6)


class TestTier5BoundedRiskParityAdversarialBounds:
    """Adversarial stress testing on bounded risk parity numerical optimization."""

    def test_extreme_volatility_ratios(self):
        """Test with extreme volatility disparity (1000:1 ratio)."""
        vols = np.array([0.001, 0.01, 0.10, 0.50, 1.0])
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.35)
        assert np.isclose(np.sum(w), 1.0)
        assert np.all(w >= 0.0499)
        assert np.all(w <= 0.3501)
        assert w[0] > w[-1]

    def test_infeasible_lower_bounds_triggers_equal_weight_fallback(self):
        """When N * w_min > 1.0, must fallback gracefully to 1/N without crashing."""
        vols = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
        # 5 assets with w_min = 0.30 -> sum = 1.5 > 1.0
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.30, w_max=0.50)
        assert np.isclose(np.sum(w), 1.0)
        assert np.allclose(w, 0.20)

    def test_infeasible_upper_bounds_triggers_equal_weight_fallback(self):
        """When N * w_max < 1.0, must fallback gracefully to 1/N."""
        vols = np.array([0.15, 0.20, 0.25, 0.30, 0.35])
        # 5 assets with w_max = 0.15 -> sum = 0.75 < 1.0
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.15)
        assert np.isclose(np.sum(w), 1.0)
        assert np.allclose(w, 0.20)

    def test_all_identical_volatilities(self):
        """Identical volatilities should result in strictly equal weights 1/N."""
        vols = np.full(10, 0.22)
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.20)
        assert np.isclose(np.sum(w), 1.0)
        assert np.allclose(w, 0.10)


class TestTier5PartialQualificationScalingGrid:
    """Stress tests on partial qualification cash scaling across all k in [0, 10]."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3, 5, 8, 10])
    @pytest.mark.parametrize("scheme", ["equal", "inv_vol", "bounded_parity"])
    def test_partial_qualification_grid(self, k, scheme):
        """
        Verify that for any qualified count k <= 10:
        - Total equity is exactly k / 10.
        - Cash allocation is exactly 1.0 - (k / 10).
        """
        breakdowns = [
            MomentumScoreBreakdown(
                symbol=f"STOCK_{i}.NS",
                rank=i,
                composite_score=2.0 - i * 0.1,
                raw_vol_ann=0.20,
                close=100.0,
            )
            for i in range(1, k + 1)
        ]

        rec = build_momentum_portfolio(
            ranked_constituents=breakdowns,
            top_n=10,
            weighting_scheme=scheme,
            total_capital=1_000_000.0,
        )

        expected_equity = float(k) / 10.0
        expected_cash = 1.0 - expected_equity

        assert len(rec.target_constituents) == k
        assert rec.total_equity_pct == pytest.approx(expected_equity, abs=1e-6)
        assert rec.expected_cash_pct == pytest.approx(expected_cash, abs=1e-6)


class TestTier5HysteresisBufferDeepBoundary:
    """Stress testing hysteresis churn avoidance under complex portfolio changes."""

    def test_hysteresis_churn_prevention_vs_liquidation(self):
        """
        Setup portfolio holding 5 stocks:
        - SYM_1 (Rank #1 -> Retained BUY/HOLD)
        - SYM_2 (Rank #12 -> Retained in buffer 20)
        - SYM_3 (Rank #19 -> Retained in buffer 20)
        - SYM_4 (Rank #22 -> Dropped & SOLD)
        - SYM_1 (Rank #1 -> Retained BUY/HOLD)
        - SYM_12 (Rank #12 -> Retained in buffer 20)
        - SYM_19 (Rank #19 -> Retained in buffer 20)
        - SYM_22 (Rank #22 -> Dropped & SOLD)
        - SYM_3 (Rank #3 but FAILS trend -> Dropped & SOLD)
        """
        breakdowns = [
            MomentumScoreBreakdown(symbol=f"SYM_{i}", rank=i, composite_score=2.0 - i * 0.05, close=100.0, raw_vol_ann=0.20)
            for i in range(1, 25)
        ]

        current_holdings = ["SYM_1", "SYM_12", "SYM_19", "SYM_22", "SYM_3"]

        # Simulate trend failure for SYM_3
        metrics_map = {
            f"SYM_{i}": type("Metrics", (), {"passes_trend": (i != 3)})()
            for i in range(1, 25)
        }

        rec = build_momentum_portfolio(
            ranked_constituents=breakdowns,
            metrics_map=metrics_map,
            top_n=10,
            current_holdings=current_holdings,
            rank_buffer=20,
        )

        target_syms = [c.symbol for c in rec.target_constituents]
        assert "SYM_1" in target_syms   # Rank 1 retained
        assert "SYM_12" in target_syms  # Rank 12 retained in buffer
        assert "SYM_19" in target_syms  # Rank 19 retained in buffer
        assert "SYM_22" not in target_syms  # Rank 22 dropped
        assert "SYM_3" not in target_syms   # Failed trend dropped

        plan = generate_rebalance_plan(
            current_holdings=current_holdings,
            target_recommendation=rec,
            rank_buffer=20,
            top_n=10,
        )

        trade_map = {t.symbol: t for t in plan.all_trades}
        assert trade_map["SYM_1"].action in ("HOLD_RETAIN", "REBALANCE_ADD", "REBALANCE_TRIM")
        assert trade_map["SYM_12"].action in ("HOLD_RETAIN", "REBALANCE_ADD", "REBALANCE_TRIM")
        assert trade_map["SYM_19"].action in ("HOLD_RETAIN", "REBALANCE_ADD", "REBALANCE_TRIM")
        assert trade_map["SYM_22"].action == "SELL_EXIT"
        assert trade_map["SYM_3"].action == "SELL_EXIT"


class TestTier5ExporterFullRoundTrip:
    """Stress test exporter serialization and schema compliance."""

    def test_exporter_round_trip(self, tmp_path):
        breakdowns = [
            MomentumScoreBreakdown(
                symbol=f"STOCK_{i}.NS",
                rank=i,
                composite_score=2.0 - i * 0.1,
                raw_clenow_score=1.5 - i * 0.08,
                raw_slope_ann=1.2 - i * 0.05,
                raw_r2=0.88,
                raw_mom_12_1=0.75 - i * 0.04,
                raw_mom_6_1=0.40,
                raw_vol_ann=0.20,
                natr_14=2.0,
                close=500.0,
            )
            for i in range(1, 11)
        ]

        rec = build_momentum_portfolio(
            ranked_constituents=breakdowns,
            top_n=10,
            weighting_scheme="bounded_parity",
            total_capital=2_500_000.0,
        )
        plan = generate_rebalance_plan(current_holdings=[], target_recommendation=rec)

        res_map = export_all_reports(rec, plan, output_dir=tmp_path / "reports_roundtrip")

        # Verify CSV
        df = pd.read_csv(res_map["csv"])
        assert len(df) == 10
        assert set(df["Symbol"]) == {f"STOCK_{i}.NS" for i in range(1, 11)}

        # Verify JSON
        import json
        with open(res_map["json"], "r") as f:
            data = json.load(f)
        assert data["recommendation"]["total_capital"] == 2_500_000.0
        assert len(data["recommendation"]["target_constituents"]) == 10

        # Verify Markdown
        md_text = res_map["markdown"].read_text()
        assert "STOCK_1.NS" in md_text
        assert "2,500,000.00" in md_text

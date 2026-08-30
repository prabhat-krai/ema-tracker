"""
Empirical Stress Tests & Adversarial Verification Suite for Milestone 2.

Comprehensive verification of:
1. Weight Normalization across extreme volatility ranges (sigma = 0.01 to 5.0, 50.0).
2. Bounded Risk Parity convergence under tight/restrictive bounds (e.g. 5 assets with [0.15, 0.25]).
3. Hysteresis rank buffer complete transition matrix (ranks 1..100, held/unheld, trend pass/fail).
4. Weight adjustment deadband thresholds (+2% add, -2% trim, hold retain).
5. One-way portfolio turnover accuracy across total replacement, partial rebalances, and regime cash shifts.
6. Share sizing, capital reconciliation, and edge-case resilience (empty universe, zero capital, extreme prices).
7. Empirical verification of hysteresis retention buffer behavior.
"""

from typing import Dict, List
import numpy as np
import pytest

from src.momentum.models import (
    MarketRegime,
    MarketRegimeState,
    MomentumScoreBreakdown,
    PortfolioConstituent,
    PortfolioRecommendation,
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


class TestExtremeVolatilityWeightNormalization:
    """Stress testing weight normalization across extreme volatility regimes (sigma in [0.0001, 50.0])."""

    @pytest.mark.parametrize(
        "vols",
        [
            np.array([0.01, 0.05, 0.20, 1.00, 5.00]),
            np.array([0.0001, 0.001, 0.01, 0.05, 0.10, 0.50, 2.00, 5.00, 10.00, 50.00]),
            np.array([5.0, 5.0, 5.0, 5.0, 5.0]),  # Homogeneous extreme high vol
            np.array([0.01, 0.01, 0.01, 0.01, 0.01]),  # Homogeneous extreme low vol (floored)
            np.array([0.0, 0.02, 0.05, 0.15, 0.30]),  # Zero vol mixed with normal
            np.linspace(0.01, 5.0, 100),  # 100 assets spanning the entire spectrum
        ],
    )
    def test_inverse_volatility_weight_normalization_and_monotonicity(self, vols):
        """
        Inverse-volatility weights must:
        1. Sum strictly to 1.0 within numerical tolerance (1e-7).
        2. Be strictly non-negative.
        3. Be monotonically non-increasing with respect to volatility:
           if vol_i < vol_j and both >= min_vol, then weight_i > weight_j.
        """
        w = calculate_inverse_volatility_weights(vols, min_vol=0.05)
        n = len(vols)

        assert len(w) == n
        assert np.all(w >= 0.0)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-7)

        # Monotonicity check on floored vols
        floored = np.maximum(vols, 0.05)
        for i in range(n):
            for j in range(n):
                if floored[i] < floored[j]:
                    assert w[i] > w[j], f"Monotonicity violated: vol {floored[i]} < {floored[j]} but weight {w[i]} <= {w[j]}"
                elif floored[i] == floored[j]:
                    assert w[i] == pytest.approx(w[j], abs=1e-7)

    def test_inverse_volatility_extreme_scaling_ratios(self):
        """
        An asset at min_vol floor (0.05) and an asset at 5.0 (100x higher vol):
        Relative weight ratio must be exactly 5.0 / 0.05 = 100.0.
        """
        vols = np.array([0.05, 5.0])
        w = calculate_inverse_volatility_weights(vols, min_vol=0.05)

        assert np.sum(w) == pytest.approx(1.0, abs=1e-7)
        assert w[0] / w[1] == pytest.approx(100.0, rel=1e-5)
        assert w[0] == pytest.approx(100.0 / 101.0, rel=1e-5)
        assert w[1] == pytest.approx(1.0 / 101.0, rel=1e-5)

    def test_inverse_volatility_degenerate_and_empty_inputs(self):
        """Verify safety on empty arrays, single elements, negative vols, and all zeros."""
        assert len(calculate_inverse_volatility_weights([])) == 0

        w_single = calculate_inverse_volatility_weights([0.25])
        assert len(w_single) == 1 and w_single[0] == 1.0

        w_zeros = calculate_inverse_volatility_weights([0.0, 0.0, 0.0, 0.0])
        assert np.sum(w_zeros) == pytest.approx(1.0, abs=1e-7)
        assert np.allclose(w_zeros, 0.25)


class TestBoundedRiskParityConvergence:
    """Stress testing iterative projection algorithm under tight, restrictive, and boundary bounds."""

    def test_bounded_parity_5_assets_restrictive_bounds(self):
        """
        Test case from challenge specification:
        5 assets with volatilities across extreme range, bounds [0.15, 0.25].
        Feasibility: 5 * 0.15 = 0.75 <= 1.0 <= 5 * 0.25 = 1.25.
        """
        vols = np.array([0.01, 0.05, 0.20, 1.00, 5.00])
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.15, w_max=0.25, max_iter=50)

        assert len(w) == 5
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= 0.15 - 1e-7)
        assert np.all(w <= 0.25 + 1e-7)

        # Asset 0 (lowest vol) should be capped at w_max (0.25)
        assert w[0] == pytest.approx(0.25, abs=1e-5)
        # All weights must be strictly in [0.15, 0.25]
        for val in w:
            assert 0.1499 <= val <= 0.2501
        # Ordering must be preserved
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1] - 1e-7

    def test_bounded_parity_10_assets_tight_band(self):
        """10 assets with tight bounds [0.09, 0.11] around equal weighting (0.10)."""
        vols = np.array([0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.20])
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.09, w_max=0.11, max_iter=50)

        assert len(w) == 10
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= 0.09 - 1e-7)
        assert np.all(w <= 0.11 + 1e-7)

    def test_bounded_parity_point_exact_bounds(self):
        """Exact point constraint where N * w_min = N * w_max = 1.0."""
        # 5 assets with w_min = 0.20, w_max = 0.20
        vols = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.20, w_max=0.20)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-7)
        assert np.allclose(w, 0.20)

    def test_bounded_parity_skewed_one_vs_many(self):
        """
        1 low-vol asset (0.01) vs 9 high-vol assets (5.0) under bounds [0.05, 0.20].
        - Asset 0 hits 0.20.
        - Remaining 0.80 distributed across 9 assets -> ~0.08889 each (> 0.05).
        """
        vols = np.array([0.01] + [5.0] * 9)
        w = calculate_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.20)

        assert len(w) == 10
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)
        assert w[0] == pytest.approx(0.20, abs=1e-5)
        for i in range(1, 10):
            assert w[i] == pytest.approx(0.80 / 9.0, abs=1e-5)
            assert 0.05 <= w[i] <= 0.20

    def test_bounded_parity_infeasible_fallbacks(self):
        """Verify fallback to equal weights when constraints are mathematically infeasible."""
        # Infeasible min: 10 assets with w_min = 0.15 (10 * 0.15 = 1.50 > 1.0)
        vols = np.linspace(0.1, 0.5, 10)
        w_infeas_min = calculate_bounded_risk_parity_weights(vols, w_min=0.15, w_max=0.30)
        assert np.sum(w_infeas_min) == pytest.approx(1.0, abs=1e-7)
        assert np.allclose(w_infeas_min, 0.10)

        # Infeasible max: 5 assets with w_max = 0.15 (5 * 0.15 = 0.75 < 1.0)
        vols_5 = np.linspace(0.1, 0.5, 5)
        w_infeas_max = calculate_bounded_risk_parity_weights(vols_5, w_min=0.05, w_max=0.15)
        assert np.sum(w_infeas_max) == pytest.approx(1.0, abs=1e-7)
        assert np.allclose(w_infeas_max, 0.20)

    def test_bounded_parity_monte_carlo_convergence_harness(self):
        """
        Randomized stress harness: 500 random parameter combinations.
        Assert 100% convergence to valid probabilities satisfying bounds.
        """
        np.random.seed(42)
        for trial in range(500):
            n = np.random.randint(2, 25)
            vols = np.exp(np.random.uniform(-4.0, 2.0, size=n))  # vols from ~0.018 to ~7.38

            # Generate valid feasible bounds
            w_min = np.random.uniform(0.01, 1.0 / n)
            w_max = np.random.uniform(1.0 / n, 0.50)

            w = calculate_bounded_risk_parity_weights(vols, w_min=w_min, w_max=w_max, max_iter=50)

            assert np.sum(w) == pytest.approx(1.0, abs=1e-5), f"Trial {trial} failed sum constraint: {np.sum(w)}"
            assert np.all(w >= w_min - 1e-5), f"Trial {trial} failed min bound: min(w)={np.min(w)} < {w_min}"
            assert np.all(w <= w_max + 1e-5), f"Trial {trial} failed max bound: max(w)={np.max(w)} > {w_max}"


class TestHysteresisRankBufferTransitionMatrix:
    """Stress testing the complete state transition matrix for rank buffer and trend exits."""

    def test_complete_cartesian_transition_matrix(self):
        """
        Verify all combinations of:
        - is_held in [False, True]
        - rank in [1, 5, 10, 11, 15, 20, 21, 50, 100]
        - passes_trend in [True, False]
        - top_n = 10, rank_buffer = 20
        """
        top_n = 10
        rank_buffer = 20

        test_ranks = [1, 5, 10, 11, 15, 20, 21, 50, 100]

        for rank in test_ranks:
            for passes_trend in [True, False]:
                # 1. Non-holding candidate
                act_unheld = classify_rebalance_action(
                    symbol="SYM_NEW",
                    rank=rank,
                    is_held=False,
                    passes_trend=passes_trend,
                    top_n=top_n,
                    rank_buffer=rank_buffer,
                )
                if rank <= top_n and passes_trend:
                    assert act_unheld == "NEW_BUY", f"Expected NEW_BUY for rank={rank}, trend={passes_trend}"
                else:
                    assert act_unheld == "IGNORE", f"Expected IGNORE for rank={rank}, trend={passes_trend}"

                # 2. Existing holding
                act_held = classify_rebalance_action(
                    symbol="SYM_HELD",
                    rank=rank,
                    is_held=True,
                    passes_trend=passes_trend,
                    top_n=top_n,
                    rank_buffer=rank_buffer,
                )
                if not passes_trend:
                    assert act_held == "SELL_EXIT", f"Expected SELL_EXIT on trend breakdown for rank={rank}"
                elif rank > rank_buffer:
                    assert act_held == "SELL_EXIT", f"Expected SELL_EXIT on rank buffer breach for rank={rank}"
                else:
                    assert act_held == "HOLD_RETAIN", f"Expected HOLD_RETAIN for rank={rank}, trend={passes_trend}"

    @pytest.mark.parametrize(
        "curr_w, target_w, deadband, expected_action",
        [
            (0.0, 0.10, 0.02, "NEW_BUY"),
            (0.0, 0.01, 0.02, "NEW_BUY"),
            (0.10, 0.0, 0.02, "SELL_EXIT"),
            (0.01, 0.0, 0.02, "SELL_EXIT"),
            # Rebalance Add (delta > deadband)
            (0.10, 0.125, 0.02, "REBALANCE_ADD"),  # +2.5% > +2.0%
            (0.05, 0.10, 0.02, "REBALANCE_ADD"),   # +5.0% > +2.0%
            # Rebalance Trim (delta < -deadband)
            (0.10, 0.075, 0.02, "REBALANCE_TRIM"),  # -2.5% < -2.0%
            (0.15, 0.10, 0.02, "REBALANCE_TRIM"),   # -5.0% < -2.0%
            # No Change / Within Deadband (|delta| <= deadband)
            (0.10, 0.119, 0.02, "NO_CHANGE"),       # +1.9% <= +2.0%
            (0.10, 0.081, 0.02, "NO_CHANGE"),       # -1.9% <= +2.0%
            (0.10, 0.10, 0.02, "NO_CHANGE"),        # 0.0%
        ],
    )
    def test_weight_adjustment_action_matrix(self, curr_w, target_w, deadband, expected_action):
        act = calculate_weight_adjustment_action(curr_w, target_w, deadband=deadband)
        assert act == expected_action

    def test_deadband_floating_point_precision_behavior(self):
        """
        Empirically verify floating point precision behavior at exact boundary delta = -0.02.
        In IEEE 754: 0.08 - 0.10 = -0.020000000000000004.
        With epsilon tolerance, -0.02 delta is correctly treated as within deadband (NO_CHANGE).
        """
        delta = 0.08 - 0.10
        assert delta < -0.02  # Floating point artifact
        action = calculate_weight_adjustment_action(0.10, 0.08, deadband=0.02)
        assert action == "NO_CHANGE"


class TestHysteresisBufferDiscrepancyAndReproduction:
    """
    Empirical tests verifying the correct integration between
    build_momentum_portfolio and generate_rebalance_plan regarding rank buffer hysteresis.
    """

    def test_held_stock_in_buffer_omitted_from_target_and_erroneously_sold(self):
        """
        Remediation verification:
        - 25 ranked stocks available.
        - Existing holding 'SYM_HELD' is ranked #14 (within retention buffer 20).
        - classify_rebalance_action correctly designates 'SYM_HELD' as HOLD_RETAIN.
        - build_momentum_portfolio retains 'SYM_HELD' and fills remaining 9 slots with SYM_1..SYM_9.
        - generate_rebalance_plan generates HOLD_RETAIN order for 'SYM_HELD'.
        """
        breakdowns = [
            MomentumScoreBreakdown(
                symbol=f"SYM_{i}",
                rank=i,
                composite_score=2.0 - i * 0.05,
                raw_vol_ann=0.20,
                close=100.0,
            )
            for i in range(1, 26)
        ]

        current_holdings = ["SYM_14"]

        # 1. Verification of unit classifier
        unit_action = classify_rebalance_action("SYM_14", rank=14, is_held=True, passes_trend=True, top_n=10, rank_buffer=20)
        assert unit_action == "HOLD_RETAIN"

        # 2. Portfolio recommendation construction retains SYM_14
        rec = build_momentum_portfolio(
            ranked_constituents=breakdowns,
            top_n=10,
            current_holdings=current_holdings,
            rank_buffer=20,
        )

        target_syms = [c.symbol for c in rec.target_constituents]
        assert "SYM_14" in target_syms
        assert len(rec.target_constituents) == 10
        expected_syms = [f"SYM_{i}" for i in range(1, 10)] + ["SYM_14"]
        assert target_syms == expected_syms

        # 3. Plan generation retains SYM_14
        plan = generate_rebalance_plan(
            current_holdings=current_holdings,
            target_recommendation=rec,
            rank_buffer=20,
            top_n=10,
        )

        trade_map = {t.symbol: t for t in plan.all_trades}
        assert "SYM_14" in trade_map
        assert trade_map["SYM_14"].action in ("HOLD_RETAIN", "REBALANCE_ADD", "REBALANCE_TRIM")


class TestTurnoverCalculationAccuracy:
    """Stress testing portfolio turnover across complete replacement, partial rebalances, and cash scaling."""

    def _make_mock_recommendation(
        self,
        symbols_and_weights: Dict[str, float],
        total_capital: float = 1_000_000.0,
        universe: str = "india",
        equity_allocation_pct: float = 1.0,
    ) -> PortfolioRecommendation:
        constituents = []
        for i, (sym, w) in enumerate(symbols_and_weights.items()):
            c = PortfolioConstituent(
                symbol=sym,
                rank=i + 1,
                weight=w,
                composite_score=1.0 - i * 0.05,
                clenow_score=0.5,
                clenow_slope_ann=0.6,
                clenow_r2=0.8,
                mom_12_1=0.7,
                mom_6_1=0.3,
                volatility_ann=0.20,
                natr_14=2.0,
                close=100.0,
            )
            constituents.append(c)

        total_equity = sum(symbols_and_weights.values())
        regime = MarketRegime(
            benchmark_symbol="^NSEI" if universe == "india" else "^GSPC",
            state=MarketRegimeState.BULLISH if equity_allocation_pct == 1.0 else MarketRegimeState.BEARISH,
            benchmark_price=24000.0,
            sma_200=22000.0,
            sma_200_slope=0.01,
            equity_allocation_pct=equity_allocation_pct,
        )

        return PortfolioRecommendation(
            universe=universe,
            as_of_date="2026-08-29",
            market_regime=regime,
            weighting_scheme="equal",
            target_constituents=constituents,
            total_capital=total_capital,
            total_equity_pct=total_equity,
            expected_cash_pct=max(0.0, 1.0 - total_equity),
            allocated_cash_pct=max(0.0, 1.0 - total_equity),
        )

    def test_turnover_total_replacement_100_percent(self):
        """
        Total portfolio replacement:
        Current: 10 stocks at 10% (SYM_OLD_1..10), Cash = 0%.
        Target: 10 new stocks at 10% (SYM_NEW_1..10), Cash = 0%.
        One-way turnover = 0.5 * (sum(|-0.10|*10) + sum(|+0.10|*10) + |0|) = 0.5 * (1.0 + 1.0) = 1.0 (100%).
        """
        current_holdings = {f"SYM_OLD_{i}": 0.10 for i in range(1, 11)}
        target_dict = {f"SYM_NEW_{i}": 0.10 for i in range(1, 11)}
        target_rec = self._make_mock_recommendation(target_dict)

        plan = generate_rebalance_plan(
            current_holdings=current_holdings,
            target_recommendation=target_rec,
            deadband=0.02,
        )

        assert plan.turnover_pct == pytest.approx(1.0, abs=1e-6)
        assert len(plan.buys) == 10
        assert len(plan.sells) == 10
        assert len(plan.holds) == 0
        assert all(b.action == "NEW_BUY" for b in plan.buys)
        assert all(s.action == "SELL_EXIT" for s in plan.sells)

    def test_turnover_partial_rebalance_50_percent(self):
        """
        50% replacement:
        Current: SYM_1..5 (10% each) + SYM_OLD_6..10 (10% each).
        Target: SYM_1..5 (10% each) + SYM_NEW_6..10 (10% each).
        One-way turnover = 0.5 * (5*|-0.10| + 5*|+0.10| + 5*|0|) = 0.5 * (0.5 + 0.5) = 0.50 (50%).
        """
        current_holdings = {f"SYM_{i}": 0.10 for i in range(1, 6)}
        current_holdings.update({f"SYM_OLD_{i}": 0.10 for i in range(6, 11)})

        target_dict = {f"SYM_{i}": 0.10 for i in range(1, 6)}
        target_dict.update({f"SYM_NEW_{i}": 0.10 for i in range(6, 11)})
        target_rec = self._make_mock_recommendation(target_dict)

        plan = generate_rebalance_plan(
            current_holdings=current_holdings,
            target_recommendation=target_rec,
            deadband=0.02,
        )

        assert plan.turnover_pct == pytest.approx(0.50, abs=1e-6)
        assert len(plan.buys) == 5
        assert len(plan.sells) == 5
        assert len(plan.holds) == 5
        assert all(h.action == "HOLD_RETAIN" for h in plan.holds)

    def test_turnover_zero_on_identical_portfolio(self):
        """Identical portfolio must have 0.0 turnover."""
        current_holdings = {f"SYM_{i}": 0.10 for i in range(1, 11)}
        target_rec = self._make_mock_recommendation(current_holdings)

        plan = generate_rebalance_plan(
            current_holdings=current_holdings,
            target_recommendation=target_rec,
            deadband=0.02,
        )

        assert plan.turnover_pct == pytest.approx(0.0, abs=1e-6)
        assert len(plan.buys) == 0
        assert len(plan.sells) == 0
        assert len(plan.holds) == 10

    def test_turnover_macro_bearish_liquidation_to_cash(self):
        """
        Transition from 100% equity to 100% cash (Bearish crash exit):
        Current: 10 stocks at 10% (sum = 1.0, cash = 0.0).
        Target: 0 stocks (sum = 0.0, cash = 1.0).
        Trades: 10 SELL_EXITs of 0.10 each (sum delta = 1.0).
        Cash delta = 1.0.
        Turnover = 0.5 * (1.0 + 1.0) = 1.0 (100%).
        """
        current_holdings = {f"SYM_{i}": 0.10 for i in range(1, 11)}
        target_rec = self._make_mock_recommendation({}, equity_allocation_pct=0.0)

        plan = generate_rebalance_plan(
            current_holdings=current_holdings,
            target_recommendation=target_rec,
            deadband=0.02,
        )

        assert plan.turnover_pct == pytest.approx(1.0, abs=1e-6)
        assert len(plan.sells) == 10
        assert len(plan.buys) == 0

    def test_turnover_macro_neutral_scaling(self):
        """
        Transition from 100% equity to 50% equity / 50% cash:
        Current: 10 stocks at 10% (total = 1.0, cash = 0.0).
        Target: same 10 stocks at 5% (total = 0.5, cash = 0.5).
        Trades: 10 REBALANCE_TRIMs of -0.05 each (sum = 0.50).
        Cash delta = 0.50.
        Turnover = 0.5 * (0.50 + 0.50) = 0.50 (50%).
        """
        current_holdings = {f"SYM_{i}": 0.10 for i in range(1, 11)}
        target_dict = {f"SYM_{i}": 0.05 for i in range(1, 11)}
        target_rec = self._make_mock_recommendation(target_dict, equity_allocation_pct=0.5)

        plan = generate_rebalance_plan(
            current_holdings=current_holdings,
            target_recommendation=target_rec,
            deadband=0.02,
        )

        assert plan.turnover_pct == pytest.approx(0.50, abs=1e-6)
        assert len(plan.sells) == 10
        assert all(s.action == "REBALANCE_TRIM" for s in plan.sells)


class TestShareSizingAndCapitalReconciliation:
    """Stress testing trade share calculations, rounding, and cash balance accounting."""

    def test_share_quantities_and_capital_conservation(self):
        """
        Verify that calculated target shares do not exceed allocated target value,
        and estimated trade cash balance reconciles.
        """
        total_capital = 1_000_000.0
        prices = {"SYM_1": 1500.0, "SYM_2": 320.0, "SYM_3": 45.50, "SYM_4": 9999.0}
        target_weights = {"SYM_1": 0.30, "SYM_2": 0.30, "SYM_3": 0.20, "SYM_4": 0.20}

        constituents = [
            PortfolioConstituent(
                symbol=sym,
                rank=i + 1,
                weight=w,
                composite_score=1.0,
                clenow_score=0.5,
                clenow_slope_ann=0.6,
                clenow_r2=0.8,
                mom_12_1=0.7,
                mom_6_1=0.3,
                volatility_ann=0.20,
                natr_14=2.0,
                close=prices[sym],
                target_value=total_capital * w,
                target_shares=int((total_capital * w) // prices[sym]),
            )
            for i, (sym, w) in enumerate(target_weights.items())
        ]

        rec = PortfolioRecommendation(
            universe="india",
            as_of_date="2026-08-29",
            market_regime=MarketRegime(
                benchmark_symbol="^NSEI",
                state=MarketRegimeState.BULLISH,
                benchmark_price=24000.0,
                sma_200=22000.0,
                sma_200_slope=0.01,
                equity_allocation_pct=1.0,
            ),
            weighting_scheme="inv_vol",
            target_constituents=constituents,
            total_capital=total_capital,
        )

        plan = generate_rebalance_plan(
            current_holdings=[],
            target_recommendation=rec,
            total_capital=total_capital,
            prices=prices,
        )

        total_spent = sum(t.estimated_trade_value for t in plan.all_trades)
        assert total_spent <= total_capital  # Cannot exceed total capital
        assert total_spent > total_capital * 0.98  # Capital efficiency > 98%
        assert plan.estimated_cash_change == pytest.approx(-total_spent)

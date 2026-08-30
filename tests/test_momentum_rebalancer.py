"""
Unit Tests for Momentum Portfolio Rebalancing & Hysteresis Engine.
"""

import pytest

from src.momentum.models import (
    MarketRegime,
    MarketRegimeState,
    MomentumScoreBreakdown,
    PortfolioConstituent,
    PortfolioRecommendation,
    RebalanceAction,
)
from src.momentum.rebalancer import (
    calculate_weight_adjustment_action,
    classify_rebalance_action,
    generate_rebalance_plan,
)


class TestClassifyRebalanceAction:
    """Tests for classify_rebalance_action with hysteresis buffer and trend breakdown."""

    def test_new_entrant_qualifies_for_buy(self):
        act = classify_rebalance_action("NEW_SYM", rank=5, is_held=False, passes_trend=True, top_n=10)
        assert act == "NEW_BUY"

    def test_new_entrant_outside_top_n_ignored(self):
        act = classify_rebalance_action("NEW_SYM", rank=14, is_held=False, passes_trend=True, top_n=10)
        assert act == "IGNORE"

    def test_new_entrant_failing_trend_ignored(self):
        act = classify_rebalance_action("NEW_SYM", rank=3, is_held=False, passes_trend=False, top_n=10)
        assert act == "IGNORE"

    def test_held_stock_retained_in_top_n(self):
        act = classify_rebalance_action("HELD_SYM", rank=4, is_held=True, passes_trend=True, top_n=10)
        assert act == "HOLD_RETAIN"

    def test_held_stock_retained_in_buffer(self):
        # Rank 16 is between top_n 10 and buffer 20 -> retained
        act = classify_rebalance_action(
            "HELD_SYM", rank=16, is_held=True, passes_trend=True, top_n=10, rank_buffer=20
        )
        assert act == "HOLD_RETAIN"

    def test_held_stock_exits_when_beyond_buffer(self):
        # Rank 25 exceeds buffer 20 -> liquidated
        act = classify_rebalance_action(
            "HELD_SYM", rank=25, is_held=True, passes_trend=True, top_n=10, rank_buffer=20
        )
        assert act == "SELL_EXIT"

    def test_held_stock_immediate_exit_on_trend_breakdown(self):
        # Even with high rank (#2), failing trend forces immediate exit
        act = classify_rebalance_action(
            "HELD_SYM", rank=2, is_held=True, passes_trend=False, top_n=10, rank_buffer=20
        )
        assert act == "SELL_EXIT"


class TestWeightAdjustmentAction:
    """Tests for deadband weight adjustment classification."""

    def test_new_buy_and_sell_exit(self):
        assert calculate_weight_adjustment_action(0.0, 0.10, deadband=0.02) == "NEW_BUY"
        assert calculate_weight_adjustment_action(0.10, 0.0, deadband=0.02) == "SELL_EXIT"

    def test_rebalance_add_and_trim(self):
        # +5% exceeds 2% deadband
        assert calculate_weight_adjustment_action(0.10, 0.15, deadband=0.02) == "REBALANCE_ADD"
        # -6% exceeds 2% deadband
        assert calculate_weight_adjustment_action(0.15, 0.09, deadband=0.02) == "REBALANCE_TRIM"

    def test_no_change_within_deadband(self):
        # +1% is within 2% deadband
        assert calculate_weight_adjustment_action(0.10, 0.11, deadband=0.02) == "NO_CHANGE"
        # -1.5% is within 2% deadband
        assert calculate_weight_adjustment_action(0.10, 0.085, deadband=0.02) == "NO_CHANGE"


class TestGenerateRebalancePlan:
    """Tests for generate_rebalance_plan."""

    @pytest.fixture
    def mock_recommendation(self):
        constituents = [
            PortfolioConstituent(
                symbol="SYM_1",
                rank=1,
                weight=0.15,
                composite_score=1.5,
                clenow_score=0.5,
                clenow_slope_ann=0.6,
                clenow_r2=0.8,
                mom_12_1=0.7,
                mom_6_1=0.3,
                volatility_ann=0.15,
                natr_14=2.0,
                close=100.0,
            ),
            PortfolioConstituent(
                symbol="SYM_2",
                rank=2,
                weight=0.10,
                composite_score=1.3,
                clenow_score=0.4,
                clenow_slope_ann=0.5,
                clenow_r2=0.8,
                mom_12_1=0.6,
                mom_6_1=0.3,
                volatility_ann=0.20,
                natr_14=2.5,
                close=50.0,
            ),
            PortfolioConstituent(
                symbol="SYM_3",
                rank=3,
                weight=0.10,
                composite_score=1.1,
                clenow_score=0.35,
                clenow_slope_ann=0.45,
                clenow_r2=0.75,
                mom_12_1=0.5,
                mom_6_1=0.25,
                volatility_ann=0.25,
                natr_14=3.0,
                close=200.0,
            ),
        ]
        regime = MarketRegime(
            benchmark_symbol="^NSEI",
            state=MarketRegimeState.BULLISH,
            benchmark_price=24000.0,
            sma_200=22000.0,
            sma_200_slope=0.01,
            equity_allocation_pct=1.0,
        )
        return PortfolioRecommendation(
            universe="india",
            as_of_date="2026-08-28",
            market_regime=regime,
            weighting_scheme="inv_vol",
            target_constituents=constituents,
            total_capital=1_000_000.0,
        )

    def test_rebalance_plan_initial_portfolio_all_buys(self, mock_recommendation):
        plan = generate_rebalance_plan(
            current_holdings=[],
            target_recommendation=mock_recommendation,
            total_capital=1_000_000.0,
        )

        assert len(plan.buys) == 3
        assert len(plan.sells) == 0
        assert len(plan.holds) == 0
        assert all(b.action == "NEW_BUY" for b in plan.buys)
        # Sum of weights in mock recommendation = 0.35 -> turnover = 0.35
        assert plan.turnover_pct == pytest.approx(0.35)

    def test_rebalance_plan_transition_with_holds_and_sells(self, mock_recommendation):
        # Current holdings: SYM_2 (held at 10%), SYM_OLD (held at 20%)
        current_weights = {"SYM_2": 0.10, "SYM_OLD": 0.20}
        plan = generate_rebalance_plan(
            current_holdings=current_weights,
            target_recommendation=mock_recommendation,
            deadband=0.02,
            total_capital=1_000_000.0,
        )

        actions = {t.symbol: t.action for t in plan.all_trades}

        # SYM_1 is not held -> NEW_BUY
        assert actions["SYM_1"] == "NEW_BUY"
        # SYM_2 is held at 10% and target is 10% (delta 0.0 <= deadband) -> HOLD_RETAIN
        assert actions["SYM_2"] == "HOLD_RETAIN"
        # SYM_3 is not held -> NEW_BUY
        assert actions["SYM_3"] == "NEW_BUY"
        # SYM_OLD is held but not in target -> SELL_EXIT
        assert actions["SYM_OLD"] == "SELL_EXIT"

        assert len(plan.sells) == 1
        assert plan.sells[0].symbol == "SYM_OLD"
        assert plan.sells[0].action == "SELL_EXIT"

    def test_rebalance_plan_trade_shares_calculation(self, mock_recommendation):
        prices = {"SYM_1": 100.0, "SYM_2": 50.0, "SYM_3": 200.0}
        plan = generate_rebalance_plan(
            current_holdings=[],
            target_recommendation=mock_recommendation,
            total_capital=1_000_000.0,
            prices=prices,
        )

        sym1_trade = [t for t in plan.all_trades if t.symbol == "SYM_1"][0]
        # Target weight 0.15 * 1,000,000 = 150,000 / 100.0 = 1500 shares
        assert sym1_trade.target_shares == 1500
        assert sym1_trade.trade_shares == 1500
        assert sym1_trade.estimated_trade_value == 150_000.0

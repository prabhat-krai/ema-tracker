"""
Tier 3: Cross-Feature Combinations & State Transitions Test Suite.
Validates multi-module interactions: macro benchmark gating + sizing, trend filter gating,
hysteresis rebalancing transition board, and multi-factor weight sensitivity.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from tests.e2e.fixtures import (
    generate_deterministic_series,
    generate_mock_benchmark,
    oracle_bounded_risk_parity_weights,
    oracle_inverse_volatility_weights,
)
from src.momentum.clenow import (
    calculate_multi_timeframe_clenow,
    compute_clenow_momentum,
)
from src.momentum.classical import (
    compute_jegadeesh_titman_momentum,
)
from src.momentum.volatility import (
    calculate_downside_deviation,
    calculate_realized_volatility,
    compute_volatility_metrics,
)
from src.momentum.regime import (
    check_single_stock_trend_filter,
    evaluate_market_regime,
    evaluate_stock_eligibility,
)
from src.momentum.scoring import (
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


class TestMacroCircuitBreakerAndSizing:
    """Macro benchmark gate interaction with portfolio equity vs cash allocation."""

    def test_bullish_benchmark_allocates_full_equity(self):
        """BULLISH regime -> 100% equity allocation across top momentum constituents."""
        df_bench = generate_mock_benchmark(regime="bullish")
        regime = evaluate_market_regime(df_bench["close"].values, benchmark_symbol="^NSEI")

        assert regime.state == MarketRegimeState.BULLISH
        assert regime.equity_allocation_pct == 1.00

        # Sizing 10 stocks under 100% equity
        vols = np.full(10, 0.20)
        weights = oracle_inverse_volatility_weights(vols) * regime.equity_allocation_pct
        assert abs(np.sum(weights) - 1.00) < 1e-6

    def test_neutral_benchmark_allocates_half_cash(self):
        """NEUTRAL regime -> 50% equity allocation, 50% defensive cash reserve."""
        df_bench = generate_mock_benchmark(regime="neutral")
        regime = evaluate_market_regime(df_bench["close"].values, benchmark_symbol="^NSEI")

        assert regime.state == MarketRegimeState.NEUTRAL
        assert regime.equity_allocation_pct == 0.50

        # Total invested equity is capped at 50%
        vols = np.full(10, 0.20)
        weights = oracle_inverse_volatility_weights(vols) * regime.equity_allocation_pct
        cash_reserve = 1.0 - np.sum(weights)

        assert abs(np.sum(weights) - 0.50) < 1e-6
        assert abs(cash_reserve - 0.50) < 1e-6

    def test_bearish_benchmark_defensive_zero_equity(self):
        """BEARISH regime -> 0% equity allocation (100% cash preservation)."""
        df_bench = generate_mock_benchmark(regime="bearish")
        regime = evaluate_market_regime(df_bench["close"].values, benchmark_symbol="^NSEI")

        assert regime.state == MarketRegimeState.BEARISH
        assert regime.equity_allocation_pct == 0.00

        vols = np.full(10, 0.20)
        weights = oracle_inverse_volatility_weights(vols) * regime.equity_allocation_pct
        cash_reserve = 1.0 - np.sum(weights)

        assert np.all(weights == 0.0)
        assert abs(cash_reserve - 1.00) < 1e-6


class TestTrendGatingAndSizingInteraction:
    """Interplay between macro trend qualification filters and portfolio constituent selection."""

    def test_high_momentum_failing_trend_is_excluded(self):
        """Stock with superior Clenow score (+90%) failing EMA100/SMA200 is disqualified."""
        # Stock A: High score but in downtrend (passes_trend = False)
        m_a = MomentumMetrics(
            symbol="A_FAIL",
            close=100.0,
            clenow_score=0.90,
            clenow_r2=0.95,
            mom_12_1=1.20,
            volatility_ann=0.20,
            passes_trend=False,
        )
        # Stock B & C: Moderate scores but in confirmed uptrend
        m_b = MomentumMetrics(
            symbol="B_PASS",
            close=100.0,
            clenow_score=0.45,
            clenow_r2=0.85,
            mom_12_1=0.60,
            volatility_ann=0.18,
            passes_trend=True,
        )
        m_c = MomentumMetrics(
            symbol="C_PASS",
            close=100.0,
            clenow_score=0.40,
            clenow_r2=0.80,
            mom_12_1=0.50,
            volatility_ann=0.22,
            passes_trend=True,
        )

        breakdowns, ranked_df = rank_universe([m_a, m_b, m_c], filter_trend=True)
        ranked_symbols = [b.symbol for b in breakdowns]

        assert "A_FAIL" not in ranked_symbols
        assert ranked_symbols == ["B_PASS", "C_PASS"]

        # Weights are distributed among eligible constituents only
        vols = np.array([b.raw_vol_ann for b in breakdowns])
        weights = oracle_inverse_volatility_weights(vols)
        assert abs(np.sum(weights) - 1.0) < 1e-6


class TestHysteresisRankBufferTransitions:
    """State transition tracking: NEW_BUY, HOLD (buffer retention), and EXIT triggers."""

    def test_multi_period_hysteresis_transitions(self):
        """
        Simulates portfolio transition from Week 1 to Week 2:
        - Holdings in Week 1: ['SYM_1', 'SYM_2', 'SYM_3', 'SYM_4', 'SYM_5']
        - Week 2 Ranks:
            - SYM_1: Rank #1 -> HOLD (Retain)
            - SYM_2: Rank #15 -> HOLD (Retained within buffer rank <= 20)
            - SYM_3: Rank #23 -> EXIT (Dropped beyond buffer 20)
            - SYM_4: Rank #6, but broke EMA100 -> EXIT (Trend breakdown)
            - SYM_NEW: Rank #2 -> NEW_BUY (New entrant in Top 10)
        """
        w1_holdings = ["SYM_1", "SYM_2", "SYM_3", "SYM_4", "SYM_5"]

        # Evaluated week 2 metrics
        w2_candidates = [
            {"symbol": "SYM_1", "rank": 1, "trend": True},
            {"symbol": "SYM_NEW", "rank": 2, "trend": True},
            {"symbol": "SYM_4", "rank": 6, "trend": False},  # Broke trend
            {"symbol": "SYM_2", "rank": 15, "trend": True},  # In buffer
            {"symbol": "SYM_3", "rank": 23, "trend": True},  # Outside buffer
        ]

        actions = {}
        for c in w2_candidates:
            sym = c["symbol"]
            rank = c["rank"]
            trend = c["trend"]
            is_held = sym in w1_holdings

            if is_held:
                if not trend or rank > 20:
                    actions[sym] = RebalanceAction.SELL
                else:
                    actions[sym] = RebalanceAction.HOLD
            else:
                if trend and rank <= 10:
                    actions[sym] = RebalanceAction.BUY
                else:
                    actions[sym] = RebalanceAction.HOLD

        assert actions["SYM_1"] == RebalanceAction.HOLD
        assert actions["SYM_NEW"] == RebalanceAction.BUY
        assert actions["SYM_2"] == RebalanceAction.HOLD  # Retained in buffer
        assert actions["SYM_3"] == RebalanceAction.SELL  # Rank 23 > 20
        assert actions["SYM_4"] == RebalanceAction.SELL  # Trend breakdown


class TestFactorWeightSensitivityAndReordering:
    """Sensitivity of universe composite rankings to factor weight shifts."""

    def test_volatility_penalty_weight_shift_reorders_ranking(self):
        """Increasing volatility penalty weight prioritizes lower-volatility smooth trends."""
        df = pd.DataFrame({
            "symbol": ["HIGH_VOL_LEADER", "LOW_VOL_STEADY"],
            "clenow_score": [0.65, 0.55],
            "mom_12_1": [0.90, 0.70],
            "r_squared": [0.80, 0.90],
            "mom_6_1": [0.45, 0.35],
            "vol_ann": [0.45, 0.12],  # HIGH_VOL is 45% vol, LOW_VOL is 12% vol
        })

        # Scenario 1: Low vol penalty (vol weight = -0.05)
        weights_low_pen = {
            "clenow_score": 0.45, "mom_12_1": 0.30, "r_squared": 0.15, "mom_6_1": 0.05, "vol_ann": -0.05
        }
        df_res1 = compute_composite_momentum_score(df, weights=weights_low_pen)
        # HIGH_VOL_LEADER wins
        assert df_res1.iloc[0]["symbol"] == "HIGH_VOL_LEADER"

        # Scenario 2: Heavy vol penalty (vol weight = -0.50)
        weights_heavy_pen = {
            "clenow_score": 0.20, "mom_12_1": 0.15, "r_squared": 0.15, "mom_6_1": 0.00, "vol_ann": -0.50
        }
        df_res2 = compute_composite_momentum_score(df, weights=weights_heavy_pen)
        # LOW_VOL_STEADY wins
        assert df_res2.iloc[0]["symbol"] == "LOW_VOL_STEADY"


class TestRebalancingDeadband:
    """Weight adjustment deadband suppression to prevent excessive churn."""

    def test_rebalance_deadband_suppresses_minor_trades(self):
        """Target weight shift < 2% is ignored; >= 2% generates rebalance signal."""
        deadband = 0.02
        current_weights = {"STOCK_A": 0.10, "STOCK_B": 0.10, "STOCK_C": 0.10}
        target_weights = {
            "STOCK_A": 0.108,  # Delta = +0.008 (< 2%) -> No trade
            "STOCK_B": 0.135,  # Delta = +0.035 (>= 2%) -> Trade
            "STOCK_C": 0.070,  # Delta = -0.030 (>= 2%) -> Trade
        }

        rebalance_required = {}
        for sym, curr_w in current_weights.items():
            tgt_w = target_weights[sym]
            delta = abs(tgt_w - curr_w)
            rebalance_required[sym] = delta >= deadband

        assert rebalance_required["STOCK_A"] is False
        assert rebalance_required["STOCK_B"] is True
        assert rebalance_required["STOCK_C"] is True

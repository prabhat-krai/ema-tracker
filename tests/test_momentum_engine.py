"""
Unit and Integration Tests for MomentumPipeline Engine.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from tests.e2e.fixtures import (
    generate_mock_benchmark,
    generate_mock_indian_universe,
    generate_mock_usa_universe,
)
from src.momentum.engine import MomentumPipeline, MomentumPipelineResult
from src.momentum.models import MarketRegimeState, RebalanceAction


class TestMomentumPipelineOffline:
    """Tests for MomentumPipeline execution using in-memory pre-loaded datasets."""

    def test_indian_universe_pipeline_execution(self):
        universe_data = generate_mock_indian_universe(n_stocks=30, seed=101)
        benchmark_df = generate_mock_benchmark(regime="bullish", symbol="^NSEI")

        pipeline = MomentumPipeline(
            universe="india",
            top_n=10,
            weighting_scheme="inv_vol",
            total_capital=1_000_000.0,
        )

        result = pipeline.run_with_data(
            universe_data=universe_data,
            benchmark_df=benchmark_df,
            benchmark_symbol="^NSEI",
            current_holdings=[],
        )

        assert isinstance(result, MomentumPipelineResult)
        assert result.universe == "india"
        assert result.market_regime.state == MarketRegimeState.BULLISH
        assert result.market_regime.equity_allocation_pct == 1.0

        # Top 10 constituents
        assert len(result.top_constituents) == 10
        top_symbols = [c.symbol for c in result.top_constituents]

        # Verify known momentum leaders are selected
        assert "TRENT.NS" in top_symbols
        assert "DIXON.NS" in top_symbols
        assert "BEL.NS" in top_symbols

        # Verify downtrend stocks are disqualified
        assert "INFY.NS" not in top_symbols
        assert "HDFCBANK.NS" not in top_symbols

        # Sizing and cash allocation
        assert result.portfolio.total_equity_pct == pytest.approx(1.0, abs=1e-6)
        assert result.portfolio.expected_cash_pct == pytest.approx(0.0, abs=1e-6)

        # Rebalancing plan
        assert len(result.rebalance_plan.buys) == 10
        assert all(b.action == "NEW_BUY" for b in result.rebalance_plan.buys)
        assert result.rebalance_plan.turnover_pct == pytest.approx(1.0, abs=1e-6)

    def test_usa_universe_pipeline_execution(self):
        universe_data = generate_mock_usa_universe(n_stocks=30, seed=202)
        benchmark_df = generate_mock_benchmark(regime="bullish", symbol="^GSPC")

        pipeline = MomentumPipeline(
            universe="usa",
            top_n=10,
            weighting_scheme="bounded_parity",
            min_weight=0.05,
            max_weight=0.20,
        )

        result = pipeline.run_with_data(
            universe_data=universe_data,
            benchmark_df=benchmark_df,
            benchmark_symbol="^GSPC",
            current_holdings=["NVDA", "INTC"],  # INTC is in downtrend, should exit
        )

        top_symbols = [c.symbol for c in result.top_constituents]
        assert "NVDA" in top_symbols
        assert "LLY" in top_symbols
        assert "INTC" not in top_symbols

        # Check bounded risk parity weights
        for c in result.top_constituents:
            assert c.weight >= 0.0499
            assert c.weight <= 0.2001

        # Check rebalancing transitions: NVDA should be HOLD, INTC should be SELL_EXIT
        actions = {t.symbol: t.action for t in result.rebalance_plan.all_trades}
        assert actions["NVDA"] in ("HOLD_RETAIN", "REBALANCE_ADD", "REBALANCE_TRIM")
        assert actions["INTC"] == "SELL_EXIT"

    def test_pipeline_neutral_regime_allocates_half_cash(self):
        universe_data = generate_mock_indian_universe(n_stocks=20, seed=102)
        benchmark_df = generate_mock_benchmark(regime="neutral", symbol="^NSEI")

        pipeline = MomentumPipeline(universe="india", top_n=10)
        result = pipeline.run_with_data(
            universe_data=universe_data,
            benchmark_df=benchmark_df,
        )

        assert result.market_regime.state == MarketRegimeState.NEUTRAL
        assert result.portfolio.total_equity_pct == pytest.approx(0.50, abs=1e-6)
        assert result.portfolio.expected_cash_pct == pytest.approx(0.50, abs=1e-6)

    def test_pipeline_bearish_regime_allocates_full_cash(self):
        universe_data = generate_mock_indian_universe(n_stocks=20, seed=103)
        benchmark_df = generate_mock_benchmark(regime="bearish", symbol="^NSEI")

        pipeline = MomentumPipeline(universe="india", top_n=10)
        result = pipeline.run_with_data(
            universe_data=universe_data,
            benchmark_df=benchmark_df,
        )

        assert result.market_regime.state == MarketRegimeState.BEARISH
        assert result.portfolio.total_equity_pct == pytest.approx(0.0, abs=1e-6)
        assert result.portfolio.expected_cash_pct == pytest.approx(1.0, abs=1e-6)

    def test_pipeline_empty_universe(self):
        pipeline = MomentumPipeline(universe="india")
        result = pipeline.run_with_data(universe_data={})

        assert len(result.top_constituents) == 0
        assert result.portfolio.total_equity_pct == 0.0
        assert result.portfolio.expected_cash_pct == 1.0


class TestMomentumPipelineLiveMock:
    """Tests for MomentumPipeline live run() with mocked data fetchers."""

    @patch("src.momentum.engine.fetch_universe_daily_data")
    @patch("src.momentum.engine.fetch_benchmark_daily_data")
    def test_pipeline_run_calls_fetchers(self, mock_fetch_bench, mock_fetch_univ):
        mock_fetch_bench.return_value = generate_mock_benchmark(regime="bullish")
        mock_fetch_univ.return_value = generate_mock_indian_universe(n_stocks=15, seed=104)

        pipeline = MomentumPipeline(universe="india", top_n=5)
        result = pipeline.run(period="1y", delay=0.0)

        assert mock_fetch_bench.called
        assert mock_fetch_univ.called
        assert len(result.top_constituents) == 5

    def test_pipeline_result_dictionary_access(self):
        universe_data = generate_mock_indian_universe(n_stocks=15, seed=105)
        benchmark_df = generate_mock_benchmark(regime="bullish")

        pipeline = MomentumPipeline(universe="india", top_n=5)
        result = pipeline.run_with_data(universe_data=universe_data, benchmark_df=benchmark_df)

        # Test dict indexing and to_dict()
        assert result["universe"] == "india"
        assert len(result["top_constituents"]) == 5
        d = result.to_dict()
        assert "portfolio" in d
        assert "rebalance_plan" in d
        assert "ranked_breakdowns" in d

"""
Unit Tests for Quantitative Momentum Multi-Format Exporter.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.momentum.exporter import (
    export_all_reports,
    export_portfolio_csv,
    export_portfolio_json,
    export_portfolio_markdown,
)
from src.momentum.models import (
    MarketRegime,
    MarketRegimeState,
    PortfolioConstituent,
    PortfolioRecommendation,
    RebalanceAction,
    RebalanceInstruction,
    RebalancePlan,
)


@pytest.fixture
def sample_recommendation():
    constituents = [
        PortfolioConstituent(
            symbol="RELIANCE.NS",
            rank=1,
            weight=0.15,
            composite_score=1.85,
            clenow_score=1.20,
            clenow_slope_ann=1.40,
            clenow_r2=0.86,
            mom_12_1=0.75,
            mom_6_1=0.35,
            volatility_ann=0.18,
            natr_14=2.1,
            close=2500.0,
            action=RebalanceAction.BUY,
            target_value=150000.0,
            target_shares=60,
        ),
        PortfolioConstituent(
            symbol="TCS.NS",
            rank=2,
            weight=0.12,
            composite_score=1.65,
            clenow_score=1.10,
            clenow_slope_ann=1.25,
            clenow_r2=0.88,
            mom_12_1=0.60,
            mom_6_1=0.30,
            volatility_ann=0.15,
            natr_14=1.8,
            close=3800.0,
            action=RebalanceAction.HOLD,
            target_value=120000.0,
            target_shares=31,
        ),
    ]

    regime = MarketRegime(
        benchmark_symbol="^NSEI",
        state=MarketRegimeState.BULLISH,
        benchmark_price=24500.0,
        sma_200=22000.0,
        sma_200_slope=0.012,
        equity_allocation_pct=1.0,
        evaluation_date="2026-08-30",
    )

    return PortfolioRecommendation(
        universe="india",
        as_of_date="2026-08-30",
        market_regime=regime,
        weighting_scheme="inv_vol",
        target_constituents=constituents,
        total_capital=1_000_000.0,
        total_equity_pct=0.27,
        expected_cash_pct=0.73,
        allocated_cash_pct=0.73,
        weighted_clenow_score=1.155,
        weighted_volatility=0.166,
        portfolio_volatility=0.075,
        diversification_ratio=1.92,
        summary={"top_n": 2, "constituents_count": 2},
    )


@pytest.fixture
def sample_rebalance_plan():
    trades = [
        RebalanceInstruction(
            symbol="RELIANCE.NS",
            action="NEW_BUY",
            current_weight=0.0,
            target_weight=0.15,
            weight_delta=0.15,
            current_shares=0,
            target_shares=60,
            trade_shares=60,
            price=2500.0,
            estimated_trade_value=150000.0,
            rank=1,
            reason="Rank #1 entrant into Top 2 momentum",
        ),
        RebalanceInstruction(
            symbol="TCS.NS",
            action="HOLD_RETAIN",
            current_weight=0.12,
            target_weight=0.12,
            weight_delta=0.0,
            current_shares=31,
            target_shares=31,
            trade_shares=0,
            price=3800.0,
            estimated_trade_value=0.0,
            rank=2,
            reason="Weight change (0.0%) within 2.0% deadband (retained)",
        ),
    ]

    return RebalancePlan(
        evaluation_date="2026-08-30",
        universe="india",
        total_capital=1_000_000.0,
        current_holdings=["TCS.NS"],
        target_holdings=["RELIANCE.NS", "TCS.NS"],
        buys=[trades[0]],
        holds=[trades[1]],
        sells=[],
        all_trades=trades,
        turnover_pct=0.15,
        estimated_cash_change=-150000.0,
        summary={"total_trades": 1, "new_buys_count": 1, "turnover_pct": 0.15},
    )


class TestPortfolioExporter:
    """Test suite for CSV, JSON, and Markdown exporters."""

    def test_export_portfolio_csv(self, tmp_path, sample_recommendation, sample_rebalance_plan):
        csv_file = tmp_path / "test_portfolio.csv"
        out_p = export_portfolio_csv(
            recommendation=sample_recommendation,
            rebalance_plan=sample_rebalance_plan,
            filepath=csv_file,
        )

        assert out_p.exists()
        df = pd.read_csv(out_p)
        assert len(df) == 2
        assert "Symbol" in df.columns
        assert "Rank" in df.columns
        assert "Weight_Pct" in df.columns
        assert "Clenow_Score" in df.columns
        assert df.iloc[0]["Symbol"] == "RELIANCE.NS"
        assert df.iloc[0]["Action"] == "BUY"
        assert df.iloc[1]["Symbol"] == "TCS.NS"

    def test_export_portfolio_json(self, tmp_path, sample_recommendation, sample_rebalance_plan):
        json_file = tmp_path / "test_portfolio.json"
        out_p = export_portfolio_json(
            recommendation=sample_recommendation,
            rebalance_plan=sample_rebalance_plan,
            filepath=json_file,
        )

        assert out_p.exists()
        with open(out_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "recommendation" in data
        assert "rebalance_plan" in data
        assert data["recommendation"]["universe"] == "india"
        assert len(data["recommendation"]["target_constituents"]) == 2
        assert data["rebalance_plan"]["turnover_pct"] == 0.15

    def test_export_portfolio_markdown(self, tmp_path, sample_recommendation, sample_rebalance_plan):
        md_file = tmp_path / "test_summary.md"
        out_p = export_portfolio_markdown(
            recommendation=sample_recommendation,
            rebalance_plan=sample_rebalance_plan,
            filepath=md_file,
        )

        assert out_p.exists()
        content = out_p.read_text(encoding="utf-8")
        assert "# 🏆 Quantitative Momentum Portfolio Summary" in content
        assert "RELIANCE.NS" in content
        assert "TCS.NS" in content
        assert "Market Regime" in content
        assert "Rebalancing & Execution Plan" in content
        assert "One-Way Turnover" in content

    def test_export_all_reports(self, tmp_path, sample_recommendation, sample_rebalance_plan):
        out_dir = tmp_path / "reports_all"
        res_map = export_all_reports(
            recommendation=sample_recommendation,
            rebalance_plan=sample_rebalance_plan,
            output_dir=out_dir,
            base_name="custom_momentum",
        )

        assert "csv" in res_map
        assert "json" in res_map
        assert "markdown" in res_map
        assert res_map["csv"].exists()
        assert res_map["json"].exists()
        assert res_map["markdown"].exists()

"""
Unit Tests for Streamlit Momentum UI Tab 4 Integration & Chart Generators.
"""

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
import plotly.express as px

from src.momentum.clenow import calculate_clenow_regression
from src.momentum.cli import _generate_synthetic_offline_universe
from src.momentum.engine import MomentumPipeline
from src.momentum.models import MarketRegimeState


class TestMomentumUIComponents:
    """Test suite verifying UI data transformation, Plotly chart builders, and Tab 4 pipelines."""

    def test_pipeline_execution_for_ui(self):
        """Verify pipeline execution format matches UI expectations."""
        u_data, b_df, b_sym = _generate_synthetic_offline_universe("india", n_stocks=20)
        pipeline = MomentumPipeline(
            universe="india",
            top_n=10,
            weighting_scheme="inv_vol",
            total_capital=1_000_000.0,
        )
        res = pipeline.run_with_data(
            universe_data=u_data,
            benchmark_df=b_df,
            benchmark_symbol=b_sym,
            current_holdings=["TCS.NS", "RELIANCE.NS"],
        )

        assert res.universe == "india"
        assert res.market_regime.state in (MarketRegimeState.BULLISH, MarketRegimeState.NEUTRAL, MarketRegimeState.BEARISH)
        assert len(res.portfolio.target_constituents) == 10
        assert res.portfolio.total_capital == 1_000_000.0
        assert res.rebalance_plan is not None

    def test_allocation_donut_chart_generation(self):
        """Verify Plotly donut chart generation for UI."""
        labels = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "Cash Reserve"]
        values = [0.15, 0.15, 0.10, 0.60]

        fig = px.pie(
            names=labels,
            values=values,
            hole=0.45,
            title="Portfolio Capital Allocation",
        )
        assert fig is not None
        assert len(fig.data) == 1
        assert fig.data[0].hole == 0.45

    def test_score_scatter_chart_generation(self):
        """Verify Plotly scatter chart generation for UI."""
        df = pd.DataFrame([
            {"Symbol": "RELIANCE.NS", "Composite_Score": 1.5, "Volatility_Ann": 18.0, "Clenow_Score": 1.2, "Weight": 15.0},
            {"Symbol": "TCS.NS", "Composite_Score": 1.2, "Volatility_Ann": 15.0, "Clenow_Score": 1.0, "Weight": 15.0},
        ])

        fig = px.scatter(
            df,
            x="Volatility_Ann",
            y="Composite_Score",
            size="Weight",
            text="Symbol",
            color="Clenow_Score",
        )
        assert fig is not None
        assert len(fig.data) == 1

    def test_clenow_exponential_fit_chart_generation(self):
        """Verify single stock exponential regression and EMA/SMA chart generation."""
        rng = np.random.RandomState(123)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq="B")
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.01, size=90)))

        alpha, beta, r2 = calculate_clenow_regression(close)
        t_axis = np.arange(len(close), dtype=np.float64)
        exp_fit = np.exp(alpha + beta * t_axis)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=close, mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=dates, y=exp_fit, mode="lines", name="Clenow Fit"))

        assert len(fig.data) == 2
        assert r2 >= 0.0

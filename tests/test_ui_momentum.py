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
from streamlit.testing.v1 import AppTest


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


from pathlib import Path

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


class TestStreamlitAppUI:
    """Headless Streamlit AppTest verification suite testing all UI features and tabs."""

    def test_app_initial_load_india(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert len(at.exception) == 0
        assert len(at.tabs) == 4
        assert at.tabs[0].label == "🚀 Weekly Action Hub (Transitions)"
        assert at.tabs[1].label == "🔍 Full Market Master Scanner"
        assert at.tabs[2].label == "📈 Stock Chart Analyzer & Backtester"
        assert at.tabs[3].label == "🏆 Quant Momentum Portfolio"

    def test_tab1_quick_inspect_selection(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        sb_alert = [s for s in at.selectbox if "Alerted Ticker" in s.label][0]
        assert len(sb_alert.options) > 0
        sb_alert.select(sb_alert.options[1]).run()
        assert len(at.exception) == 0

    def test_tab2_search_and_signal_filtering(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        search_input = [t for t in at.text_input if "Search Ticker" in t.label][0]
        search_input.input("RELIANCE").run()
        assert len(at.exception) == 0
        filtered_df = next(
            df.value for df in at.dataframe
            if "Signal" in df.value.columns and "Formatted Price" in df.value.columns
        )
        assert not filtered_df.empty
        assert any("RELIANCE" in sym for sym in filtered_df["Symbol"])

    def test_tab3_ticker_selection_and_slider(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        sb_stock = [s for s in at.selectbox if "ticker to analyze" in s.label][0]
        assert len(sb_stock.options) > 0
        sb_stock.select(sb_stock.options[1]).run()
        assert len(at.exception) == 0

        slider_years = at.slider[0]
        slider_years.set_value(3).run()
        assert len(at.exception) == 0

    def test_tab4_momentum_execution_and_exports(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        btn_run = [b for b in at.button if "Run Quantitative Momentum Screener" in b.label][0]
        btn_run.click().run()
        assert len(at.exception) == 0

        # Verify Tab 4 constituents table has 10 constituents
        constituents_df = next(
            df.value for df in at.dataframe
            if "Clenow Score" in df.value.columns
        )
        assert len(constituents_df) == 10
        assert "Clenow Score" in constituents_df.columns

        # Verify download buttons have valid endpoints and are enabled
        dl_btns = at.get("download_button")
        assert len(dl_btns) == 3
        for btn in dl_btns:
            assert not btn.proto.disabled
            assert len(btn.proto.url) > 0

    def test_sidebar_universe_switch_usa(self):
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        sidebar_market = at.sidebar.selectbox[0]
        sidebar_market.select("🇺🇸 USA (S&P 500)").run()
        assert len(at.exception) == 0
        assert at.sidebar.selectbox[0].value == "🇺🇸 USA (S&P 500)"

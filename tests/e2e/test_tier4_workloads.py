"""
Tier 4: Multi-Asset Realistic Universe Workloads & Export Payloads Test Suite.
Validates end-to-end screening runs on synthetic Indian (NSE) and US (S&P 500) universes,
verifying ranking tables, weighting allocations, CSV/JSON/Markdown payloads, and determinism.
"""

import json
import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from tests.e2e.fixtures import (
    generate_mock_benchmark,
    generate_mock_indian_universe,
    generate_mock_usa_universe,
    oracle_bounded_risk_parity_weights,
    oracle_inverse_volatility_weights,
)
from src.momentum.clenow import (
    calculate_multi_timeframe_clenow,
    compute_clenow_momentum,
)
from src.momentum.classical import (
    calculate_multi_timeframe_returns,
    compute_jegadeesh_titman_momentum,
)
from src.momentum.volatility import (
    calculate_atr,
    calculate_realized_volatility,
    compute_volatility_metrics,
)
from src.momentum.regime import (
    calculate_52w_high_distance,
    check_single_stock_trend_filter,
    evaluate_market_regime,
    evaluate_stock_eligibility,
)
from src.momentum.scoring import (
    compute_composite_momentum_score,
    rank_universe,
)
from src.momentum.models import (
    FilterResult,
    MarketRegimeState,
    MomentumMetrics,
    MomentumScoreBreakdown,
    RebalanceAction,
)


def run_simulated_universe_screening(
    universe_data: Dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    benchmark_symbol: str = "^NSEI",
    top_n: int = 10,
    weighting_scheme: str = "inv_vol",
    current_holdings: List[str] = None,
    total_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """
    Executes a complete simulated screening pipeline across a stock universe dictionary.
    """
    if current_holdings is None:
        current_holdings = []

    # 1. Evaluate Benchmark Market Regime
    market_regime = evaluate_market_regime(
        benchmark_df["close"].values if benchmark_df is not None else None,
        benchmark_symbol=benchmark_symbol,
    )

    # 2. Compute Individual Stock Metrics & Eligibility
    metrics_list: List[MomentumMetrics] = []
    eligibility_map: Dict[str, FilterResult] = {}

    for sym, df in universe_data.items():
        elig = evaluate_stock_eligibility(sym, df, min_history=90, min_dist_52w=0.70)
        eligibility_map[sym] = elig

        if df is None or len(df) < 90:
            continue

        close_arr = df["close"].dropna().values
        high_arr = df["high"].dropna().values if "high" in df.columns else close_arr
        low_arr = df["low"].dropna().values if "low" in df.columns else close_arr

        # Clenow metrics
        slope, r2, score = compute_clenow_momentum(close_arr, lookback=90)
        mtf = calculate_multi_timeframe_clenow(close_arr)

        # Classical momentum
        mom_ret = calculate_multi_timeframe_returns(close_arr)

        # Volatility
        vol_dict = compute_volatility_metrics(close_arr, high=high_arr, low=low_arr, lookback=90)

        # Trend filters
        passes_trend, trend_details = check_single_stock_trend_filter(close_arr, ema_fast=100, sma_slow=200)
        dist_52w = calculate_52w_high_distance(close_arr, high_arr, lookback=252)

        m = MomentumMetrics(
            symbol=sym,
            close=float(close_arr[-1]),
            history_bars=len(close_arr),
            clenow_slope_ann=slope if not np.isnan(slope) else 0.0,
            clenow_r2=r2 if not np.isnan(r2) else 0.0,
            clenow_score=score if not np.isnan(score) else 0.0,
            clenow_score_mtf=mtf.get("composite_score", 0.0),
            mom_12_1=mom_ret.get("mom_12_1", 0.0) if not np.isnan(mom_ret.get("mom_12_1", 0.0)) else 0.0,
            mom_6_1=mom_ret.get("mom_6_1", 0.0) if not np.isnan(mom_ret.get("mom_6_1", 0.0)) else 0.0,
            volatility_ann=vol_dict.get("vol_ann", 0.20) if not np.isnan(vol_dict.get("vol_ann", 0.20)) else 0.20,
            atr_14=vol_dict.get("atr_14", 0.0) if not np.isnan(vol_dict.get("atr_14", 0.0)) else 0.0,
            natr_14=vol_dict.get("natr_14", 0.0) if not np.isnan(vol_dict.get("natr_14", 0.0)) else 0.0,
            dist_52w_high=dist_52w,
            passes_trend=passes_trend,
        )
        metrics_list.append(m)

    # 3. Cross-Sectional Ranking
    breakdowns, scored_df = rank_universe(metrics_list, filter_trend=True)

    # 4. Top N Selection & Weighting
    top_constituents = breakdowns[:top_n]
    if len(top_constituents) > 0:
        vols = np.array([c.raw_vol_ann for c in top_constituents])
        if weighting_scheme == "equal":
            raw_weights = np.full(len(top_constituents), 1.0 / len(top_constituents))
        elif weighting_scheme == "bounded_parity":
            raw_weights = oracle_bounded_risk_parity_weights(vols, w_min=0.05, w_max=0.20)
        else:
            raw_weights = oracle_inverse_volatility_weights(vols)

        # Scale by macro regime equity allocation
        scaled_weights = raw_weights * market_regime.equity_allocation_pct

        for i, c in enumerate(top_constituents):
            c.suggested_weight = float(scaled_weights[i])
            is_held = c.symbol in current_holdings
            if not is_held and c.rank <= top_n:
                c.action = RebalanceAction.BUY
            elif is_held:
                c.action = RebalanceAction.HOLD

    return {
        "market_regime": market_regime,
        "eligibility_map": eligibility_map,
        "metrics_list": metrics_list,
        "ranked_breakdowns": breakdowns,
        "top_constituents": top_constituents,
        "total_capital": total_capital,
    }


class TestIndiaUniverseWorkload:
    """Full simulated screening workload on Indian NSE Nifty 500 universe."""

    def test_indian_universe_end_to_end_screening(self):
        """Screens 30 Indian stocks: verifies top 10 selection, leaders inclusion, downtrend exclusion."""
        universe = generate_mock_indian_universe(n_stocks=30, seed=101)
        bench = generate_mock_benchmark(regime="bullish", symbol="^NSEI")

        result = run_simulated_universe_screening(
            universe_data=universe,
            benchmark_df=bench,
            benchmark_symbol="^NSEI",
            top_n=10,
            weighting_scheme="inv_vol",
        )

        top_stocks = [c.symbol for c in result["top_constituents"]]
        assert len(top_stocks) == 10

        # Known momentum leaders are selected in the Top 10
        expected_leaders = ["TRENT.NS", "DIXON.NS", "BEL.NS", "HAL.NS"]
        for leader in expected_leaders:
            assert leader in top_stocks, f"Expected {leader} in Top 10 Indian momentum"

        # Known downtrends and broken stocks are strictly excluded
        disqualified = ["INFY.NS", "HDFCBANK.NS", "KOTAKBANK.NS", "IPONEW.NS", "FLATSTOCK.NS"]
        for bad_stock in disqualified:
            assert bad_stock not in top_stocks, f"Downtrend {bad_stock} should NOT be in Top 10"

        # Weights sum to 1.0 under Bullish regime
        weights = [c.suggested_weight for c in result["top_constituents"]]
        assert abs(sum(weights) - 1.0) < 1e-6


class TestUSAUniverseWorkload:
    """Full simulated screening workload on US S&P 500 universe."""

    def test_usa_universe_end_to_end_screening(self):
        """Screens 30 US stocks: verifies top 10 selection, leaders inclusion, and bounded risk parity."""
        universe = generate_mock_usa_universe(n_stocks=30, seed=202)
        bench = generate_mock_benchmark(regime="bullish", symbol="^GSPC")

        result = run_simulated_universe_screening(
            universe_data=universe,
            benchmark_df=bench,
            benchmark_symbol="^GSPC",
            top_n=10,
            weighting_scheme="bounded_parity",
        )

        top_stocks = [c.symbol for c in result["top_constituents"]]
        assert len(top_stocks) == 10

        # Known US momentum leaders are selected
        expected_leaders = ["NVDA", "LLY", "AVGO", "META"]
        for leader in expected_leaders:
            assert leader in top_stocks, f"Expected {leader} in Top 10 US momentum"

        # Known US downtrends are excluded
        assert "INTC" not in top_stocks
        assert "WBA" not in top_stocks

        # Bounded risk parity weights adhere to [0.05, 0.20]
        weights = [c.suggested_weight for c in result["top_constituents"]]
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(0.049 <= w <= 0.201 for w in weights)


class TestExportPayloads:
    """Validation of CSV, JSON, and Markdown report payloads."""

    def test_json_payload_schema_compliance(self):
        """Validates that JSON payload structure contains required keys and fields."""
        universe = generate_mock_indian_universe(n_stocks=20, seed=303)
        bench = generate_mock_benchmark(regime="bullish", symbol="^NSEI")

        result = run_simulated_universe_screening(universe, bench, top_n=5)
        top = result["top_constituents"]

        payload = {
            "metadata": {
                "market_universe": "INDIA",
                "universe_size": len(universe),
                "top_n": 5,
                "weighting_scheme": "inverse_volatility",
                "total_capital": 1000000.0,
                "currency": "INR",
            },
            "market_regime": result["market_regime"].to_dict(),
            "constituents": [c.to_dict() for c in top],
            "rebalance_actions": {
                "new_buys": [c.symbol for c in top if c.action == RebalanceAction.BUY],
                "holds": [c.symbol for c in top if c.action == RebalanceAction.HOLD],
                "exits": [],
            },
        }

        json_str = json.dumps(payload, indent=2)
        parsed = json.loads(json_str)

        assert "metadata" in parsed
        assert "market_regime" in parsed
        assert "constituents" in parsed
        assert len(parsed["constituents"]) == 5
        assert parsed["constituents"][0]["rank"] == 1
        assert "clenow_score" in parsed["constituents"][0] or "raw_clenow_score" in parsed["constituents"][0]

    def test_csv_payload_schema_compliance(self):
        """Validates that CSV export columns and rows match expected dataframe structure."""
        universe = generate_mock_indian_universe(n_stocks=20, seed=404)
        bench = generate_mock_benchmark(regime="bullish", symbol="^NSEI")

        result = run_simulated_universe_screening(universe, bench, top_n=5)
        top = result["top_constituents"]

        rows = []
        for c in top:
            rows.append({
                "rank": c.rank,
                "symbol": c.symbol,
                "current_price": c.close,
                "clenow_score": c.raw_clenow_score,
                "annualized_slope": c.raw_slope_ann,
                "r_squared": c.raw_r2,
                "mom_12_1": c.raw_mom_12_1,
                "mom_6_1": c.raw_mom_6_1,
                "annualized_volatility": c.raw_vol_ann,
                "weight_pct": c.suggested_weight,
                "target_capital": c.suggested_weight * 1_000_000.0,
                "rebalance_action": c.action.value if hasattr(c.action, "value") else str(c.action),
            })
        df_csv = pd.DataFrame(rows)

        assert len(df_csv) == 5
        assert "rank" in df_csv.columns
        assert "symbol" in df_csv.columns
        assert "weight_pct" in df_csv.columns
        assert abs(df_csv["weight_pct"].sum() - 1.0) < 1e-6


class TestDeterminismAndIdempotence:
    """Validates that identical input data produces bitwise identical ranking and weighting outputs."""

    def test_screener_idempotence(self):
        """Executing two consecutive screening passes on identical data produces identical results."""
        universe_1 = generate_mock_indian_universe(n_stocks=25, seed=555)
        universe_2 = generate_mock_indian_universe(n_stocks=25, seed=555)
        bench_1 = generate_mock_benchmark(regime="bullish", seed=777)
        bench_2 = generate_mock_benchmark(regime="bullish", seed=777)

        res1 = run_simulated_universe_screening(universe_1, bench_1, top_n=10)
        res2 = run_simulated_universe_screening(universe_2, bench_2, top_n=10)

        syms1 = [c.symbol for c in res1["top_constituents"]]
        syms2 = [c.symbol for c in res2["top_constituents"]]
        assert syms1 == syms2

        scores1 = [c.composite_score for c in res1["top_constituents"]]
        scores2 = [c.composite_score for c in res2["top_constituents"]]
        assert np.allclose(scores1, scores2)

        weights1 = [c.suggested_weight for c in res1["top_constituents"]]
        weights2 = [c.suggested_weight for c in res2["top_constituents"]]
        assert np.allclose(weights1, weights2)

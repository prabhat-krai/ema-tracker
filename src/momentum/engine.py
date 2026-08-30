"""
Quantitative Momentum Engine & Screening Pipeline.

Orchestrates data ingestion, benchmark regime analysis, trend eligibility filtering,
cross-sectional momentum scoring, top-N portfolio allocation, and rebalancing plan generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union
import numpy as np
import pandas as pd

from src.data_fetcher import fetch_benchmark_daily_data, fetch_universe_daily_data
from src.momentum.classical import calculate_multi_timeframe_returns
from src.momentum.clenow import (
    calculate_multi_timeframe_clenow,
    compute_clenow_momentum,
)
from src.momentum.models import (
    FilterResult,
    MarketRegime,
    MarketRegimeState,
    MomentumMetrics,
    MomentumScoreBreakdown,
    PortfolioConstituent,
    PortfolioRecommendation,
    RebalanceAction,
    RebalancePlan,
)
from src.momentum.portfolio import build_momentum_portfolio
from src.momentum.rebalancer import generate_rebalance_plan
from src.momentum.regime import (
    calculate_52w_high_distance,
    check_golden_regime,
    check_single_stock_trend_filter,
    evaluate_market_regime,
    evaluate_stock_eligibility,
)
from src.momentum.scoring import DEFAULT_FACTOR_WEIGHTS, rank_universe
from src.momentum.volatility import compute_volatility_metrics

logger = logging.getLogger(__name__)


@dataclass
class MomentumPipelineResult:
    """Complete structured output from MomentumPipeline execution."""
    universe: str
    as_of_date: str
    market_regime: MarketRegime
    eligibility_map: Dict[str, FilterResult]
    metrics_list: List[MomentumMetrics]
    ranked_breakdowns: List[MomentumScoreBreakdown]
    ranked_dataframe: pd.DataFrame
    portfolio: PortfolioRecommendation
    rebalance_plan: RebalancePlan
    total_screened: int = 0
    total_eligible: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_constituents(self) -> List[PortfolioConstituent]:
        """Convenience accessor for recommended target constituents."""
        return self.portfolio.target_constituents

    def __getitem__(self, item: str) -> Any:
        """Allow dict-style key access for backward compatibility and test ergonomics."""
        if hasattr(self, item):
            return getattr(self, item)
        elif item in self.metadata:
            return self.metadata[item]
        raise KeyError(f"MomentumPipelineResult has no attribute or key '{item}'")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to Python dictionary."""
        return {
            "universe": self.universe,
            "as_of_date": self.as_of_date,
            "market_regime": self.market_regime.to_dict() if hasattr(self.market_regime, "to_dict") else self.market_regime,
            "eligibility_map": {k: v.to_dict() for k, v in self.eligibility_map.items()},
            "metrics_list": [m.to_dict() for m in self.metrics_list],
            "ranked_breakdowns": [b.to_dict() for b in self.ranked_breakdowns],
            "top_constituents": [c.to_dict() for c in self.top_constituents],
            "portfolio": self.portfolio.to_dict() if hasattr(self.portfolio, "to_dict") else self.portfolio,
            "rebalance_plan": self.rebalance_plan.to_dict() if hasattr(self.rebalance_plan, "to_dict") else self.rebalance_plan,
            "total_screened": self.total_screened,
            "total_eligible": self.total_eligible,
            "metadata": self.metadata,
        }


class MomentumPipeline:
    """
    High-level quantitative momentum screening and portfolio construction pipeline.
    """

    def __init__(
        self,
        universe: Literal["india", "usa"] = "india",
        top_n: int = 10,
        weighting_scheme: str = "inv_vol",
        rank_buffer: int = 20,
        min_weight: float = 0.05,
        max_weight: float = 0.20,
        deadband: float = 0.02,
        total_capital: float = 1_000_000.0,
        factor_weights: Optional[Dict[str, float]] = None,
        filter_trend: bool = True,
        min_history_bars: int = 90,
        min_dist_52w: float = 0.70,
    ):
        self.universe = universe
        self.top_n = top_n
        self.weighting_scheme = weighting_scheme
        self.rank_buffer = rank_buffer
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.deadband = deadband
        self.total_capital = total_capital
        self.factor_weights = factor_weights or DEFAULT_FACTOR_WEIGHTS.copy()
        self.filter_trend = filter_trend
        self.min_history_bars = min_history_bars
        self.min_dist_52w = min_dist_52w

    def run(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "2y",
        delay: float = 0.05,
        current_holdings: Optional[Union[List[str], Dict[str, float]]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> MomentumPipelineResult:
        """
        Executes end-to-end screening with live yfinance data ingestion.

        Args:
            symbols: Optional explicit list of symbols. If None, fetches universe from config.
            period: Lookback period string (default: "2y").
            delay: Rate limiting delay in seconds (default: 0.05).
            current_holdings: Currently held portfolio stocks or weights.
            progress_callback: Optional callback(current, total, symbol).

        Returns:
            MomentumPipelineResult dataclass.
        """
        logger.info(f"Initiating Momentum Pipeline for universe '{self.universe}'")

        # 1. Fetch Benchmark Data
        benchmark_sym = "^NSEI" if self.universe == "india" else "^GSPC"
        logger.info(f"Fetching benchmark data for {benchmark_sym}")
        benchmark_df = fetch_benchmark_daily_data(benchmark_symbol=benchmark_sym, period=period, delay=delay)

        # 2. Fetch Universe Price Data
        universe_data = fetch_universe_daily_data(
            universe=self.universe,
            period=period,
            min_bars=self.min_history_bars,
            delay=delay,
            symbols=symbols,
            progress_callback=progress_callback,
        )

        return self.run_with_data(
            universe_data=universe_data,
            benchmark_df=benchmark_df,
            benchmark_symbol=benchmark_sym,
            current_holdings=current_holdings,
        )

    def run_with_data(
        self,
        universe_data: Dict[str, Optional[pd.DataFrame]],
        benchmark_df: Optional[pd.DataFrame] = None,
        benchmark_symbol: Optional[str] = None,
        current_holdings: Optional[Union[List[str], Dict[str, float]]] = None,
        as_of_date: Optional[str] = None,
    ) -> MomentumPipelineResult:
        """
        Executes pipeline using pre-loaded in-memory DataFrames.
        Enables deterministic testing, offline screening, and simulation backtests.

        Args:
            universe_data: Dict mapping symbol -> OHLCV DataFrame.
            benchmark_df: Benchmark OHLCV DataFrame.
            benchmark_symbol: Benchmark ticker symbol.
            current_holdings: Currently held portfolio holdings.
            as_of_date: Evaluation date string.

        Returns:
            MomentumPipelineResult dataclass.
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime("%Y-%m-%d")
        if benchmark_symbol is None:
            benchmark_symbol = "^NSEI" if self.universe == "india" else "^GSPC"

        # 1. Evaluate Benchmark Market Regime
        benchmark_close = (
            benchmark_df["close"].dropna().values
            if benchmark_df is not None and "close" in benchmark_df.columns
            else None
        )
        market_regime = evaluate_market_regime(
            benchmark_prices=benchmark_close,
            benchmark_symbol=benchmark_symbol,
        )
        market_regime.evaluation_date = as_of_date
        logger.info(
            f"Market Regime for {benchmark_symbol}: {market_regime.state.value} "
            f"(Equity Allocation: {market_regime.equity_allocation_pct*100:.0f}%)"
        )

        # 2. Process Individual Stock Metrics & Eligibility
        eligibility_map: Dict[str, FilterResult] = {}
        metrics_list: List[MomentumMetrics] = []

        total_screened = len(universe_data)

        for sym, df in universe_data.items():
            elig = evaluate_stock_eligibility(
                symbol=sym,
                df=df,
                min_history=self.min_history_bars,
                min_dist_52w=self.min_dist_52w,
            )
            eligibility_map[sym] = elig

            if df is None or df.empty or len(df) < self.min_history_bars:
                continue

            close_arr = df["close"].dropna().values
            if len(close_arr) < self.min_history_bars:
                continue

            high_arr = df["high"].dropna().values if "high" in df.columns else close_arr
            low_arr = df["low"].dropna().values if "low" in df.columns else close_arr

            # Clenow metrics
            slope, r2, score = compute_clenow_momentum(close_arr, lookback=90)
            mtf = calculate_multi_timeframe_clenow(close_arr)

            # Classical intermediate returns
            mom_ret = calculate_multi_timeframe_returns(close_arr)

            # Volatility & ATR metrics
            vol_dict = compute_volatility_metrics(close_arr, high=high_arr, low=low_arr, lookback=90)

            # Macro trend filters
            passes_trend, trend_details = check_single_stock_trend_filter(close_arr, ema_fast=100, sma_slow=200)
            is_golden = check_golden_regime(close_arr)
            dist_52w = calculate_52w_high_distance(close_arr, high_arr, lookback=min(252, len(close_arr)))

            m = MomentumMetrics(
                symbol=sym,
                close=float(close_arr[-1]),
                history_bars=len(close_arr),
                clenow_slope_ann=slope if not np.isnan(slope) else 0.0,
                clenow_r2=r2 if not np.isnan(r2) else 0.0,
                clenow_score=score if not np.isnan(score) else 0.0,
                clenow_score_90=mtf.get("score_90d", 0.0) if not np.isnan(mtf.get("score_90d", 0.0)) else 0.0,
                clenow_score_126=mtf.get("score_126d", 0.0) if not np.isnan(mtf.get("score_126d", 0.0)) else 0.0,
                clenow_score_252=mtf.get("score_252d", 0.0) if not np.isnan(mtf.get("score_252d", 0.0)) else 0.0,
                clenow_score_mtf=mtf.get("composite_score", 0.0) if not np.isnan(mtf.get("composite_score", 0.0)) else 0.0,
                mom_12_1=mom_ret.get("mom_12_1", 0.0) if not np.isnan(mom_ret.get("mom_12_1", 0.0)) else 0.0,
                mom_6_1=mom_ret.get("mom_6_1", 0.0) if not np.isnan(mom_ret.get("mom_6_1", 0.0)) else 0.0,
                mom_3_1=mom_ret.get("mom_3_1", 0.0) if not np.isnan(mom_ret.get("mom_3_1", 0.0)) else 0.0,
                ret_3m=mom_ret.get("ret_3m", 0.0) if not np.isnan(mom_ret.get("ret_3m", 0.0)) else 0.0,
                ret_6m=mom_ret.get("ret_6m", 0.0) if not np.isnan(mom_ret.get("ret_6m", 0.0)) else 0.0,
                ret_12m=mom_ret.get("ret_12m", 0.0) if not np.isnan(mom_ret.get("ret_12m", 0.0)) else 0.0,
                volatility_ann=vol_dict.get("vol_ann", 0.20) if not np.isnan(vol_dict.get("vol_ann", 0.20)) else 0.20,
                volatility_down=vol_dict.get("vol_down", 0.0) if not np.isnan(vol_dict.get("vol_down", 0.0)) else 0.0,
                max_jump_ratio=vol_dict.get("max_jump_ratio", 0.0) if not np.isnan(vol_dict.get("max_jump_ratio", 0.0)) else 0.0,
                jump_penalty=vol_dict.get("jump_penalty", 1.0) if not np.isnan(vol_dict.get("jump_penalty", 1.0)) else 1.0,
                atr_14=vol_dict.get("atr_14", 0.0) if not np.isnan(vol_dict.get("atr_14", 0.0)) else 0.0,
                natr_14=vol_dict.get("natr_14", 0.0) if not np.isnan(vol_dict.get("natr_14", 0.0)) else 0.0,
                ema_10=trend_details.get("ema_fast", 0.0),
                ema_50=elig.ema_50,
                ema_100=trend_details.get("ema_fast", 0.0),
                sma_200=trend_details.get("sma_slow", 0.0),
                dist_52w_high=dist_52w,
                passes_trend=passes_trend,
                golden_alignment=is_golden,
            )
            metrics_list.append(m)

        total_eligible = len([m for m in metrics_list if m.passes_trend])

        # 3. Cross-Sectional Ranking
        ranked_breakdowns, ranked_df = rank_universe(
            metrics_list=metrics_list,
            weights=self.factor_weights,
            filter_trend=self.filter_trend,
        )

        # 4. Portfolio Allocation & Sizing
        current_symbols_list = (
            list(current_holdings.keys())
            if isinstance(current_holdings, dict)
            else (current_holdings or [])
        )

        portfolio = build_momentum_portfolio(
            ranked_constituents=ranked_breakdowns,
            top_n=self.top_n,
            weighting_scheme=self.weighting_scheme,
            current_holdings=current_symbols_list,
            rank_buffer=self.rank_buffer,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
            market_regime=market_regime,
            total_capital=self.total_capital,
            universe=self.universe,
            as_of_date=as_of_date,
        )

        # 5. Rebalancing Plan Generation
        price_map = {m.symbol: m.close for m in metrics_list}
        rebalance_plan = generate_rebalance_plan(
            current_holdings=current_holdings or [],
            target_recommendation=portfolio,
            deadband=self.deadband,
            total_capital=self.total_capital,
            prices=price_map,
            rank_buffer=self.rank_buffer,
            top_n=self.top_n,
            as_of_date=as_of_date,
        )

        return MomentumPipelineResult(
            universe=self.universe,
            as_of_date=as_of_date,
            market_regime=market_regime,
            eligibility_map=eligibility_map,
            metrics_list=metrics_list,
            ranked_breakdowns=ranked_breakdowns,
            ranked_dataframe=ranked_df,
            portfolio=portfolio,
            rebalance_plan=rebalance_plan,
            total_screened=total_screened,
            total_eligible=total_eligible,
        )

"""
Quantitative Momentum Engine Package.

Provides research-backed momentum metrics, trend regime qualification,
volatility normalization, and cross-sectional scoring & ranking.
"""

from src.momentum.models import (
    MarketRegimeState,
    MarketRegime,
    FilterResult,
    MomentumMetrics,
    MomentumScoreBreakdown,
    PortfolioConstituent,
    PortfolioRecommendation,
    RebalanceAction,
    RebalanceInstruction,
    RebalancePlan,
)
from src.momentum.clenow import (
    compute_clenow_momentum,
    calculate_clenow_momentum,
    calculate_multi_timeframe_clenow,
)
from src.momentum.classical import (
    compute_jegadeesh_titman_momentum,
    calculate_jegadeesh_titman_momentum,
    calculate_intermediate_momentum,
    calculate_multi_timeframe_returns,
)
from src.momentum.volatility import (
    calculate_realized_volatility,
    calculate_downside_deviation,
    calculate_jump_ratio_penalty,
    calculate_atr,
    calculate_volatility_adjusted_momentum,
    compute_volatility_metrics,
)
from src.momentum.regime import (
    calculate_sma,
    calculate_ema,
    check_single_stock_trend_filter,
    check_golden_regime,
    calculate_52w_high_distance,
    evaluate_market_regime,
    evaluate_stock_eligibility,
)
from src.momentum.scoring import (
    winsorized_z_score,
    calculate_percentile_rank,
    compute_composite_momentum_score,
    rank_universe,
    DEFAULT_FACTOR_WEIGHTS,
)
from src.momentum.portfolio import (
    calculate_equal_weights,
    calculate_inverse_volatility_weights,
    calculate_bounded_risk_parity_weights,
    calculate_portfolio_risk_metrics,
    build_momentum_portfolio,
)
from src.momentum.rebalancer import (
    classify_rebalance_action,
    calculate_weight_adjustment_action,
    generate_rebalance_plan,
)
from src.momentum.engine import (
    MomentumPipeline,
    MomentumPipelineResult,
)

__all__ = [
    # Models
    "MarketRegimeState",
    "MarketRegime",
    "FilterResult",
    "MomentumMetrics",
    "MomentumScoreBreakdown",
    "PortfolioConstituent",
    "PortfolioRecommendation",
    "RebalanceAction",
    "RebalanceInstruction",
    "RebalancePlan",
    # Clenow Momentum
    "compute_clenow_momentum",
    "calculate_clenow_momentum",
    "calculate_multi_timeframe_clenow",
    # Classical Momentum
    "compute_jegadeesh_titman_momentum",
    "calculate_jegadeesh_titman_momentum",
    "calculate_intermediate_momentum",
    "calculate_multi_timeframe_returns",
    # Volatility & Risk
    "calculate_realized_volatility",
    "calculate_downside_deviation",
    "calculate_jump_ratio_penalty",
    "calculate_atr",
    "calculate_volatility_adjusted_momentum",
    "compute_volatility_metrics",
    # Regime & Trend
    "calculate_sma",
    "calculate_ema",
    "check_single_stock_trend_filter",
    "check_golden_regime",
    "calculate_52w_high_distance",
    "evaluate_market_regime",
    "evaluate_stock_eligibility",
    # Scoring & Ranking
    "winsorized_z_score",
    "calculate_percentile_rank",
    "compute_composite_momentum_score",
    "rank_universe",
    "DEFAULT_FACTOR_WEIGHTS",
    # Portfolio & Weighting
    "calculate_equal_weights",
    "calculate_inverse_volatility_weights",
    "calculate_bounded_risk_parity_weights",
    "calculate_portfolio_risk_metrics",
    "build_momentum_portfolio",
    # Rebalancing & Turnover
    "classify_rebalance_action",
    "calculate_weight_adjustment_action",
    "generate_rebalance_plan",
    # Engine Pipeline
    "MomentumPipeline",
    "MomentumPipelineResult",
]


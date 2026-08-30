"""
Momentum Models & Data Structures.

Strongly typed dataclasses and enumerations for quantitative momentum scoring,
market regime qualification, and cross-sectional ranking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class MarketRegimeState(str, Enum):
    """Macro market regime classification."""
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


class RebalanceAction(str, Enum):
    """Portfolio rebalancing action signal."""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class MarketRegime:
    """Represents macro market regime evaluation for benchmark indices."""
    benchmark_symbol: str
    state: MarketRegimeState
    benchmark_price: float
    sma_200: float
    sma_200_slope: float
    equity_allocation_pct: float  # 1.0 for Bullish, 0.5 for Neutral, 0.0 for Bearish
    evaluation_date: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_symbol": self.benchmark_symbol,
            "state": self.state.value if isinstance(self.state, MarketRegimeState) else str(self.state),
            "benchmark_price": self.benchmark_price,
            "sma_200": self.sma_200,
            "sma_200_slope": self.sma_200_slope,
            "equity_allocation_pct": self.equity_allocation_pct,
            "evaluation_date": self.evaluation_date,
            "details": self.details,
        }


@dataclass
class FilterResult:
    """Eligibility and macro trend filter result for a single stock."""
    symbol: str
    is_eligible: bool
    reasons: List[str] = field(default_factory=list)
    history_days: int = 0
    price: float = 0.0
    ema_50: float = 0.0
    ema_100: float = 0.0
    sma_200: float = 0.0
    price_above_ema100: bool = False
    price_above_sma200: bool = False
    golden_alignment: bool = False
    dist_52w_high: float = 0.0  # Close / 52w High
    adtv_20: float = 0.0  # 20-day Average Daily Trading Volume in currency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "is_eligible": self.is_eligible,
            "reasons": self.reasons,
            "history_days": self.history_days,
            "price": self.price,
            "ema_50": self.ema_50,
            "ema_100": self.ema_100,
            "sma_200": self.sma_200,
            "price_above_ema100": self.price_above_ema100,
            "price_above_sma200": self.price_above_sma200,
            "golden_alignment": self.golden_alignment,
            "dist_52w_high": self.dist_52w_high,
            "adtv_20": self.adtv_20,
        }


@dataclass
class MomentumMetrics:
    """Raw and annualized momentum, volatility, and trend metrics for a stock."""
    symbol: str
    close: float
    history_bars: int = 0

    # Andreas Clenow Exponential Regression Metrics
    clenow_slope_ann: float = 0.0  # Annualized exponential slope (e^(250*beta) - 1)
    clenow_r2: float = 0.0          # R^2 coefficient of determination
    clenow_score: float = 0.0       # clenow_slope_ann * clenow_r2 (default lookback, e.g. 90d)
    clenow_score_90: float = 0.0
    clenow_score_126: float = 0.0
    clenow_score_252: float = 0.0
    clenow_score_mtf: float = 0.0   # Multi-timeframe composite: 0.40*90 + 0.30*126 + 0.30*252

    # Classical Jegadeesh & Titman Momentum Metrics
    mom_12_1: float = 0.0          # (P_{t-21} - P_{t-252}) / P_{t-252}
    mom_6_1: float = 0.0           # (P_{t-21} - P_{t-126}) / P_{t-126}
    mom_3_1: float = 0.0           # (P_{t-21} - P_{t-63}) / P_{t-63}
    ret_3m: float = 0.0            # Unskipped 3-month return
    ret_6m: float = 0.0            # Unskipped 6-month return
    ret_12m: float = 0.0           # Unskipped 12-month return

    # Volatility & Risk Penalties
    volatility_ann: float = 0.0    # Annualized realized volatility (sigma_daily * sqrt(252))
    volatility_down: float = 0.0   # Annualized downside semi-deviation
    max_jump_ratio: float = 0.0    # Max daily jump / daily std
    jump_penalty: float = 1.0      # Multiplier penalty in (0, 1]
    atr_14: float = 0.0            # 14-day Average True Range
    natr_14: float = 0.0           # Normalized ATR (%)

    # Trend & Moving Averages
    ema_10: float = 0.0
    ema_50: float = 0.0
    ema_100: float = 0.0
    sma_200: float = 0.0
    dist_52w_high: float = 0.0     # Close / 52w High (1.0 = at high)
    passes_trend: bool = False
    golden_alignment: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "close": self.close,
            "history_bars": self.history_bars,
            "clenow_slope_ann": self.clenow_slope_ann,
            "clenow_r2": self.clenow_r2,
            "clenow_score": self.clenow_score,
            "clenow_score_90": self.clenow_score_90,
            "clenow_score_126": self.clenow_score_126,
            "clenow_score_252": self.clenow_score_252,
            "clenow_score_mtf": self.clenow_score_mtf,
            "mom_12_1": self.mom_12_1,
            "mom_6_1": self.mom_6_1,
            "mom_3_1": self.mom_3_1,
            "ret_3m": self.ret_3m,
            "ret_6m": self.ret_6m,
            "ret_12m": self.ret_12m,
            "volatility_ann": self.volatility_ann,
            "volatility_down": self.volatility_down,
            "max_jump_ratio": self.max_jump_ratio,
            "jump_penalty": self.jump_penalty,
            "atr_14": self.atr_14,
            "natr_14": self.natr_14,
            "ema_10": self.ema_10,
            "ema_50": self.ema_50,
            "ema_100": self.ema_100,
            "sma_200": self.sma_200,
            "dist_52w_high": self.dist_52w_high,
            "passes_trend": self.passes_trend,
            "golden_alignment": self.golden_alignment,
            "metadata": self.metadata,
        }


@dataclass
class MomentumScoreBreakdown:
    """Cross-sectional score breakdown, composite score, and ranking."""
    symbol: str
    rank: int = 0
    composite_score: float = 0.0
    percentile_rank: float = 0.0

    # Standardized Z-Score Factor Components
    z_clenow: float = 0.0
    z_mom_12_1: float = 0.0
    z_r2: float = 0.0
    z_mom_6_1: float = 0.0
    z_vol_ann: float = 0.0

    # Underlying raw metric values for transparency
    raw_clenow_score: float = 0.0
    raw_slope_ann: float = 0.0
    raw_r2: float = 0.0
    raw_mom_12_1: float = 0.0
    raw_mom_6_1: float = 0.0
    raw_vol_ann: float = 0.0
    close: float = 0.0
    natr_14: float = 0.0

    # Sizing & allocation
    suggested_weight: float = 0.0
    action: RebalanceAction = RebalanceAction.HOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "composite_score": self.composite_score,
            "percentile_rank": self.percentile_rank,
            "z_clenow": self.z_clenow,
            "z_mom_12_1": self.z_mom_12_1,
            "z_r2": self.z_r2,
            "z_mom_6_1": self.z_mom_6_1,
            "z_vol_ann": self.z_vol_ann,
            "raw_clenow_score": self.raw_clenow_score,
            "raw_slope_ann": self.raw_slope_ann,
            "raw_r2": self.raw_r2,
            "raw_mom_12_1": self.raw_mom_12_1,
            "raw_mom_6_1": self.raw_mom_6_1,
            "raw_vol_ann": self.raw_vol_ann,
            "close": self.close,
            "natr_14": self.natr_14,
            "suggested_weight": self.suggested_weight,
            "action": self.action.value if isinstance(self.action, RebalanceAction) else str(self.action),
        }


@dataclass
class PortfolioConstituent:
    """Individual stock allocation and metrics within recommended portfolio."""
    symbol: str
    rank: int
    weight: float
    composite_score: float
    clenow_score: float
    clenow_slope_ann: float
    clenow_r2: float
    mom_12_1: float
    mom_6_1: float
    volatility_ann: float
    natr_14: float
    close: float
    action: RebalanceAction = RebalanceAction.HOLD
    target_value: float = 0.0
    target_shares: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "weight": self.weight,
            "composite_score": self.composite_score,
            "clenow_score": self.clenow_score,
            "clenow_slope_ann": self.clenow_slope_ann,
            "clenow_r2": self.clenow_r2,
            "mom_12_1": self.mom_12_1,
            "mom_6_1": self.mom_6_1,
            "volatility_ann": self.volatility_ann,
            "natr_14": self.natr_14,
            "close": self.close,
            "action": self.action.value if isinstance(self.action, RebalanceAction) else str(self.action),
            "target_value": self.target_value,
            "target_shares": self.target_shares,
            "metadata": self.metadata,
        }


@dataclass
class PortfolioRecommendation:
    """Top N quantitative momentum portfolio allocation recommendation."""
    universe: str
    as_of_date: str
    market_regime: MarketRegime
    weighting_scheme: str
    target_constituents: List[PortfolioConstituent]
    total_capital: float = 1_000_000.0
    total_equity_pct: float = 1.0
    expected_cash_pct: float = 0.0
    allocated_cash_pct: float = 0.0
    weighted_clenow_score: float = 0.0
    weighted_volatility: float = 0.0
    portfolio_volatility: float = 0.0
    diversification_ratio: float = 1.0
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe": self.universe,
            "as_of_date": self.as_of_date,
            "market_regime": self.market_regime.to_dict() if hasattr(self.market_regime, "to_dict") else self.market_regime,
            "weighting_scheme": self.weighting_scheme,
            "target_constituents": [c.to_dict() for c in self.target_constituents],
            "total_capital": self.total_capital,
            "total_equity_pct": self.total_equity_pct,
            "expected_cash_pct": self.expected_cash_pct,
            "allocated_cash_pct": self.allocated_cash_pct,
            "weighted_clenow_score": self.weighted_clenow_score,
            "weighted_volatility": self.weighted_volatility,
            "portfolio_volatility": self.portfolio_volatility,
            "diversification_ratio": self.diversification_ratio,
            "summary": self.summary,
        }


@dataclass
class RebalanceInstruction:
    """Discrete trade instruction for a portfolio holding during rebalancing."""
    symbol: str
    action: str  # "NEW_BUY", "HOLD_RETAIN", "SELL_EXIT", "REBALANCE_ADD", "REBALANCE_TRIM", "NO_CHANGE"
    current_weight: float
    target_weight: float
    weight_delta: float
    current_shares: int = 0
    target_shares: int = 0
    trade_shares: int = 0
    price: float = 0.0
    estimated_trade_value: float = 0.0
    rank: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "weight_delta": self.weight_delta,
            "current_shares": self.current_shares,
            "target_shares": self.target_shares,
            "trade_shares": self.trade_shares,
            "price": self.price,
            "estimated_trade_value": self.estimated_trade_value,
            "rank": self.rank,
            "reason": self.reason,
        }


@dataclass
class RebalancePlan:
    """Complete portfolio rebalancing execution plan with trade orders and turnover."""
    evaluation_date: str
    universe: str
    total_capital: float
    current_holdings: List[str]
    target_holdings: List[str]
    buys: List[RebalanceInstruction] = field(default_factory=list)
    holds: List[RebalanceInstruction] = field(default_factory=list)
    sells: List[RebalanceInstruction] = field(default_factory=list)
    all_trades: List[RebalanceInstruction] = field(default_factory=list)
    turnover_pct: float = 0.0
    estimated_cash_change: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_date": self.evaluation_date,
            "universe": self.universe,
            "total_capital": self.total_capital,
            "current_holdings": self.current_holdings,
            "target_holdings": self.target_holdings,
            "buys": [b.to_dict() for b in self.buys],
            "holds": [h.to_dict() for h in self.holds],
            "sells": [s.to_dict() for s in self.sells],
            "all_trades": [t.to_dict() for t in self.all_trades],
            "turnover_pct": self.turnover_pct,
            "estimated_cash_change": self.estimated_cash_change,
            "summary": self.summary,
        }


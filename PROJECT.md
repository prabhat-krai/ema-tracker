# Project: Quantitative Momentum Portfolio Tracker

## Architecture
The Quantitative Momentum Portfolio Tracker is structured as a high-performance, mathematically rigorous quantitative finance system seamlessly integrated into the existing `ema-tracker` ecosystem.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      User Request                      │
                    │               (ORIGINAL_REQUEST.md)                    │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
                 ▼                                                             ▼
   ┌───────────────────────────┐                                 ┌───────────────────────────┐
   │   Implementation Track    │                                 │    E2E Testing Track      │
   │  (M1 -> M2 -> M3 -> M4)   │                                 │ (Opaque-Box Tiers 1-4)    │
   └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                 │                                                             │
                 ├─► M1: Core Momentum Scoring Engine [DONE]                   ├─► Test Infrastructure [DONE]
                 ├─► M2: Portfolio Selection & Weighting                       ├─► Tiers 1-4 Test Suite [DONE]
                 ├─► M3: Streamlit UI, CLI & Exporters                         └─► Publishes TEST_READY.md [DONE]
                 └─► M4: Final E2E Pass & Tier 5 Hardening <───────────────────┘
```

### Module Boundaries
1. `src/momentum/models.py`: Strongly typed dataclasses for MomentumMetrics, PortfolioConstituent, MarketRegime, PortfolioRecommendation, RebalancePlan.
2. `src/momentum/clenow.py`: Closed-form OLS exponential regression ($\ln(P_t) = \alpha + \beta t$), annualized slope ($\exp(250\beta)-1$), $R^2$ calculation, Clenow momentum score ($\text{Annualized Slope} \times R^2$).
3. `src/momentum/classical.py`: Jegadeesh & Titman (1993) 12-1 intermediate momentum ($t-252$ to $t-21$ return) and multi-timeframe returns.
4. `src/momentum/volatility.py`: Realized annualized volatility ($\sigma_{\text{ann}} = \sigma_{\text{daily}}\sqrt{252}$), ATR14, and downside/smoothness penalties.
5. `src/momentum/regime.py`: Macro trend filters ($P > \text{EMA}_{100}$, $P > \text{SMA}_{200}$), golden alignment, and benchmark regime check (`^NSEI`, `^GSPC`).
6. `src/momentum/scoring.py`: Winsorized cross-sectional Z-scoring and multi-factor composite ranking.
7. `src/momentum/portfolio.py`: Top 10 selection, Equal Weighting, Inverse-Volatility Weighting, Bounded Risk Parity (5% to 20%).
8. `src/momentum/rebalancer.py`: Hysteresis rank buffer (Buy $\le 10$, Hold $\le 20$, Sell $> 20$), trend breakdown exits, turnover management.
9. `src/momentum/exporter.py`: Multi-format reporting engine generating CSV, JSON, and Markdown executive summaries in `reports/momentum/`.
10. `src/momentum/engine.py` & `src/momentum/cli.py`: Unified screening pipeline and CLI runner.
11. `src/data_fetcher.py`: Extended with daily OHLCV price fetching (`fetch_daily_data()`) and benchmark index caching.
12. `src/app.py`: Integrated Tab 4 ("🏆 Quant Momentum Portfolio") with interactive metric cards, Top 10 table with gradient styling, rebalancing transition board, and Plotly charts.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Clenow Exponential Regression Model | OLS log-price regression $\ln(P_t)=\alpha+\beta t$, annualized slope $\exp(250\beta)-1$, $R^2$, and Clenow score | M1 | R1 / ORIGINAL_REQUEST |
| 2 | Jegadeesh & Titman 12-1 Momentum | Intermediate momentum from $t-252$ to $t-21$ skipping the recent 1 month | M1 | R1 / ORIGINAL_REQUEST |
| 3 | Volatility Normalization & Smoothness Penalty | Realized annualized volatility $\sigma_{\text{ann}}$, downside semi-deviation, and jump penalty | M1 | R1 / ORIGINAL_REQUEST |
| 4 | Trend Regime & Moving Average Filters | $P > \text{EMA}_{100}$, $P > \text{SMA}_{200}$, Golden Alignment, and Benchmark macro circuit breaker | M1 | R1 / ORIGINAL_REQUEST |
| 5 | Cross-Sectional Composite Ranking | Winsorized Z-scoring, multi-factor weighting, and robust percentile ranking | M1 | R1 / ORIGINAL_REQUEST |
| 6 | Robust Edge-Case Handling | Graceful handling of missing history (<252d, min 90d), zero volume, NaNs, zero volatility, and split adjustments | M1 | R1 / ORIGINAL_REQUEST |
| 7 | Multi-Universe Data & Ingestion Pipeline | Daily OHLCV data fetcher for Indian (NSE Nifty 500) and US (S&P 500) universes with caching and rate limiting | M2 | R2 / ORIGINAL_REQUEST |
| 8 | Top 10 Portfolio Selection & Component Transparency | Filtering and ranking top 10 momentum stocks with full component transparency (slope, $R^2$, vol, raw mom, rank) | M2 | R2 / ORIGINAL_REQUEST |
| 9 | Portfolio Weighting Schemes | Equal Weighting (10%), Inverse-Volatility Weighting, and Bounded Risk Parity (5%-20%) | M2 | R2 / ORIGINAL_REQUEST |
| 10 | Rebalancing & Hysteresis Exit Rules | Hysteresis rank buffer (Hold $\le 20$), trend breakdown exits, and turnover reduction | M2 | R2 / ORIGINAL_REQUEST |
| 11 | Streamlit UI Momentum Portfolio Tab | Dedicated Tab 4 in `src/app.py` with KPI cards, Top 10 dataframe with heatmaps, rebalance board, and Plotly charts | M3 | R3 / ORIGINAL_REQUEST |
| 12 | CLI Screening Runner | Standalone CLI (`python -m src.momentum`) with flags (`--universe`, `--top`, `--weighting`, `--export`, etc.) | M3 | R3 / ORIGINAL_REQUEST |
| 13 | Multi-Format Report Exporter | Automated export of CSV, JSON, and Markdown summaries into `reports/momentum/` | M3 | R3 / ORIGINAL_REQUEST |
| 14 | Shell Script Integration | Integration into `run_india.sh`, `run_usa.sh`, and dedicated `run_momentum.sh` runner | M3 | R3 / ORIGINAL_REQUEST |
| 15 | E2E Test Suite (Tiers 1-4) | Comprehensive opaque-box test suite covering feature coverage, boundaries, combinations, and real-world workloads | E2E Track / M4 | R4 / ORIGINAL_REQUEST |
| 16 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial testing, stress testing, and edge-case code path coverage | M4 | R4 / ORIGINAL_REQUEST |
| 17 | Verification Scripts for Indian & US Markets | Automated verification scripts (`scripts/verify_momentum.py`) validating live/sample data end-to-end | M4 | R4 / ORIGINAL_REQUEST |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Design test infrastructure (`TEST_INFRA.md`), build 4-tier opaque-box test suite (Tiers 1-4), publish `TEST_READY.md` | none | DONE |
| M1 | Momentum Scoring & Ranking Core Engine | Implement `src/momentum/models.py`, `clenow.py`, `classical.py`, `volatility.py`, `regime.py`, `scoring.py`, and unit tests | none | DONE |
| M2 | Portfolio Selection, Weighting & Ingestion | Implement `src/momentum/portfolio.py`, `rebalancer.py`, `engine.py`, extend `src/data_fetcher.py` for daily data, Indian & US universe pipelines | M1 | DONE |
| M3 | UI Integration, CLI & Exporters | Implement `src/app.py` Tab 4, `src/momentum/cli.py`, `exporter.py`, shell scripts (`run_momentum.sh`, etc.) | M2 | DONE |
| M4 | Final E2E Test Suite Pass & Adversarial Hardening | Phase 1: 100% pass on E2E Test Suite (Tiers 1-4); Phase 2: Tier 5 Adversarial Coverage Hardening & live verification scripts | E2E, M3 | DONE |

---

## Interface Contracts

### Data Fetcher Contract (`src/data_fetcher.py`)
```python
def fetch_daily_data(
    ticker: str,
    period: str = "2y",
    min_bars: int = 90,
    delay: float = 0.05
) -> Optional[pd.DataFrame]:
    """
    Fetches daily OHLCV data for a ticker with automatic retry and rate-limiting.
    Returns DataFrame indexed by Datetime with columns: ['Open', 'High', 'Low', 'Close', 'Volume'].
    Returns None if data is missing or length < min_bars.
    """
```

### Clenow Math Contract (`src/momentum/clenow.py`)
```python
def calculate_clenow_momentum(
    prices: np.ndarray,
    trading_days: int = 250
) -> Tuple[float, float, float]:
    """
    Computes (annualized_slope, r_squared, clenow_score).
    log(prices) = alpha + beta * t
    annualized_slope = exp(beta * trading_days) - 1.0
    r_squared = Pearson correlation squared
    clenow_score = annualized_slope * r_squared
    """
```

### Portfolio Allocation Contract (`src/momentum/portfolio.py`)
```python
def build_momentum_portfolio(
    ranked_constituents: List[MomentumScoreBreakdown],
    metrics_map: Dict[str, MomentumMetrics],
    top_n: int = 10,
    weighting_scheme: str = "inv_vol",  # 'equal', 'inv_vol', 'bounded_parity'
    current_holdings: Optional[List[str]] = None,
    rank_buffer: int = 20,
    max_weight: float = 0.20,
    min_weight: float = 0.05
) -> PortfolioRecommendation:
    """
    Generates Top N portfolio recommendations with exact component weights,
    rebalance transitions (BUY, HOLD, SELL), and portfolio-level risk metrics.
    """
```

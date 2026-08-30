# Quantitative Momentum Portfolio Tracker — Test Infrastructure & Philosophy

## 1. Testing Philosophy & Guiding Principles

The Quantitative Momentum Portfolio Tracker employs an **opaque-box, requirement-driven 4-tier verification methodology**. 

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 TIER 4: Workload Tests                   │
                  │  Full E2E Multi-Asset Universes (Indian & US), Ranking,  │
                  │  Portfolio Sizing, CSV/JSON/Markdown Payload Integrity   │
                  ├──────────────────────────────────────────────────────────┤
                  │                TIER 3: Combination Tests                 │
                  │  Cross-Feature Interactions: Macro Circuit Breaker,      │
                  │  Regime Filter Gating + Sizing, Hysteresis Transitions   │
                  ├──────────────────────────────────────────────────────────┤
                  │                TIER 2: Boundary & Corner Tests           │
                  │  Exact 90d/252d Cutoffs, Zero Vol Clamping, NaNs/Infs,   │
                  │  Extreme Jump Outliers, Degenerate Universes (N < 10)    │
                  ├──────────────────────────────────────────────────────────┤
                  │                TIER 1: Core Feature Coverage             │
                  │  >= 5 Mathematical Unit Tests per Quant Feature with     │
                  │  Closed-Form Analytical Ground Truth Oracles             │
                  └──────────────────────────────────────────────────────────┘
```

### Core Tenets:
1. **Mathematical Ground Truth (No Facades)**: Every mathematical calculation (Clenow regression slope, coefficient of determination $R^2$, 12-1 intermediate momentum, realized volatility, Winsorized Z-score standardization, inverse-volatility sizing) is validated against closed-form analytical oracles derived directly from literature specifications (Andreas Clenow 2015, Jegadeesh & Titman 1993, Barroso & Santa-Clara 2015).
2. **Opaque-Box Requirement Verification**: Tests treat modules as opaque systems defined strictly by their interface contracts (`PROJECT.md`, `ORIGINAL_REQUEST.md`). Tests exercise observable outputs given controlled inputs without asserting internal private implementation states.
3. **Deterministic Fixture Generation**: Synthetic time series generator (`tests/e2e/fixtures.py`) generates fully reproducible price histories for both Indian (`.NS`) and US equities covering all regimes (exponential uptrends, choppy noise, downtrends, earnings gap-ups, flat trading, zero volume, market crash benchmarks).
4. **Adversarial & Numerical Stress**: Every feature is challenged with adversarial inputs: zero volatility flatlines, scale multiplier invariance ($P_t \to 100 \times P_t$), sudden $+50\%$ jump days, severe multi-month drawdowns, missing/corrupted columns, and all-NaN arrays.

---

## 2. 4-Tier Test Architecture

### Tier 1: Core Feature Coverage (`tests/e2e/test_tier1_features.py`)
Provides $\ge 5$ exhaustive mathematical tests for each of the 8 core quantitative momentum features:
1. **Clenow Exponential Regression & Annualized Slope**:
   - Analytical exponential curve ($P_t = 100(1.002)^t$) ground truth verification ($\text{Slope}_{\text{ann}} = (1.002)^{250}-1$).
   - Flat price series ($\beta = 0, \text{Slope}_{\text{ann}} = 0$).
   - Negative drift downtrend series ($\text{Slope}_{\text{ann}} < 0$).
   - Scale invariance ($P_t \to c \cdot P_t$ preserves slope and $R^2$).
   - Variable lookback windows ($N = 90, 126, 252$).
2. **Coefficient of Determination ($R^2$) Trend Smoothness**:
   - Perfect exponential series produces $R^2 = 1.000000$.
   - Zero variance flat series returns safe $R^2 = 0.0$ without division-by-zero.
   - Noise penalty: adding Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ monotonically decreases $R^2$.
   - Bound adherence: $0.0 \le R^2 \le 1.0$ across diverse synthetic paths.
   - Clenow momentum score calculation ($\text{Score} = \text{Slope}_{\text{ann}} \times R^2$).
3. **Jegadeesh & Titman 12-1 Intermediate Momentum**:
   - Exact analytical calculation ($(P_{t-21} - P_{t-252}) / P_{t-252}$).
   - Microstructure skip verification: severe $-20\%$ crash in the last 21 trading days does NOT contaminate 12-1 return.
   - Intermediate 6-1 momentum ($(P_{t-21} - P_{t-126}) / P_{t-126}$).
   - Insufficient history handling (length $< 252$ returns `NaN`).
   - Zero or negative base price protection ($P_{t-252} \le 0$ returns `NaN`).
4. **Realized Volatility & Risk Normalization**:
   - Realized annualized volatility ($\sigma_{\text{ann}} = \sigma_{\text{daily}} \sqrt{252}$) vs NumPy standard deviation.
   - Flat series zero volatility verification ($\sigma_{\text{ann}} = 0.0$).
   - Annualization scaling factor ($\sqrt{252}$).
   - Downside semi-deviation (penalizes negative returns only).
   - Jump ratio penalty calculation ($\text{MaxJump} / \sigma_{\text{daily}}$).
5. **Moving Average Regime & Macro Trend Filters**:
   - Dual moving average condition ($\text{Close} > \text{EMA}_{100}$ and $\text{Close} > \text{SMA}_{200}$).
   - Golden alignment ($\text{EMA}_{50} > \text{EMA}_{100} > \text{SMA}_{200}$).
   - Ascending 200 SMA slope condition ($\text{SMA}_{200}(t) \ge \text{SMA}_{200}(t-20)$).
   - 52-Week High distance ratio ($\text{Close} / \max(\text{High}_{252}) \ge 0.75$).
   - Macro benchmark gate (`^NSEI` / `^GSPC` above vs below $\text{SMA}_{200}$).
6. **Cross-Sectional Winsorized Z-Score & Composite Scoring**:
   - Zero mean and unit variance verification of standardized factor arrays.
   - Winsorization clipping at $[-3.0, +3.0]$ on extreme $10\sigma$ outliers.
   - Multi-factor composite weighting equation adherence ($w_1 Z_{\text{clenow}} + w_2 Z_{12-1} + w_3 Z_{R^2} + w_4 Z_{6-1} - w_5 Z_{\text{vol}}$).
   - Score tie-breaking and ordinal rank sorting ($1 \dots M$).
   - Rescaling on uniform constant factor values (zero standard deviation safe fallback).
7. **Portfolio Weighting Engines**:
   - Equal Weighting: $w_i = 1/K = 0.10$ and $\sum w_i = 1.0$.
   - Inverse-Volatility Weighting: $w_i \propto 1/\sigma_i$ and $\sum w_i = 1.0$.
   - Inverse proportionality verification (stock with $\sigma=10\%$ gets $2\times$ weight of stock with $\sigma=20\%$).
   - Bounded Risk Parity: weights clipped to $[w_{\min}, w_{\max}] = [0.05, 0.20]$ and normalized to 1.0.
   - Single-asset universe allocation ($K = 1 \implies w_1 = 1.0$).
8. **Rebalancing Hysteresis Rank Buffer**:
   - New entrant (non-holding with Rank $\le 10$) assigned `NEW_BUY`.
   - Existing holding with Rank 14 (within buffer $11 \dots 20$) assigned `HOLD`.
   - Existing holding with Rank 25 (dropped beyond buffer 20) assigned `EXIT`.
   - Existing holding breaking 100-day EMA assigned immediate `EXIT` regardless of rank.
   - Turnover control: rank buffer avoids turnover for small rank fluctuations.

---

### Tier 2: Boundary & Corner Cases (`tests/e2e/test_tier2_boundaries.py`)
Validates system resilience against extreme edge and boundary conditions:
- **Exact 90-Day Regression Cutoff**: Exactly 89 bars returns `NaN`; exactly 90 bars computes valid metrics.
- **Exact 252-Day 12-1 Cutoff**: Exactly 251 bars returns `NaN`; exactly 252 bars computes valid 12-1 return.
- **Zero Volatility Safe Clamping**: Constant price series clamped with $\epsilon = 10^{-6}$ / floored at $\sigma_{\min} = 0.05$ without raising `ZeroDivisionError` during inverse-volatility sizing.
- **All-NaN and Empty Data Series**: Completely empty DataFrame or all-NaN prices gracefully returns `None` / `NaN` without uncaught exceptions.
- **Single-Asset Universe ($M = 1$)**: Pipeline processes universe of size 1, allocating 100% weight.
- **Small Universe ($1 < M < 10$)**: Handles universe smaller than target portfolio size $K=10$, allocating available weights and remaining to cash reserve.
- **Extreme Outlier Returns (+1000% jump)**: Winsorized at $Z = +3.0$, preventing single-asset distortion of the entire universe.
- **Missing / Mismatched OHLCV Columns**: Handled gracefully with warning and safe skip.

---

### Tier 3: Cross-Feature Combinations (`tests/e2e/test_tier3_combinations.py`)
Validates composite interactions across multiple pipeline stages:
- **Macro Benchmark Gate + Asset Allocation**:
  - `BULLISH` market ($\text{Benchmark} > \text{SMA}_{200}$): 100% allocated across Top 10 stocks.
  - `NEUTRAL` market ($\text{Benchmark} > \text{SMA}_{200}$ with descending slope): 50% equity allocation, 50% cash reserve.
  - `BEARISH` market ($\text{Benchmark} \le \text{SMA}_{200}$): 0% equity allocation, 100% cash defensive stance.
- **Trend Filter Gating + Multi-Factor Sizing**:
  - High Clenow score stock failing $\text{Close} > \text{EMA}_{100}$ is disqualified from portfolio eligibility.
  - Passing stocks receive proportional inverse-volatility weights summing to target allocation.
- **Hysteresis Rank Buffer Transitions (NEW_BUY / HOLD / EXIT)**:
  - Multi-period portfolio evolution tracking transitions across sequential rebalances.
  - Verification of simultaneous new entry, retention within buffer, and trend breakdown exit.
- **Multi-Format Export Synchronization**:
  - Verification that CSV, JSON, and Markdown exporters produce identical numerical constituents, weights, and ranks.

---

### Tier 4: Real-World Multi-Asset Workloads (`tests/e2e/test_tier4_workloads.py`)
Validates complete end-to-end execution flows across multi-asset universes:
- **Full Indian Universe Simulation (NSE Nifty 500)**:
  - 30+ synthetic Indian stocks (`.NS`) representing diverse market behaviors.
  - Full screening pipeline: data validation $\to$ feature computation $\to$ trend filtering $\to$ composite ranking $\to$ top 10 selection $\to$ inverse-vol weighting $\to$ report export.
  - Validation of ranking table schema, constituent count, weight summation, and INR currency formatting.
- **Full US Universe Simulation (S&P 500)**:
  - 30+ synthetic US stocks (`AAPL`, `MSFT`, `BRK-B`, `NVDA`, etc.).
  - Full pipeline verification under USD currency formatting and bounded risk-parity sizing.
- **Payload Schema & Export Verification**:
  - Validates generated CSV against expected headers.
  - Validates generated JSON against strict schema (metadata, market regime, portfolio summary, constituents, rebalance actions).
  - Validates generated Markdown summary structure.
- **Determinism & Idempotence**:
  - Two consecutive runs with identical data produce bitwise identical rankings, weights, and JSON exports.

---

## 3. Feature Coverage Checklist & Traceability Matrix

| Feature ID | Description | Source Requirement | Primary Test Location | Test Count |
|---|---|---|---|:---:|
| **F-01** | Andreas Clenow Exponential Regression Slope ($\text{Slope}_{\text{ann}}$) | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestClenowSlope` | 6 |
| **F-02** | Coefficient of Determination ($R^2$) Trend Smoothness | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestClenowRSquared` | 6 |
| **F-03** | Andreas Clenow Momentum Score ($\text{Slope}_{\text{ann}} \times R^2$) | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestClenowScore` | 5 |
| **F-04** | Jegadeesh & Titman 12-1 Intermediate Momentum | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestClassicalMomentum` | 6 |
| **F-05** | Realized Volatility & Downside / Jump Risk Penalties | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestVolatilityMetrics` | 6 |
| **F-06** | Single-Stock Moving Average & Trend Filters (100 EMA, 200 SMA, 52W High) | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestRegimeFilters` | 6 |
| **F-07** | Macro Market Benchmark Gate (`^NSEI` / `^GSPC`) | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestBenchmarkRegime` | 5 |
| **F-08** | Winsorized Z-Score Standardization & Composite Scoring | R1 / ORIGINAL_REQUEST | `test_tier1_features.py::TestCompositeScoring` | 6 |
| **F-09** | Portfolio Weighting Schemes (EW, IVW, Bounded Parity) | R2 / ORIGINAL_REQUEST | `test_tier1_features.py::TestPortfolioWeighting` | 6 |
| **F-10** | Rebalancing Hysteresis Rank Buffer & Exit Signals | R2 / ORIGINAL_REQUEST | `test_tier1_features.py::TestRebalancingBuffer` | 6 |
| **F-11** | Boundary & Numerical Corner Cases | R4 / ORIGINAL_REQUEST | `test_tier2_boundaries.py` | 10 |
| **F-12** | Cross-Feature Combinations & State Transitions | R1, R2 / ORIGINAL_REQUEST | `test_tier3_combinations.py` | 8 |
| **F-13** | End-to-End Indian (NSE) Universe Screening Workload | R2, R3 / ORIGINAL_REQUEST | `test_tier4_workloads.py::TestIndiaUniverseWorkload` | 4 |
| **F-14** | End-to-End US (S&P 500) Universe Screening Workload | R2, R3 / ORIGINAL_REQUEST | `test_tier4_workloads.py::TestUSAUniverseWorkload` | 4 |
| **F-15** | Multi-Format Reporting Pipeline (CSV / JSON / Markdown) | R3 / ORIGINAL_REQUEST | `test_tier4_workloads.py::TestExportPayloads` | 4 |

---

## 4. Test Execution & CI Commands

### Run Full E2E Test Suite:
```bash
./venv/bin/pytest tests/e2e/ -v
```

### Run by Tier:
```bash
# Tier 1: Core Feature Coverage
./venv/bin/pytest tests/e2e/test_tier1_features.py -v

# Tier 2: Boundary & Corner Cases
./venv/bin/pytest tests/e2e/test_tier2_boundaries.py -v

# Tier 3: Cross-Feature Combinations
./venv/bin/pytest tests/e2e/test_tier3_combinations.py -v

# Tier 4: Real-World Multi-Asset Workloads
./venv/bin/pytest tests/e2e/test_tier4_workloads.py -v
```

### Run Entire Repository Test Suite:
```bash
./venv/bin/pytest -v
```

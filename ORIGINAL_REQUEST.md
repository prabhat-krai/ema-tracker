# Original User Request

## 2026-08-29T18:09:22Z

Build a research-backed Quantitative Momentum Portfolio Tracker that analyzes the existing universe of tracked stocks (Indian Nifty 500 / Midcap 150 and US S&P 500), evaluates leading quantitative momentum literature to implement an advanced momentum scoring and ranking algorithm, and surfaces the top 10 momentum stocks in both CLI and the Streamlit web dashboard.

Working directory: .
Integrity mode: development

## Requirements

### R1. Momentum Research & Quantitative Ranking Engine
- Research leading empirical and academic momentum literature (such as Andreas Clenow's exponential regression slope × $R^2$ trend-following model, Jegadeesh & Titman's 12-1 intermediate momentum, and risk-adjusted / volatility-normalized momentum metrics).
- Implement a robust momentum scoring engine that calculates multi-factor or multi-timeframe momentum scores, adjusting for volatility/smoothness (e.g., penalizing gap-driven erratic moves) and filtering for strong trend regimes (e.g., above key moving averages).

### R2. Top 10 Momentum Portfolio Selection
- Apply the momentum engine across the tracked Indian (NSE / Nifty 500) and US (S&P 500) stock universes.
- Select and rank the Top 10 momentum stocks for the selected universe, providing full transparency into component scores (e.g. raw momentum, slope, $R^2$ fit, volatility penalty, overall rank).
- Include portfolio allocation recommendations (e.g. equal weight vs. inverse-volatility / risk-parity weighting) and clear rebalancing / exit criteria when a stock falls out of momentum rank.

### R3. User Interface and CLI Integration
- Integrate a dedicated "Momentum Portfolio" tab/view into the existing Streamlit dashboard (`src/app.py`), showing interactive ranking tables, momentum factor breakdowns, performance metrics, and individual stock momentum charts.
- Provide CLI entry points and report generator (e.g., standalone command and script options in `run_india.sh` / `run_usa.sh` or a dedicated runner) with clear logging and exportable outputs (CSV/JSON/Markdown).

### R4. Verification & Validation
- Implement automated test suites (unit tests for math/momentum calculations, universe handling, and edge cases such as missing price history or stock splits).
- Include verification scripts to execute against live/sample data for both Indian and US universes and verify reproducibility and runtime performance.

## Acceptance Criteria

### Algorithm & Research Implementation
- [ ] Quantitative momentum engine implements documented research-backed momentum metrics (e.g. Clenow exponential regression slope × $R^2$, 12-1 month return, and volatility normalization).
- [ ] Stock ranking handles edge cases gracefully (insufficient history, low liquidity, NaN data).

### Portfolio Ranking & Selection
- [ ] Able to run on both Indian (`NSE`/Nifty 500) and US (`S&P 500`) stock universes and output the top 10 momentum constituents with rank metrics.
- [ ] Outputs detailed scoring breakdowns and suggested portfolio weights for the top 10 assets.

### UI & CLI Usability
- [ ] Streamlit web application (`src/app.py`) includes a dedicated "Momentum Portfolio" tab displaying the Top 10 rankings, metrics, and visualization charts.
- [ ] CLI command exists to execute momentum screening and output results to console and file (CSV/log).

### Automated Testing & Verification
- [ ] Automated pytest test suite passes covering momentum scoring logic, data fetching edge cases, and portfolio ranking.
- [ ] End-to-end execution script completes successfully without errors on both market universes.

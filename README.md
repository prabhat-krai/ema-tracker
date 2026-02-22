# Investment Screener - EMA TA Rules

> **Note**: This project was built using **Antigravity + Gemini 3 Pro**.

Automated stock screener for Indian and US markets using EMA-based technical analysis rules.

## Features

- Screens Nifty 100 + Nifty Midcap 150 (250 stocks) by default
- Optional USA mode for Top 100 US stocks (`-USA` / `--usa`)
- Uses free yfinance data (no API key required)
- Implements EMA-based TA Rules flowchart
- Outputs buy/sell signals for manual trading

## TA Rules Flowchart

The screener implements the following decision tree:

```
                    ┌─────────────────────────┐
                    │  Are EMAs converging?   │
                    │  (10W, 20W, 40W within   │
                    │   3% of each other)      │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
             YES                                  NO
              │                                   │
              ▼                                   ▼
    ┌─────────────────────┐             ┌─────────────────────┐
    │ Has it broken       │             │ Has it broken       │
    │ support?            │             │ 40W EMA?            │
    └─────────┬───────────┘             └─────────┬───────────┘
              │                                   │
       ┌──────┴──────┐                     ┌──────┴──────┐
      YES            NO                   YES            NO
       │              │                    │              │
       ▼              ▼                    ▼              ▼
   ┌───────┐  ┌─────────────────┐     ┌───────┐  ┌─────────────────┐
   │ EXIT  │  │ Has it broken   │     │ EXIT  │  │ Has it broken   │
   │  🔴   │  │ resistance?     │     │  🔴   │  │ 20W EMA?        │
   └───────┘  └────────┬────────┘     └───────┘  └────────┬────────┘
                       │                                  │
                ┌──────┴──────┐                    ┌──────┴──────┐
               YES            NO                  YES            NO
                │              │                   │              │
                ▼              ▼                   ▼              ▼
         ┌──────────┐   ┌───────────┐      ┌───────────┐  ┌─────────────────┐
         │ BULLISH  │   │   WAIT    │      │ CAUTIOUS  │  │ Has it broken   │
         │   ✅     │   │   🟡      │      │    🟠     │  │ 10W EMA?        │
         └──────────┘   └───────────┘      └───────────┘  └────────┬────────┘
                                                                   │
                                                            ┌──────┴──────┐
                                                           YES            NO
                                                            │              │
                                                            ▼              ▼
                                                     ┌───────────┐  ┌───────────┐
                                                     │  FADING   │  │ HOLD/ADD  │
                                                     │    🟣     │  │    🟢     │
                                                     └───────────┘  └───────────┘
```

### Decision Logic Summary

| Condition | Signal | Action |
|-----------|--------|--------|
| EMAs converging + broke support | 🔴 EXIT | Sell immediately |
| EMAs converging + broke resistance | ✅ BULLISH | Buy signal |
| EMAs converging + no breakout | 🟡 WAIT | Watch for breakout |
| Not converging + below 40W EMA | 🔴 EXIT | Sell immediately |
| Not converging + below 20W EMA | 🟠 CAUTIOUS | Reduce exposure |
| Not converging + below 10W EMA | 🟣 FADING | Momentum weakening |
| Not converging + above all EMAs | 🟢 HOLD/ADD | Strong - maintain/add |

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Run (All 250 Stocks)
```bash
./run_india.sh
```
Or manually:
```bash
source venv/bin/activate
python -m src.main
```

### Run USA Market
```bash
./run_usa.sh
```

### Run Both Markets
```bash
./run_both.sh
```

### Market Selection

- Default mode (no market flag): India universe (Nifty 100 + Nifty Midcap 150)
- USA mode: add `-USA` or `--usa` to use the USA Top 100 universe
- `--tickers` overrides the universe list:
  - India tickers: use `--tickers` without `-USA`
  - USA tickers: use `--tickers` with `-USA`

### Command Line Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `-n`, `--stocks` | Process top N stocks from the selected universe | `python -m src.main -n 10` |
| `-t`, `--tickers` | Process specific comma-separated tickers | `python -m src.main -t RELIANCE,TCS` |
| `-b`, `--backtest` | Run backtest on historical data (last 1 year) | `python -m src.main --backtest` |
| `-USA`, `--usa` | Use Top 100 USA stock universe | `python -m src.main -USA` |
| `-ga`, `--ga` | Run Action Generator on existing logs only (Skip scan) | `python -m src.main --ga` |
| `-d`, `--delay` | Delay in seconds between API calls (default: 2) | `python -m src.main -d 1` |
| `-y`, `--years` | Number of years for backtest (default: 1) | `python -m src.main --backtest --years 5` |
| `-v`, `--verbose` | Enable verbose debug logging | `python -m src.main -v` |

### Examples

**1. Analyze top 10 stocks only:**
```bash
python -m src.main -n 10
```

**2. Analyze specific stocks:**
```bash
python -m src.main -t RELIANCE,INFY,TATAMOTORS
```

**3. Run Backtest on specific stocks (Simulates last 1 year):**
```bash
python -m src.main --backtest -t MTARTECH
```

**4. Run Backtest on top 50 stocks:**
```bash
python -m src.main --backtest -n 50
```

**5. Analyze Top 100 USA stocks:**
```bash
python -m src.main -USA
```

**6. Analyze specific USA stocks:**
```bash
python -m src.main -USA -t AAPL,MSFT,NVDA
```

**7. Run Backtest on USA Top 100 (or subset):**
```bash
python -m src.main -USA --backtest
python -m src.main -USA --backtest -n 25
```

**8. Run 5-year Backtest:**
```bash
python -m src.main --backtest --years 5 -t RELIANCE
```

## Signal Types

| Signal | Meaning |
|--------|---------|
| ✅ BULLISH | Buy signal - resistance breakout |
| 🔴 EXIT | Sell signal - broken support/EMA |
| 🟠 CAUTIOUS | Reduce exposure |
| 🟣 FADING | Momentum weakening |
| 🟢 HOLD/ADD | Maintain position |
| 🟡 WAIT | Watch list |

## Output

Signals are logged to:
- Console (real-time)
- `logs/INDIA_DD-MM-YYYY.log` (or `USA_DD-MM-YYYY.log`)

### Actionable Reports
After scanning, the tool will automatically locate the previous week's log file and compare it against today's snapshot.

Any **transitions** in state will be logged in `actions/[MARKET]-ACTIONS_DD-MM-YYYY.csv`.

The actionable transitions documented are:
- `🚀 NEW BUY` (Transition from any state to BULLISH)
- `🚨 NEW SELL` (Transition from any state to EXIT)
- `⚠️ DOWNGRADE` (Transitioning down to CAUTIOUS or FADING)
- `📈 UPGRADE` (Transitioning up to WAIT from EXIT, or to HOLD_ADD from CAUTIOUS)

This cuts through the noise of unchanged stocks and highlights exactly what you need to pay attention to!

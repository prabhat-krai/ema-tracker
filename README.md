# Investment Screener - EMA TA Rules

Automated stock screener for Indian markets that analyzes top 250 companies using EMA-based technical analysis rules.

## Features

- Screens Nifty 100 + Nifty Midcap 150 (250 stocks)
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
         │   🟢     │   │   🟡      │      │    🟠     │  │ 10W EMA?        │
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
| EMAs converging + broke resistance | 🟢 BULLISH | Buy signal |
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
source venv/bin/activate
python -m src.main
```

### Command Line Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `-n`, `--stocks` | Process top N stocks from list | `python -m src.main -n 10` |
| `-t`, `--tickers` | Process specific comma-separated tickers | `python -m src.main -t RELIANCE,TCS` |
| `-b`, `--backtest` | Run backtest on historical data (last 1 year) | `python -m src.main --backtest` |
| `-d`, `--delay` | Delay in seconds between API calls (default: 2) | `python -m src.main -d 1` |
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

## Signal Types

| Signal | Meaning |
|--------|---------|
| 🟢 BULLISH | Buy signal - resistance breakout |
| 🔴 EXIT | Sell signal - broken support/EMA |
| 🟠 CAUTIOUS | Reduce exposure |
| 🟣 FADING | Momentum weakening |
| 🟢 HOLD/ADD | Maintain position |
| 🟡 WAIT | Watch list |

## Output

Signals are logged to:
- Console (real-time)
- `logs/signals_YYYY-MM-DD_HH-MM-SS-ffffff.log`

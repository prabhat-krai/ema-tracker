"""
Backtest module for EMA TA Rules Strategy.
Simulates trading over historical data to evaluate performance.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from . import config
from .technical import analyze_stock
from .ta_rules_engine import analyze_with_ta_rules, SignalResult, Signal

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    # shares: int = 1  # Simplified: 1 share per trade for now, or fixed capital

    @property
    def return_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    trades: List[Trade]


class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings: Dict[str, Trade] = {}  # symbol -> active Trade
        self.closed_trades: List[Trade] = []
        self.log: List[str] = []

    def process_signal(self, date: datetime, result: SignalResult):
        symbol = result.symbol
        price = result.current_price

        entry_signals = {Signal.BULLISH, Signal.HOLD_ADD}
        exit_signals = {Signal.EXIT}

        # Entry/Hold rule: stay invested only while signal remains bullish/hold-add.
        if result.signal in entry_signals:
            if symbol not in self.holdings:
                trade = Trade(symbol=symbol, entry_date=date, entry_price=price)
                self.holdings[symbol] = trade
                signal_type = "BREAKOUT" if result.signal == Signal.BULLISH else "TREND"
                self.log.append(f"{date.strftime('%Y-%m-%d')} | BUY  | {symbol:10} | {price:7.2f} | {signal_type}")
            return

        # Exit rule: close only on explicit bearish breakdown signals.
        if result.signal in exit_signals and symbol in self.holdings:
            trade = self.holdings.pop(symbol)
            trade.exit_date = date
            trade.exit_price = price
            self.closed_trades.append(trade)

            ret = trade.return_pct * 100
            self.log.append(
                f"{date.strftime('%Y-%m-%d')} | SELL | {symbol:10} | {price:.2f} | "
                f"Return: {ret:.2f}% | Signal: {result.signal.value}"
            )

    def get_performance(self, current_prices: Dict[str, float] = None) -> BacktestResult:
        all_trades = self.closed_trades.copy()
        
        # Mark open positions to market if prices provided
        if self.holdings:
            for symbol, trade in self.holdings.items():
                if current_prices and symbol in current_prices:
                    # Create a temporary closed trade for stats
                    temp_trade = Trade(
                        symbol=trade.symbol,
                        entry_date=trade.entry_date,
                        entry_price=trade.entry_price,
                        exit_date=datetime.now(),
                        exit_price=current_prices[symbol]
                    )
                    all_trades.append(temp_trade)
        
        wins = [t for t in all_trades if t.return_pct is not None and t.return_pct > 0]
        losses = [t for t in all_trades if t.return_pct is not None and t.return_pct <= 0]
        
        avg_return = 0.0
        valid_trades = [t for t in all_trades if t.return_pct is not None]
        if valid_trades:
            avg_return = sum(t.return_pct for t in valid_trades) / len(valid_trades)

        return BacktestResult(
            total_trades=len(valid_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(valid_trades) if valid_trades else 0.0,
            total_return=avg_return,
            trades=valid_trades
        )

def run_backtest_for_symbol(
    symbol: str, 
    df: pd.DataFrame, 
    lookback_weeks: int = 52
) -> Portfolio:
    """
    Run backtest for a single symbol over the last N weeks.
    Steps through the dataframe week by week, protecting future data from lookahead bias.
    """
    portfolio = Portfolio()
    
    # We need enough history for EMAs (40 weeks) + Backtest duration
    min_history = config.EMA_PERIODS["long"] + 10
    total_weeks = len(df)
    
    start_index = max(min_history, total_weeks - lookback_weeks)
    
    if start_index >= total_weeks:
        logger.warning(f"{symbol}: Not enough data for backtest")
        return portfolio
        
    # Step through each week
    for i in range(start_index, total_weeks):
        # Slice data up to current week i (inclusive)
        # i+1 because iloc slice end is exclusive
        current_data = df.iloc[:i+1]
        current_date_idx = df.index[i]
        
        # ensure it's a datetime
        if isinstance(current_date_idx, pd.Timestamp):
             current_date = current_date_idx.to_pydatetime()
        else:
             current_date = datetime.now() # Fallback
        
        # Analyze
        indicators = analyze_stock(symbol, current_data)
        if indicators:
            signal_result = analyze_with_ta_rules(indicators)
            
            # Execute logic
            portfolio.process_signal(current_date, signal_result)
            
    return portfolio

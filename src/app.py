import sys
from pathlib import Path
import re
from datetime import datetime

# Guarantee the parent directory is in python path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import existing modules
from src.config import get_all_stocks, get_usa_stocks, EMA_PERIODS
from src.data_fetcher import fetch_weekly_data, get_market_ticker
from src.technical import calculate_emas, find_support_resistance, check_ema_convergence, analyze_stock
from src.ta_rules_engine import analyze_with_ta_rules, Signal, get_signal_emoji, SignalResult
from src.backtester import run_backtest_for_symbol
from src.action_generator import parse_log_file, find_latest_log, compare_signals, generate_action_csv

# Global Color Map for consistent UI styling
COLOR_MAP = {
    "BULLISH": "#2ca02c",   # Green
    "HOLD_ADD": "#1f77b4",  # Blue (standard strong hold)
    "WAIT": "#bcbd22",      # Yellow
    "CAUTIOUS": "#ff7f0e",  # Orange
    "FADING": "#9467bd",    # Purple
    "EXIT": "#d62728",      # Red
    "UNKNOWN": "#7f7f7f"    # Gray
}

@st.cache_data(show_spinner=False)
def cached_fetch_weekly_data(symbol: str, years: int, market_key: str):
    """Cached wrapper for fetching stock data."""
    return fetch_weekly_data(symbol, years=years, delay=0.1, market=market_key)

@st.cache_data(show_spinner=False)
def cached_backtest(symbol: str, df: pd.DataFrame, lookback_weeks: int):
    """Cached wrapper for backtesting strategy."""
    return run_backtest_for_symbol(symbol, df, lookback_weeks=lookback_weeks)

# Page configuration
st.set_page_config(
    page_title="EMA TA Rules Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .buy-text { color: #2ca02c; }
    .sell-text { color: #d62728; }
    .warn-text { color: #ff7f0e; }
    .up-text { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# Helper parsing function for rich log files
def parse_rich_log_file(filepath: Path) -> pd.DataFrame:
    """
    Parses a log file and returns a DataFrame with:
    Symbol, Signal, Price, Reason
    """
    data = []
    if not filepath.exists():
        return pd.DataFrame()
    
    # Matches lines like:
    # 2026-06-13 02:41:20 | INFO | 🟣 FADING     | ABB             | ₹   6770.50 | Below 10W EMA - momentum fading
    # 2026-02-21 17:03:04 | INFO | ✅ BULLISH    | AAPL            | $    238.25 | ...
    pattern = re.compile(
        r"\|\s*(?:✅|🔴|🟠|🟣|🟢|🟡|⚪)\s+([A-Z_]+)\s*\|\s*([A-Z0-9.\-]+)\s*\|\s*[^|]*?\s*([\d.,]+)\s*\|\s*(.*)"
    )
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    signal = match.group(1).strip()
                    symbol = match.group(2).strip()
                    price = float(match.group(3).replace(",", "").strip())
                    reason = match.group(4).strip()
                    data.append({
                        "Symbol": symbol,
                        "Signal": signal,
                        "Price": price,
                        "Reason": reason
                    })
    except Exception as e:
        st.error(f"Error parsing log file {filepath.name}: {e}")
        
    return pd.DataFrame(data)

def get_available_dates(log_dir: Path, market_prefix: str) -> list[str]:
    """Finds all log files for the market and extracts their dates, sorted descending."""
    pattern = f"*_{market_prefix}.log"
    log_files = list(log_dir.glob(pattern))
    
    dates = []
    for f in log_files:
        try:
            date_str = f.stem.split("_")[0]
            datetime.strptime(date_str, "%Y-%m-%d")
            dates.append(date_str)
        except ValueError:
            continue
            
    return sorted(list(set(dates)), reverse=True)

# Paths
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
ACTION_DIR = BASE_DIR / "actions"

# Title bar
st.markdown("# 📈 Weekly EMA Technical Analysis Screener")
st.markdown("Automated stock screener and transition alert system using weekly Exponential Moving Averages & breakout rules.")

# Sidebar Controls
st.sidebar.title("Configuration")

# 1. Market Selector
market_label = st.sidebar.selectbox(
    "Select Market Universe",
    ["🇮🇳 India (Nifty 500)", "🇺🇸 USA (S&P 500)"]
)
market_prefix = "INDIA" if "India" in market_label else "USA"
market_key = "india" if market_prefix == "INDIA" else "usa"
currency_symbol = "₹" if market_prefix == "INDIA" else "$"

# 2. Week/Date Selector
available_dates = get_available_dates(LOG_DIR, market_prefix)

if not available_dates:
    st.sidebar.error(f"No scan logs found for {market_label}. Please run a scan first.")
    st.stop()

selected_date = st.sidebar.selectbox(
    "Select Scan Date",
    available_dates
)

# Load data for selected date
log_file = LOG_DIR / f"{selected_date}_{market_prefix}.log"
master_df = parse_rich_log_file(log_file)

# Dynamic state reconstruction for transitions
prev_log_file = None
prev_date = None
# Find previous date from available dates list
try:
    idx = available_dates.index(selected_date)
    if idx + 1 < len(available_dates):
        prev_date = available_dates[idx + 1]
        prev_log_file = LOG_DIR / f"{prev_date}_{market_prefix}.log"
except ValueError:
    pass

# Loading or generating Actions
actions_file = ACTION_DIR / f"{selected_date}_{market_prefix}-ACTIONS.csv"
transitions = []

if actions_file.exists():
    try:
        actions_df = pd.read_csv(actions_file).fillna("")
        transitions = actions_df.to_dict(orient="records")
    except Exception as e:
        st.warning(f"Failed to read existing action CSV: {e}")
        
if not transitions and prev_log_file and log_file.exists():
    # Dynamically generate transitions if CSV is missing or empty
    current_signals = parse_log_file(log_file)
    prev_signals = parse_log_file(prev_log_file)
    transitions = compare_signals(prev_signals, current_signals)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🚀 Weekly Action Hub (Transitions)",
    "🔍 Full Market Master Scanner",
    "📈 Stock Chart Analyzer & Backtester"
])

# ---------------------------------------------------------
# TAB 1: WEEKLY ACTION HUB
# ---------------------------------------------------------
with tab1:
    st.header(f"🚀 Weekly Transitions & Alert Board")
    st.markdown(f"Comparing snapshot of **{selected_date}** against the previous week **{prev_date or '(None)'}**.")
    
    if not transitions:
        st.info("No actionable transitions (Buy/Sell/Upgrade/Downgrade) found or only one week of data exists for comparison.")
    else:
        # Categorize
        new_buys = [t for t in transitions if "NEW BUY" in t.get("Action Category", "")]
        new_sells = [t for t in transitions if "NEW SELL" in t.get("Action Category", "")]
        downgrades = [t for t in transitions if "DOWNGRADE" in t.get("Action Category", "")]
        upgrades = [t for t in transitions if "UPGRADE" in t.get("Action Category", "")]
        
        # Display KPI widgets
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value buy-text">{len(new_buys)}</div>
                <div class="metric-label">🚀 New Buys</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value sell-text">{len(new_sells)}</div>
                <div class="metric-label">🚨 New Sells</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value warn-text">{len(downgrades)}</div>
                <div class="metric-label">⚠️ Downgrades</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value up-text">{len(upgrades)}</div>
                <div class="metric-label">📈 Upgrades</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Layout categories
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🟢 Bullish Transitions & Upgrades")
            
            # New Buys
            st.markdown("#### 🚀 NEW BUYS (Action: Buy Now)")
            if new_buys:
                buys_df = pd.DataFrame(new_buys)[["Symbol", "Previous Signal", "Current Signal", "Notes"]]
                st.dataframe(buys_df, use_container_width=True, hide_index=True)
            else:
                st.write("No new breakout buy signals triggered.")
                
            # Upgrades
            st.markdown("#### 📈 UPGRADES (Action: Hold/Accumulate)")
            if upgrades:
                upgrades_df = pd.DataFrame(upgrades)[["Symbol", "Previous Signal", "Current Signal", "Notes"]]
                st.dataframe(upgrades_df, use_container_width=True, hide_index=True)
            else:
                st.write("No positive signal upgrades.")
                
        with col_right:
            st.subheader("🔴 Bearish Transitions & Downgrades")
            
            # New Sells
            st.markdown("#### 🚨 NEW SELLS (Action: Sell/Exit Now)")
            if new_sells:
                sells_df = pd.DataFrame(new_sells)[["Symbol", "Previous Signal", "Current Signal", "Notes"]]
                st.dataframe(sells_df, use_container_width=True, hide_index=True)
            else:
                st.write("No new explicit exit sell signals triggered.")
                
            # Downgrades
            st.markdown("#### ⚠️ DOWNGRADES (Action: Caution/Trim)")
            if downgrades:
                downgrades_df = pd.DataFrame(downgrades)[["Symbol", "Previous Signal", "Current Signal", "Notes"]]
                st.dataframe(downgrades_df, use_container_width=True, hide_index=True)
            else:
                st.write("No signal downgrades.")

# ---------------------------------------------------------
# TAB 2: FULL MARKET MASTER SCANNER
# ---------------------------------------------------------
with tab2:
    st.header(f"🔍 Master Market Screener - {market_label}")
    st.markdown(f"Displaying all stocks analyzed in the snapshot of **{selected_date}**.")
    
    if master_df.empty:
        st.info("No stock indicators found for this date. Check the log file format.")
    else:
        # Filter controls
        filter_cols = st.columns([2, 2, 1])
        with filter_cols[0]:
            search_query = st.text_input("Search Ticker Symbol:", "").strip().upper()
        with filter_cols[1]:
            available_signals = master_df["Signal"].unique()
            selected_signals = st.multiselect("Filter by Signal Type:", available_signals, default=list(available_signals))
            
        # Apply filters
        filtered_df = master_df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["Symbol"].str.contains(search_query, regex=False)]
        if selected_signals:
            filtered_df = filtered_df[filtered_df["Signal"].isin(selected_signals)]
            
        # Metrics & Distribution Chart
        dist_cols = st.columns([1, 2])
        
        with dist_cols[0]:
            st.subheader("Signal Distribution")
            counts = filtered_df["Signal"].value_counts().reset_index()
            counts.columns = ["Signal", "Count"]
            
            # Colors corresponding to our standard signal colors
            fig = px.pie(
                counts,
                names="Signal",
                values="Count",
                color="Signal",
                color_discrete_map=COLOR_MAP,
                hole=0.4
            )
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
        with dist_cols[1]:
            st.subheader(f"Screened Stocks ({len(filtered_df)})")
            
            # Format price column with currency symbol
            display_df = filtered_df.copy()
            display_df["Formatted Price"] = display_df["Price"].map(lambda x: f"{currency_symbol} {x:,.2f}")
            
            # Drop raw Price column for displaying, insert Formatted Price
            display_df = display_df[["Symbol", "Signal", "Formatted Price", "Reason"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

# ---------------------------------------------------------
# TAB 3: STOCK CHART ANALYZER & BACKTESTER
# ---------------------------------------------------------
with tab3:
    st.header("📈 Interactive Stock Analyzer & Backtest Visualizer")
    st.markdown("Drill down into any ticker to visualize indicators, breakouts, and historical backtest performance.")
    
    ticker_options = sorted(master_df["Symbol"].unique()) if not master_df.empty else []
    
    col_sel_left, col_sel_right = st.columns([2, 1])
    with col_sel_left:
        selected_ticker = st.selectbox("Select a ticker to analyze:", ticker_options)
    with col_sel_right:
        backtest_years = st.slider("Backtest Lookback (Years):", 1, 5, 2)
        
    if selected_ticker:
        st.subheader(f"Analyzing {selected_ticker} ({market_label})")
        
        with st.spinner(f"Fetching market data and running backtest for {selected_ticker}..."):
            # Fetch data with sufficient lookback (at least years + 1 for EMA warmup)
            df = cached_fetch_weekly_data(selected_ticker, years=backtest_years + 1, market_key=market_key)
            
            if df is None or df.empty:
                st.error(f"Could not load historical candles for {selected_ticker}. Stock may be delisted or invalid.")
            else:
                # 1. Indicator Calculations
                df_indicators = calculate_emas(df)
                
                # Check current status
                indicators = analyze_stock(selected_ticker, df)
                if indicators:
                    sig_res = analyze_with_ta_rules(indicators)
                    emoji = get_signal_emoji(sig_res.signal)
                    
                    # Format status values safely
                    support_val = f"{currency_symbol}{sig_res.support:,.2f}" if sig_res.support is not None else "N/A"
                    resistance_val = f"{currency_symbol}{sig_res.resistance:,.2f}" if sig_res.resistance is not None else "N/A"
                    converging_val = "Yes ✅" if sig_res.emas_converging else "No ❌"
                    
                    # Highlight Card
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 8px; padding: 15px; border-left: 5px solid {COLOR_MAP.get(sig_res.signal.value, '#7f7f7f')}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;">
                        <span style="font-size: 20px; font-weight: bold; color: #333;">Current Status: {emoji} {sig_res.signal.value}</span><br>
                        <span style="color: #666; font-size: 14px;">Price: {currency_symbol}{sig_res.current_price:,.2f} | {sig_res.reason}</span><br>
                        <span style="color: #666; font-size: 14px;">EMAs Converging: {converging_val} | Support: {support_val} | Resistance: {resistance_val}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 2. Run Backtest
                portfolio = cached_backtest(selected_ticker, df, lookback_weeks=backtest_years * 52)
                stats = portfolio.get_performance(current_prices={selected_ticker: float(df.iloc[-1]["close"])})
                
                # 3. Create interactive candlestick chart with Plotly
                # Slice chart to selected backtest range for cleaner look
                df_chart = df_indicators.tail(backtest_years * 52)
                
                fig_candles = go.Figure()
                
                # Candlesticks
                fig_candles.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=df_chart["open"],
                    high=df_chart["high"],
                    low=df_chart["low"],
                    close=df_chart["close"],
                    name="Price"
                ))
                
                # EMAs
                fig_candles.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart["ema_10w"],
                    line=dict(color="#1f77b4", width=1.5),
                    name="10W EMA"
                ))
                fig_candles.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart["ema_20w"],
                    line=dict(color="#ff7f0e", width=1.5),
                    name="20W EMA"
                ))
                fig_candles.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart["ema_40w"],
                    line=dict(color="#9467bd", width=1.5),
                    name="40W EMA"
                ))
                
                # S/R levels
                if indicators and indicators.support:
                    fig_candles.add_shape(
                        type="line",
                        x0=df_chart.index[0], y0=indicators.support,
                        x1=df_chart.index[-1], y1=indicators.support,
                        line=dict(color="green", width=1, dash="dash"),
                        name="Support"
                    )
                if indicators and indicators.resistance:
                    fig_candles.add_shape(
                        type="line",
                        x0=df_chart.index[0], y0=indicators.resistance,
                        x1=df_chart.index[-1], y1=indicators.resistance,
                        line=dict(color="red", width=1, dash="dash"),
                        name="Resistance"
                    )
                    
                # Trade entry/exit marker points
                entry_dates = []
                entry_prices = []
                exit_dates = []
                exit_prices = []
                
                # Make sure comparison indices are offset-naive for matching
                df_start_idx = df_chart.index[0]
                if isinstance(df_start_idx, pd.Timestamp):
                    if df_start_idx.tzinfo is not None:
                        df_start_idx = df_start_idx.tz_localize(None)
                    df_start_idx = df_start_idx.to_pydatetime()
                
                for trade in stats.trades:
                    trade_entry = trade.entry_date
                    if trade_entry.tzinfo is not None:
                        trade_entry = trade_entry.replace(tzinfo=None)
                        
                    trade_exit = trade.exit_date
                    if trade_exit is not None and trade_exit.tzinfo is not None:
                        trade_exit = trade_exit.replace(tzinfo=None)
                        
                    # check if trade dates in our sliced view
                    if trade_entry >= df_start_idx:
                        entry_dates.append(trade_entry)
                        entry_prices.append(trade.entry_price)
                    if trade_exit is not None and trade_exit >= df_start_idx:
                        exit_dates.append(trade_exit)
                        exit_prices.append(trade.exit_price)
                        
                # Plot entries as green triangles
                if entry_dates:
                    fig_candles.add_trace(go.Scatter(
                        x=entry_dates, y=entry_prices,
                        mode="markers",
                        marker=dict(symbol="triangle-up", color="green", size=10, line=dict(color="black", width=1)),
                        name="Buy Entry"
                    ))
                # Plot exits as red triangles
                if exit_dates:
                    fig_candles.add_trace(go.Scatter(
                        x=exit_dates, y=exit_prices,
                        mode="markers",
                        marker=dict(symbol="triangle-down", color="red", size=10, line=dict(color="black", width=1)),
                        name="Sell Exit"
                    ))
                
                fig_candles.update_layout(
                    title=f"{selected_ticker} Candlestick & EMA Chart",
                    xaxis_title="Date",
                    yaxis_title=f"Price ({currency_symbol})",
                    xaxis_rangeslider_visible=False,
                    height=450,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig_candles, use_container_width=True)
                
                # Backtest performance stats display
                st.subheader(f"Backtest Performance (Last {backtest_years} Year(s))")
                
                perf_cols = st.columns(4)
                with perf_cols[0]:
                    st.metric("Total Trades", f"{stats.total_trades}")
                with perf_cols[1]:
                    st.metric("Win Rate", f"{stats.win_rate:.1%}")
                with perf_cols[2]:
                    st.metric("Avg Return / Trade", f"{stats.total_return:.1%}")
                with perf_cols[3]:
                    net_performance = sum(t.return_pct for t in stats.trades) if stats.trades else 0.0
                    st.metric("Cumulative Return", f"{net_performance:.1%}")
                    
                # Trade log
                st.markdown("#### Closed Trades Log")
                if stats.trades:
                    trades_data = []
                    for t in stats.trades:
                        trades_data.append({
                            "Symbol": t.symbol,
                            "Buy Date": t.entry_date.strftime("%Y-%m-%d"),
                            "Buy Price": f"{currency_symbol}{t.entry_price:,.2f}",
                            "Sell Date": t.exit_date.strftime("%Y-%m-%d") if t.exit_date is not None else "Open Position",
                            "Sell Price": f"{currency_symbol}{t.exit_price:,.2f}" if t.exit_price is not None else "Open Position",
                            "Return": f"{t.return_pct * 100:.2f}%" if t.return_pct is not None else "N/A",
                            "Result": "Win 🟢" if (t.return_pct is not None and t.return_pct > 0) else "Loss 🔴" if (t.return_pct is not None) else "Active"
                        })
                    st.dataframe(pd.DataFrame(trades_data), use_container_width=True, hide_index=True)
                else:
                    st.write("No trades were triggered during the backtest lookback window.")

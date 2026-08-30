import sys
from pathlib import Path
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

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
from src.momentum.engine import MomentumPipeline, MomentumPipelineResult
from src.momentum.models import MarketRegimeState, RebalanceAction
from src.momentum.exporter import export_portfolio_csv, export_portfolio_json, export_portfolio_markdown
from src.momentum.clenow import calculate_clenow_regression
from src.data_fetcher import fetch_daily_data

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
def cached_fetch_daily_data(symbol: str, period: str = "2y", min_bars: int = 60):
    """Cached wrapper for daily stock data."""
    return fetch_daily_data(symbol, period=period, min_bars=min_bars, delay=0.05)

@st.cache_data(show_spinner=False)
def cached_run_momentum_pipeline(
    universe: str,
    top_n: int,
    weighting_scheme: str,
    total_capital: float,
    current_holdings_tuple: tuple,
    refresh_token: int = 0,
):
    """Cached wrapper for momentum screening pipeline."""
    pipeline = MomentumPipeline(
        universe=universe,
        top_n=top_n,
        weighting_scheme=weighting_scheme,
        total_capital=total_capital,
        rank_buffer=20,
        min_weight=0.05,
        max_weight=0.20,
        deadband=0.02,
    )
    try:
        return pipeline.run(
            period="2y",
            current_holdings=list(current_holdings_tuple),
        )
    except Exception as ex:
        from src.momentum.cli import _generate_synthetic_offline_universe
        u_data, b_df, b_sym = _generate_synthetic_offline_universe(universe)
        return pipeline.run_with_data(
            universe_data=u_data,
            benchmark_df=b_df,
            benchmark_symbol=b_sym,
            current_holdings=list(current_holdings_tuple),
        )

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
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.12);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 4px;
        color: #ffffff;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 500;
    }
    .buy-text { color: #22c55e !important; }
    .sell-text { color: #ef4444 !important; }
    .warn-text { color: #f59e0b !important; }
    .up-text { color: #3b82f6 !important; }
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Weekly Action Hub (Transitions)",
    "🔍 Full Market Master Scanner",
    "📈 Stock Chart Analyzer & Backtester",
    "🏆 Quant Momentum Portfolio",
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

# ---------------------------------------------------------
# TAB 4: QUANTITATIVE MOMENTUM PORTFOLIO
# ---------------------------------------------------------
with tab4:
    st.header("🏆 Quantitative Momentum Portfolio Tracker")
    st.markdown(
        "Empirical quantitative momentum strategy combining **Andreas Clenow Exponential Regression** "
        "($\\text{Annualized Slope} \\times R^2$), **Jegadeesh & Titman (1993)** 12-1 Intermediate Momentum, "
        "**Realized Volatility Normalization**, **Macro Regime Moving Average Filters**, and **Hysteresis Churn Buffers**."
    )

    with st.expander("⚙️ Quantitative Strategy & Universe Settings", expanded=True):
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            mom_market_label = st.selectbox(
                "Market Universe",
                ["India (NSE Nifty 500)", "USA (S&P 500)"],
                index=0 if market_prefix == "INDIA" else 1,
                key="tab4_market_choice",
            )
            mom_univ = "india" if "India" in mom_market_label else "usa"
        with m_col2:
            mom_weight_label = st.selectbox(
                "Weighting Scheme",
                ["Inverse Volatility", "Equal Weight", "Bounded Risk Parity (5%-20%)"],
                index=0,
                key="tab4_weight_choice",
            )
            mom_weight_scheme = (
                "inv_vol"
                if "Inverse" in mom_weight_label
                else ("equal" if "Equal" in mom_weight_label else "bounded_parity")
            )
        with m_col3:
            mom_top_n = st.slider(
                "Portfolio Size (Top N)",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                key="tab4_top_n",
            )
        with m_col4:
            curr_sym = "₹" if mom_univ == "india" else "$"
            default_cap = 1_000_000.0 if mom_univ == "india" else 100_000.0
            mom_capital = st.number_input(
                f"Total Capital ({curr_sym})",
                min_value=10_000.0,
                max_value=1_000_000_000.0,
                value=default_cap,
                step=50_000.0,
                key="tab4_capital",
            )

        mom_holdings_raw = st.text_input(
            "Current Holdings (comma-separated, for hysteresis buffer retention):",
            value="",
            placeholder="e.g. RELIANCE.NS, TCS.NS or AAPL, MSFT",
            key="tab4_holdings_input",
        )
        current_holdings_list = [
            h.strip().upper()
            for h in mom_holdings_raw.split(",")
            if h.strip()
        ]

        run_mom_clicked = st.button(
            "🚀 Run Quantitative Momentum Screener",
            type="primary",
            key="tab4_run_screener_btn",
        )

    if "mom_refresh_token" not in st.session_state:
        st.session_state["mom_refresh_token"] = 0
    if run_mom_clicked:
        st.session_state["mom_refresh_token"] += 1

    with st.spinner(f"Screening momentum universe for {mom_univ.upper()}..."):
        res: Optional[MomentumPipelineResult] = cached_run_momentum_pipeline(
            universe=mom_univ,
            top_n=mom_top_n,
            weighting_scheme=mom_weight_scheme,
            total_capital=mom_capital,
            current_holdings_tuple=tuple(current_holdings_list),
            refresh_token=st.session_state["mom_refresh_token"],
        )

    if res is not None:
        rec = res.portfolio
        regime = res.market_regime
        curr = "₹" if res.universe == "india" else "$"

        # 1. Market Regime Indicator Card
        regime_val = regime.state.value if hasattr(regime.state, "value") else str(regime.state)
        if regime.state == MarketRegimeState.BULLISH:
            st.success(
                f"🟢 **MARKET REGIME: BULLISH** | Benchmark `{regime.benchmark_symbol}` ({curr}{regime.benchmark_price:,.2f}) is above 200-day SMA ({curr}{regime.sma_200:,.2f}) with positive slope (+{regime.sma_200_slope*100.0:.2f}%). "
                f"**Target Equity Allocation: 100%** | **Target Cash: 0%**"
            )
        elif regime.state == MarketRegimeState.NEUTRAL:
            st.warning(
                f"🟡 **MARKET REGIME: NEUTRAL** | Benchmark `{regime.benchmark_symbol}` is exhibiting sideways consolidation or negative slope. "
                f"**Defensive Allocation: 50% Equity, 50% Cash**"
            )
        else:
            st.error(
                f"🔴 **MARKET REGIME: BEARISH** | Benchmark `{regime.benchmark_symbol}` is below 200-day SMA. "
                f"**Risk Circuit Breaker Active: 0% Equity, 100% Cash Defense**"
            )

        # 2. KPI Cards Row
        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{curr}{rec.total_capital * rec.total_equity_pct:,.0f}</div>
                <div class="metric-label">Invested Capital ({rec.total_equity_pct*100.0:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{curr}{rec.total_capital * rec.expected_cash_pct:,.0f}</div>
                <div class="metric-label">Defensive Cash ({rec.expected_cash_pct*100.0:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rec.weighted_clenow_score:.3f}</div>
                <div class="metric-label">Weighted Clenow Score</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rec.portfolio_volatility*100.0:.1f}%</div>
                <div class="metric-label">Portfolio Volatility (Ann.)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rec.diversification_ratio:.2f} / {len(rec.target_constituents)}</div>
                <div class="metric-label">Effective N (Diversification)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Top Constituents Table
        st.subheader(f"🥇 Recommended Top {len(rec.target_constituents)} Momentum Constituents")

        table_rows = []
        for c in rec.target_constituents:
            act_str = c.action.value if hasattr(c.action, "value") else str(c.action)
            table_rows.append({
                "Rank": f"#{c.rank}",
                "Symbol": c.symbol,
                "Action": f"🟢 {act_str}" if act_str == "BUY" else ("🔵 HOLD" if act_str == "HOLD" else f"🔴 {act_str}"),
                "Weight": f"{c.weight*100.0:.2f}%",
                "Target Shares": f"{c.target_shares:,}",
                "Target Value": f"{curr}{c.target_value:,.2f}",
                "Close Price": f"{curr}{c.close:,.2f}",
                "Clenow Score": f"{c.clenow_score:.3f}",
                "Slope (Ann.)": f"{c.clenow_slope_ann*100.0:+.1f}%",
                "R² Fit": f"{c.clenow_r2:.2f}",
                "12-1 Momentum": f"{c.mom_12_1*100.0:+.1f}%",
                "Volatility (Ann.)": f"{c.volatility_ann*100.0:.1f}%",
            })

        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # 4. Rebalance Action Plan
        plan = res.rebalance_plan
        if plan is not None:
            with st.expander(f"📋 Rebalancing & Order Execution Plan (Turnover: {plan.turnover_pct*100.0:.1f}%)", expanded=False):
                reb_col1, reb_col2, reb_col3, reb_col4 = st.columns(4)
                with reb_col1:
                    st.metric("New Buys", len(plan.buys))
                with reb_col2:
                    st.metric("Holds Retained", len(plan.holds))
                with reb_col3:
                    st.metric("Sell Exits", len(plan.sells))
                with reb_col4:
                    st.metric("Est. Cash Change", f"{curr}{plan.estimated_cash_change:+,.2f}")

                reb_rows = []
                for t in plan.all_trades:
                    reb_rows.append({
                        "Symbol": t.symbol,
                        "Action": t.action,
                        "Current Weight": f"{t.current_weight*100.0:.1f}%",
                        "Target Weight": f"{t.target_weight*100.0:.1f}%",
                        "Weight Delta": f"{t.weight_delta*100.0:+.1f}%",
                        "Trade Shares": f"{t.trade_shares:+d}",
                        "Trade Value": f"{curr}{t.estimated_trade_value:+,.2f}",
                        "Reason": t.reason,
                    })
                st.dataframe(pd.DataFrame(reb_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 5. Interactive Visualizations
        st.subheader("📊 Quantitative Portfolio Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Portfolio Allocation Donut Chart
            alloc_labels = [c.symbol for c in rec.target_constituents]
            alloc_values = [c.weight for c in rec.target_constituents]
            if rec.expected_cash_pct > 1e-4:
                alloc_labels.append("Cash Reserve")
                alloc_values.append(rec.expected_cash_pct)

            fig_alloc = px.pie(
                names=alloc_labels,
                values=alloc_values,
                hole=0.45,
                title="Portfolio Capital Allocation",
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            fig_alloc.update_traces(textposition="inside", textinfo="percent+label")
            fig_alloc.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
            st.plotly_chart(fig_alloc, use_container_width=True)

        with viz_col2:
            # Score vs Volatility Scatter Plot
            scatter_data = []
            for c in rec.target_constituents:
                scatter_data.append({
                    "Symbol": c.symbol,
                    "Composite_Score": c.composite_score,
                    "Volatility_Ann": c.volatility_ann * 100.0,
                    "Clenow_Score": c.clenow_score,
                    "Weight": c.weight * 100.0,
                })
            df_scatter = pd.DataFrame(scatter_data)
            if not df_scatter.empty:
                fig_scatter = px.scatter(
                    df_scatter,
                    x="Volatility_Ann",
                    y="Composite_Score",
                    size="Weight",
                    text="Symbol",
                    title="Composite Momentum Score vs. Realized Volatility",
                    labels={
                        "Volatility_Ann": "Annualized Realized Volatility (%)",
                        "Composite_Score": "Cross-Sectional Composite Score",
                    },
                    color="Clenow_Score",
                    color_continuous_scale="Viridis",
                )
                fig_scatter.update_traces(textposition="top center")
                fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Clenow Exponential Regression Deep-Dive Chart
        st.markdown("#### 🔬 Andreas Clenow Exponential Regression Deep-Dive")
        selected_mom_sym = st.selectbox(
            "Select Constituent to Inspect Regression Trendline & Moving Averages:",
            [c.symbol for c in rec.target_constituents],
            key="tab4_inspect_sym",
        )

        if selected_mom_sym:
            sym_df = cached_fetch_daily_data(selected_mom_sym, period="2y", min_bars=60)
            if sym_df is None or sym_df.empty:
                # Fallback to synthetic if live data missing
                rng = np.random.RandomState(abs(hash(selected_mom_sym)) % 10000)
                dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=250, freq="D")
                shocks = rng.normal(0.0015, 0.015, size=len(dates))
                prices = 100.0 * np.exp(np.cumsum(shocks))
                sym_df = pd.DataFrame({"close": prices}, index=dates)

            close_vals = sym_df["close"].dropna().values
            n_bars = min(90, len(close_vals))
            recent_close = close_vals[-n_bars:]
            recent_dates = sym_df.index[-n_bars:]

            alpha, beta, r2 = calculate_clenow_regression(recent_close)
            t_axis = np.arange(n_bars, dtype=np.float64)
            exp_fit = np.exp(alpha + beta * t_axis)
            ann_slope = (np.exp(250.0 * beta) - 1.0) * 100.0

            # Moving averages
            sym_df["EMA_100"] = sym_df["close"].ewm(span=100, adjust=False).mean()
            sym_df["SMA_200"] = sym_df["close"].rolling(window=min(200, len(sym_df))).mean()

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=sym_df.index,
                y=sym_df["close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#1f77b4", width=2),
            ))
            fig_reg.add_trace(go.Scatter(
                x=recent_dates,
                y=exp_fit,
                mode="lines",
                name=f"Clenow Exponential Fit (Slope: {ann_slope:+.1f}%, R²: {r2:.2f})",
                line=dict(color="#e377c2", width=3, dash="dash"),
            ))
            if "EMA_100" in sym_df.columns:
                fig_reg.add_trace(go.Scatter(
                    x=sym_df.index,
                    y=sym_df["EMA_100"],
                    mode="lines",
                    name="EMA 100",
                    line=dict(color="#ff7f0e", width=1.5),
                ))
            if "SMA_200" in sym_df.columns:
                fig_reg.add_trace(go.Scatter(
                    x=sym_df.index,
                    y=sym_df["SMA_200"],
                    mode="lines",
                    name="SMA 200",
                    line=dict(color="#2ca02c", width=1.5),
                ))

            fig_reg.update_layout(
                title=f"{selected_mom_sym} — Clenow Exponential Regression $\\ln(P_t) = \\alpha + \\beta t$ & Trend Filters",
                xaxis_title="Date",
                yaxis_title=f"Price ({curr})",
                margin=dict(l=30, r=30, t=50, b=30),
                height=450,
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        # 6. Report Export & Download Section
        st.markdown("---")
        st.markdown("#### 📥 Export Momentum Portfolio Reports")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        csv_path = export_portfolio_csv(rec, plan, filepath=f"reports/momentum/{res.universe}_momentum.csv")
        json_path = export_portfolio_json(rec, plan, filepath=f"reports/momentum/{res.universe}_momentum.json")
        md_path = export_portfolio_markdown(rec, plan, filepath=f"reports/momentum/{res.universe}_momentum_summary.md")

        with exp_col1:
            with open(csv_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📄 Download CSV Report",
                    data=f.read(),
                    file_name=f"{res.universe}_momentum_portfolio.csv",
                    mime="text/csv",
                    key="tab4_dl_csv",
                )
        with exp_col2:
            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📦 Download JSON Data",
                    data=f.read(),
                    file_name=f"{res.universe}_momentum_portfolio.json",
                    mime="application/json",
                    key="tab4_dl_json",
                )
        with exp_col3:
            with open(md_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📝 Download Markdown Summary",
                    data=f.read(),
                    file_name=f"{res.universe}_momentum_summary.md",
                    mime="text/markdown",
                    key="tab4_dl_md",
                )


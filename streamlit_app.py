import streamlit as st
import requests
import pandas as pd
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Advanced AI Trading Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API URL
API_URL = st.secrets.get("API_URL", "https://ai-trading-system-production.up.railway.app")

# Forex pairs
MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
MINOR_PAIRS = ['NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDCAD', 'AUDCHF']
ALL_PAIRS = MAJOR_PAIRS + MINOR_PAIRS

def fetch_data(endpoint):
    """Fetch data from API"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def format_currency(value):
    """Format currency values"""
    try:
        return f"${float(value):,.2f}" if value else "$0.00"
    except:
        return "$0.00"

def format_price(value, symbol):
    """Format price based on symbol"""
    try:
        decimals = 3 if 'JPY' in symbol else 5
        return f"{float(value):.{decimals}f}"
    except:
        return "0.00000"

def create_price_chart(data, symbol):
    """Create price chart for a symbol"""
    if not data:
        return None
    
    df = pd.DataFrame(data)
    if 'price' not in df.columns:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price'],
        mode='lines',
        name=symbol,
        line=dict(color='#8B5CF6', width=2)
    ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def main():
    # Header
    st.markdown("# 🤖 Advanced AI Trading System")
    st.markdown("**Neural Network | Multi-Currency | Auto-Trading**")
    
    # Fetch data
    account_info = fetch_data("account-info")
    market_data = fetch_data("market-data")
    predictions = fetch_data("ai-predictions")
    trading_signals = fetch_data("trading-signals")
    system_status = fetch_data("system-status")
    
    # Connection status
    active_pairs = system_status.get("active_pairs", 0)
    if active_pairs > 0:
        st.success(f"🟢 {active_pairs} Currency Pairs Active")
    else:
        st.error("🔴 Waiting for MT5 Connection")
    
    # Account Overview
    st.subheader("📊 Account Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Balance", format_currency(account_info.get('balance', 0)))
    with col2:
        st.metric("📈 Equity", format_currency(account_info.get('equity', 0)))
    with col3:
        profit = account_info.get('profit', 0)
        st.metric("💵 P&L", format_currency(profit), delta=f"{profit:.2f}")
    with col4:
        st.metric("🤖 Predictions", system_status.get('total_predictions', 0))
    
    # Auto-Trading Control
    st.subheader("🤖 Auto-Trading Control")
    auto_trading = system_status.get('auto_trading_enabled', False)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🟢 Enable Auto-Trading" if not auto_trading else "🔴 Disable Auto-Trading"):
            try:
                response = requests.post(f"{API_URL}/toggle-auto-trading", 
                                       json={"enabled": not auto_trading})
                if response.status_code == 200:
                    st.success("Auto-trading toggled successfully!")
                    st.rerun()
            except:
                st.error("Failed to toggle auto-trading")
    
    with col2:
        status_color = "🟢" if auto_trading else "🔴"
        st.info(f"{status_color} Auto-Trading: {'ENABLED' if auto_trading else 'DISABLED'}")
    
    # Charts Section
    st.subheader("📈 Multi-Currency Charts")
    
    # Major Pairs
    st.markdown("### Major Currency Pairs")
    cols = st.columns(3)
    
    for i, pair in enumerate(MAJOR_PAIRS):
        with cols[i % 3]:
            pair_data = market_data.get(pair, [])
            if pair_data:
                latest_price = pair_data[-1].get('price', 0)
                st.metric(pair, format_price(latest_price, pair))
                
                # Create mini chart
                if len(pair_data) > 1:
                    chart_data = pd.DataFrame(pair_data)
                    st.line_chart(chart_data.set_index('timestamp')['price'], height=150)
                else:
                    st.info("Building chart...")
            else:
                st.info(f"Waiting for {pair} data...")
    
    # Trading Signals
    st.subheader("🎯 Recent Trading Signals")
    
    if trading_signals:
        signals_df = pd.DataFrame(trading_signals[-10:])  # Last 10 signals
        
        for _, signal in signals_df.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    action_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal.get('action', 'HOLD'), "🟡")
                    st.write(f"**{signal.get('symbol')}** {action_color} {signal.get('action')}")
                
                with col2:
                    st.write(f"Entry: {format_price(signal.get('entry_price', 0), signal.get('symbol'))}")
                
                with col3:
                    confidence = signal.get('confidence', 0) * 100
                    st.write(f"Confidence: {confidence:.1f}%")
                
                with col4:
                    st.write(f"Size: {signal.get('position_size', 0)} lots")
                
                st.markdown("---")
    else:
        st.info("No trading signals generated yet")
    
    # AI Predictions Summary
    st.subheader("🧠 AI Predictions Summary")
    
    if predictions:
        pred_cols = st.columns(len(ALL_PAIRS[:6]))  # Show first 6 pairs
        
        for i, pair in enumerate(ALL_PAIRS[:6]):
            if pair in predictions and predictions[pair]:
                with pred_cols[i]:
                    latest_pred = predictions[pair][-1]
                    action = latest_pred.get('prediction', {}).get('action', 'HOLD')
                    confidence = latest_pred.get('prediction', {}).get('confidence', 0)
                    
                    action_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "🟡")
                    
                    st.metric(
                        pair,
                        f"{action_color} {action}",
                        f"{confidence*100:.0f}% confidence"
                    )
    
    # System Statistics
    st.subheader("⚙️ System Statistics")
    
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Active Pairs", f"{system_status.get('active_pairs', 0)}/20")
    
    with stat_cols[1]:
        st.metric("Total Predictions", system_status.get('total_predictions', 0))
    
    with stat_cols[2]:
        st.metric("Trading Signals", len(trading_signals))
    
    with stat_cols[3]:
        demo_mode = system_status.get('demo_mode', True)
        st.metric("Trading Mode", "DEMO" if demo_mode else "LIVE")
    
    # Pattern Analysis
    st.subheader("📊 Pattern Analysis")
    
    pattern_data = fetch_data("pattern-analysis")
    if pattern_data:
        pattern_cols = st.columns(3)
        
        for i, (symbol, pattern) in enumerate(list(pattern_data.items())[:6]):
            with pattern_cols[i % 3]:
                if pattern and 'pattern' in pattern:
                    pattern_info = pattern['pattern']
                    pattern_type = pattern_info.get('pattern', 'Unknown')
                    strength = pattern_info.get('strength', 0)
                    
                    st.write(f"**{symbol}**")
                    st.write(f"Pattern: {pattern_type}")
                    st.progress(strength, text=f"Strength: {strength:.1%}")
    else:
        st.info("Analyzing market patterns...")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧠 Neural AI Processing | 📊 Multi-Currency Analysis | ⚡ Real-time Predictions | 🛡️ Risk Management</p>
        <p style="color: #ef4444;">⚠️ Trading involves substantial risk of loss</p>
        <p>Last update: {}</p>
    </div>
    """.format(datetime.now().strftime('%H:%M:%S')), unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(1)

# Auto-refresh every 10 seconds
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

if st.button("🔄 Refresh Data") or st.session_state.refresh_count % 30 == 0:
    st.rerun()

st.session_state.refresh_count += 1

if __name__ == "__main__":
    main()

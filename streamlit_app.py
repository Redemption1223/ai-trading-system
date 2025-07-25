import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

# Configure page
st.set_page_config(
    page_title="AI Trading Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .status-connected {
        background-color: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-disconnected {
        background-color: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

def fetch_data(endpoint, default=None):
    """Fetch data from API with error handling"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return default or []
    except requests.exceptions.RequestException as e:
        st.warning(f"Connection error: {str(e)}")
        return default or []
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return default or []

def format_currency(value):
    """Format currency values"""
    return f"${value:,.2f}" if isinstance(value, (int, float)) else "$0.00"

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Neural AI Trading System</div>', unsafe_allow_html=True)
    
    # Connection status
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        system_status = fetch_data("system-status", {})
        is_connected = system_status.get("market_data_count", 0) > 0
        
        if is_connected:
            st.markdown('<div class="status-connected">🟢 MT5 Connected & Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-disconnected">🔴 Waiting for MT5 Connection</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Auto-refresh setup
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    # Create containers for real-time updates
    metrics_container = st.container()
    main_content_container = st.container()
    
    with metrics_container:
        # Account metrics
        st.subheader("📊 Account Overview")
        
        account_info = fetch_data("account-info", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            balance = account_info.get('balance', 0)
            st.metric("💰 Balance", format_currency(balance))
        
        with col2:
            equity = account_info.get('equity', 0)
            st.metric("📈 Equity", format_currency(equity))
        
        with col3:
            profit = account_info.get('profit', 0)
            delta = f"+{profit:.2f}" if profit > 0 else f"{profit:.2f}"
            st.metric("💵 Profit/Loss", format_currency(profit), delta=delta)
        
        with col4:
            ai_status = "🟢 Active" if system_status.get("ai_trained", False) else "🟡 Learning"
            st.metric("🤖 AI Status", ai_status)
    
    with main_content_container:
        # Main dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Real-time Price Chart")
            
            market_data = fetch_data("market-data", [])
            
            if market_data and len(market_data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(market_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Create price chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['price'],
                    mode='lines+markers',
                    name='Price',
                    line=dict(color='#00ff88', width=2),
                    marker=dict(size=4)
                ))
                
                fig.update_layout(
                    title=f"{df.iloc[-1]['symbol'] if not df.empty else 'EURUSD'} - Live Price",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    template="plotly_dark",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show latest price info
                if not df.empty:
                    latest = df.iloc[-1]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Current Price", f"{latest['price']:.5f}")
                    with col_b:
                        st.metric("Volume", f"{latest['volume']:,}")
                    with col_c:
                        price_change = df['price'].pct_change().iloc[-1] * 100
                        st.metric("Change %", f"{price_change:.3f}%")
            else:
                st.info("📡 Waiting for market data from MT5...")
                st.markdown("""
                **To connect MT5:**
                1. Install the Expert Advisor in MetaEditor
                2. Set the webhook URL to your API endpoint
                3. Attach EA to any chart
                4. Data will appear here automatically
                """)
        
        with col2:
            # AI Predictions
            st.subheader("🧠 AI Predictions")
            
            predictions = fetch_data("ai-predictions", [])
            
            if predictions:
                for i, pred in enumerate(predictions[-5:]):
                    prediction_data = pred.get('prediction', {})
                    action = prediction_data.get('action', 'HOLD')
                    confidence = prediction_data.get('confidence', 0.5)
                    reasoning = prediction_data.get('reasoning', 'No reasoning')
                    
                    # Color coding
                    if action == "BUY":
                        color = "🟢"
                        bg_color = "#10b981"
                    elif action == "SELL":
                        color = "🔴"
                        bg_color = "#ef4444"
                    else:
                        color = "🟡"
                        bg_color = "#f59e0b"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                            <strong>{color} {action}</strong><br>
                            Confidence: {confidence:.1%}<br>
                            <small>{reasoning}</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🤖 AI is analyzing market data...")
            
            # Market News
            st.subheader("📰 Market News")
            
            news = fetch_data("news", [])
            
            for article in news[:3]:
                with st.expander(f"📄 {article.get('title', 'News Article')[:50]}..."):
                    st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                    st.write(f"**Description:** {article.get('description', 'No description')}")
                    if article.get('sentiment'):
                        sentiment_color = "🟢" if article['sentiment'] == "bullish" else "🔴" if article['sentiment'] == "bearish" else "🟡"
                        st.write(f"**Sentiment:** {sentiment_color} {article['sentiment'].title()}")
        
        # System Information
        st.markdown("---")
        st.subheader("⚙️ System Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.info(f"**Market Data Points:** {system_status.get('market_data_count', 0)}")
        
        with info_col2:
            st.info(f"**AI Predictions Made:** {system_status.get('predictions_count', 0)}")
        
        with info_col3:
            last_update = system_status.get('last_update', 'Never')
            if last_update != 'Never':
                try:
                    last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    last_update = last_update.strftime('%H:%M:%S')
                except:
                    pass
            st.info(f"**Last Update:** {last_update}")
        
        # Risk Management
        st.subheader("🛡️ Risk Management")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.metric("Max Drawdown", "5%", help="Maximum allowed account drawdown")
        with risk_col2:
            st.metric("Position Size", "Auto", help="AI-calculated position sizing")
        with risk_col3:
            st.metric("Daily Risk", "2%", help="Maximum daily risk exposure")
    
    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.session_state.refresh_counter += 1
    st.rerun()

if __name__ == "__main__":
    main()

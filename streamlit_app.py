import streamlit as st
import requests
import pandas as pd
import time
import json
from datetime import datetime

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
    .prediction-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    .buy-signal {
        background-color: #10b981;
        color: white;
    }
    .sell-signal {
        background-color: #ef4444;
        color: white;
    }
    .hold-signal {
        background-color: #f59e0b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = st.secrets.get("API_URL", "https://ai-trading-system-production.up.railway.app")

def fetch_data(endpoint, default=None):
    """Fetch data from API with error handling"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=10)
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
    try:
        return f"${float(value):,.2f}" if value else "$0.00"
    except:
        return "$0.00"

def create_simple_chart(market_data):
    """Create a simple line chart"""
    if not market_data:
        return None
    
    df = pd.DataFrame(market_data)
    if 'price' in df.columns:
        return st.line_chart(df.set_index('timestamp')['price'])
    return None

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Neural AI Trading System</div>', unsafe_allow_html=True)
    
    # Connection status
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            system_status = fetch_data("system-status", {})
            is_connected = system_status.get("market_data_count", 0) > 0
            
            if is_connected:
                st.markdown('<div class="status-connected">🟢 MT5 Connected & Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-disconnected">🔴 Waiting for MT5 Connection</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="status-disconnected">🔴 API Connection Error</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Account metrics
    st.subheader("📊 Account Overview")
    
    try:
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
            ai_status = "🟢 Active" if system_status.get("ai_active", False) else "🟡 Learning"
            st.metric("🤖 AI Status", ai_status)
    except Exception as e:
        st.error(f"Error loading account info: {e}")
    
    # Main dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Real-time Price Chart")
        
        try:
            market_data = fetch_data("market-data", [])
            
            if market_data and len(market_data) > 0:
                # Convert to DataFrame for display
                df = pd.DataFrame(market_data)
                
                # Show latest price info
                if not df.empty:
                    latest = df.iloc[-1]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Current Price", f"{latest.get('price', 0):.5f}")
                    with col_b:
                        st.metric("Volume", f"{latest.get('volume', 0):,}")
                    with col_c:
                        if len(df) > 1:
                            price_change = ((latest.get('price', 0) - df.iloc[-2].get('price', 0)) / df.iloc[-2].get('price', 1)) * 100
                            st.metric("Change %", f"{price_change:.3f}%")
                
                # Simple line chart
                if 'price' in df.columns and len(df) > 1:
                    st.line_chart(df['price'])
                else:
                    st.info("📊 Building chart with incoming data...")
                    
            else:
                st.info("📡 Waiting for market data from MT5...")
                st.markdown("""
                **To connect MT5:**
                1. Install the Expert Advisor in MetaEditor
                2. Set the webhook URL to your API endpoint
                3. Attach EA to any chart
                4. Data will appear here automatically
                """)
        except Exception as e:
            st.error(f"Error loading market data: {e}")
    
    with col2:
        # AI Predictions
        st.subheader("🧠 AI Predictions")
        
        try:
            predictions = fetch_data("ai-predictions", [])
            
            if predictions:
                for pred in predictions[-5:]:
                    prediction_data = pred.get('prediction', {})
                    action = prediction_data.get('action', 'HOLD')
                    confidence = prediction_data.get('confidence', 0.5)
                    reasoning = prediction_data.get('reasoning', 'No reasoning')
                    
                    # Create prediction box
                    css_class = f"{action.lower()}-signal"
                    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
                    
                    st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        {emoji} <strong>{action}</strong><br>
                        Confidence: {confidence:.1%}<br>
                        <small>{reasoning}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🤖 AI is analyzing market data...")
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
        
        # Market News
        st.subheader("📰 Market News")
        
        try:
            news = fetch_data("news", [])
            
            for article in news[:3]:
                with st.expander(f"📄 {article.get('title', 'News Article')[:50]}..."):
                    st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                    st.write(f"**Description:** {article.get('description', 'No description')}")
                    if article.get('sentiment'):
                        sentiment_color = "🟢" if article['sentiment'] == "bullish" else "🔴" if article['sentiment'] == "bearish" else "🟡"
                        st.write(f"**Sentiment:** {sentiment_color} {article['sentiment'].title()}")
        except Exception as e:
            st.error(f"Error loading news: {e}")
    
    # System Information
    st.markdown("---")
    st.subheader("⚙️ System Information")
    
    try:
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
    except Exception as e:
        st.error(f"Error loading system info: {e}")
    
    # Risk Management
    st.subheader("🛡️ Risk Management")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.metric("Max Drawdown", "5%", help="Maximum allowed account drawdown")
    with risk_col2:
        st.metric("Position Size", "Auto", help="AI-calculated position sizing")
    with risk_col3:
        st.metric("Daily Risk", "2%", help="Maximum daily risk exposure")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 AI Trading System • Real-time ML Processing • Secure MT5 Integration</p>
        <p>Last refresh: {}</p>
    </div>
    """.format(datetime.now().strftime('%H:%M:%S')), unsafe_allow_html=True)

# Auto-refresh
if st.button("🔄 Refresh Data"):
    st.rerun()

# Auto-refresh every 30 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Auto refresh every 30 seconds
if time.time() - st.session_state.last_refresh > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()

if __name__ == "__main__":
    main()

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Trading Dashboard", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API URL
API_URL = st.secrets.get("API_URL", "https://ai-trading-system-production.up.railway.app")

def fetch_data_safe(endpoint):
    """Safely fetch data from API with timeout"""
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed"}
    except Exception as e:
        return {"error": str(e)}

def main():
    # Header
    st.title("🤖 Advanced AI Trading System")
    st.markdown("**Multi-Currency Neural Network Trading Platform**")
    
    # Test API Connection
    st.subheader("🔗 API Connection Test")
    
    with st.spinner("Testing API connection..."):
        api_test = fetch_data_safe("")
    
    if "error" in api_test:
        st.error(f"❌ API Connection Failed: {api_test['error']}")
        st.info("🔧 Troubleshooting: Check if your Railway API is running")
        
        # Show API URL for debugging
        st.code(f"API URL: {API_URL}")
        
        # Manual test button
        if st.button("🔄 Retry Connection"):
            st.rerun()
            
    else:
        st.success("✅ API Connected Successfully!")
        
        # Show API response
        st.json(api_test)
    
    # Account Info Section
    st.subheader("💰 Account Information")
    
    account_data = fetch_data_safe("account-info")
    
    if "error" not in account_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            balance = account_data.get('balance', 0)
            st.metric("Balance", f"${balance:,.2f}")
        
        with col2:
            equity = account_data.get('equity', 0)
            st.metric("Equity", f"${equity:,.2f}")
        
        with col3:
            profit = account_data.get('profit', 0)
            st.metric("Profit/Loss", f"${profit:,.2f}")
    else:
        st.warning(f"Account data error: {account_data['error']}")
    
    # Market Data Section
    st.subheader("📊 Market Data")
    
    market_data = fetch_data_safe("market-data")
    
    if "error" not in market_data and market_data:
        # Show available pairs
        st.write(f"**Available pairs:** {len(market_data)} currencies")
        
        # Display each pair's latest data
        for symbol, data in market_data.items():
            if data:  # Check if data exists for this pair
                latest = data[-1]  # Get latest data point
                
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    st.write(f"**{symbol}**")
                
                with col2:
                    price = latest.get('price', 0)
                    decimals = 3 if 'JPY' in symbol else 5
                    st.write(f"Price: {price:.{decimals}f}")
                
                with col3:
                    timestamp = latest.get('timestamp', '')
                    st.write(f"Updated: {timestamp}")
    else:
        st.info("📡 Waiting for market data from MT5...")
        if "error" in market_data:
            st.error(f"Market data error: {market_data['error']}")
    
    # System Status
    st.subheader("⚙️ System Status")
    
    system_status = fetch_data_safe("system-status")
    
    if "error" not in system_status:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            active_pairs = system_status.get('active_pairs', 0)
            total_pairs = system_status.get('total_pairs', 20)
            st.metric("Active Pairs", f"{active_pairs}/{total_pairs}")
        
        with col2:
            predictions = system_status.get('total_predictions', 0)
            st.metric("AI Predictions", predictions)
        
        with col3:
            auto_trading = system_status.get('auto_trading_enabled', False)
            st.metric("Auto Trading", "ON" if auto_trading else "OFF")
    else:
        st.warning(f"System status error: {system_status['error']}")
    
    # Trading Signals
    st.subheader("🎯 Recent Trading Signals")
    
    signals = fetch_data_safe("trading-signals")
    
    if "error" not in signals and signals:
        st.write(f"**{len(signals)} signals generated**")
        
        # Show last 5 signals
        for signal in signals[-5:]:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    symbol = signal.get('symbol', 'Unknown')
                    action = signal.get('action', 'Unknown')
                    st.write(f"**{symbol}** - {action}")
                
                with col2:
                    confidence = signal.get('confidence', 0) * 100
                    st.write(f"Confidence: {confidence:.1f}%")
                
                with col3:
                    entry_price = signal.get('entry_price', 0)
                    st.write(f"Entry: {entry_price:.5f}")
                
                st.markdown("---")
    else:
        st.info("🤖 No trading signals yet")
        if "error" in signals:
            st.error(f"Signals error: {signals['error']}")
    
    # Debug Information
    with st.expander("🔍 Debug Information"):
        st.write("**API Endpoints Tested:**")
        st.write(f"- Base: {API_URL}")
        st.write(f"- Account: {API_URL}/account-info")
        st.write(f"- Market: {API_URL}/market-data")
        st.write(f"- Status: {API_URL}/system-status")
        st.write(f"- Signals: {API_URL}/trading-signals")
        
        st.write("**Current Time:**", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        if st.button("🧪 Test All Endpoints"):
            with st.spinner("Testing all endpoints..."):
                endpoints = ["", "account-info", "market-data", "system-status", "trading-signals"]
                for endpoint in endpoints:
                    result = fetch_data_safe(endpoint)
                    st.write(f"**{endpoint or 'root'}:** {'✅ OK' if 'error' not in result else '❌ ' + result['error']}")
    
    # Manual Refresh
    st.markdown("---")
    if st.button("🔄 Refresh All Data"):
        st.rerun()
    
    st.markdown("**Last update:** " + datetime.now().strftime('%H:%M:%S'))

if __name__ == "__main__":
    main()

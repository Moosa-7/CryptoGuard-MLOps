import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# API URL (Internal Docker Network)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="CryptoGuard AI", layout="wide")

st.title("🛡️ CryptoGuard AI Dashboard")
st.markdown("### Enterprise-Grade Financial Intelligence System")

# Sidebar Navigation
page = st.sidebar.selectbox("Choose Module", ["💰 Bitcoin Forecaster", "🚨 Fraud Detection", "👥 User Segmentation"])

# ==========================================
# PAGE 1: BITCOIN FORECASTER
# ==========================================
if page == "💰 Bitcoin Forecaster":
    st.header("Bitcoin Price Direction & Forecasting")
    
    # 1. Fetch Live Data for Context
    ticker = "BTC-USD"
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)
    
    # Fix MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    
    # 2. Display Live Metrics
    col1, col2, col3 = st.columns(3)
    current_price = last_row['Close']
    change = current_price - prev_row['Close']
    
    col1.metric("Current Price", f"${current_price:,.2f}", f"{change:,.2f}")
    col2.metric("Volume", f"{last_row['Volume']:,}")
    col3.metric("Volatility (7d)", f"{data['Close'].pct_change().rolling(7).std().iloc[-1]:.4f}")

    # 3. Chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'])])
    fig.update_layout(title="BTC-USD Price Action (3 Months)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 4. AI Prediction Section
    st.subheader("🤖 AI Analysis")
    if st.button("Generate Forecast"):
        # Construct Payload from live data logic
        # We need to engineer the same features: Lag_1, Returns, MA_7, MA_30, Volatility
        # Note: We do this calc live here to feed the API
        
        # Calculate features manually for the LAST row to send to API
        closes = data['Close']
        payload = {
            "close": float(closes.iloc[-1]),
            "volume": float(data['Volume'].iloc[-1]),
            "lag_1": float(closes.iloc[-2]),
            "daily_return": float(closes.pct_change().iloc[-1]),
            "ma_7": float(closes.rolling(7).mean().iloc[-1]),
            "ma_30": float(closes.rolling(30).mean().iloc[-1]),
            "volatility": float(closes.pct_change().rolling(7).std().iloc[-1])
        }
        
        try:
            response = requests.post(f"{API_URL}/predict/btc", json=payload)
            res_json = response.json()
            
            # Display Result
            c1, c2 = st.columns(2)
            
            signal_color = "green" if res_json['signal'] == "Buy" else "red" if res_json['signal'] == "Sell" else "gray"
            
            c1.markdown(f"#### AI Signal: :{signal_color}[{res_json['signal'].upper()}]")
            c1.info(f"Confidence: {res_json['confidence']}")
            
            c2.metric("Predicted Next Price", f"${res_json['predicted_next_price']:,.2f}")
            
        except Exception as e:
            st.error(f"API Error: {e}")
            st.warning("Make sure FastAPI is running! (uvicorn src.api.main:app --reload)")

# ==========================================
# PAGE 2: FRAUD DETECTION
# ==========================================
elif page == "🚨 Fraud Detection":
    st.header("Real-Time Transaction Audit")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Transaction Details")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
        
        # Simulation buttons
        if st.button("Simulate LEGIT Transaction"):
            # Random normal features
            feats = np.random.normal(0, 1, 28).tolist()
            st.session_state['fraud_feats'] = [0] + feats + [amount] # Time=0
            
        if st.button("Simulate FRAUD Transaction"):
            # Extreme outliers typical of fraud
            feats = np.random.normal(5, 10, 28).tolist() # High variance
            st.session_state['fraud_feats'] = [0] + feats + [amount]

    # Predict Logic
    if 'fraud_feats' in st.session_state:
        payload = {"features": st.session_state['fraud_feats']}
        
        try:
            response = requests.post(f"{API_URL}/predict/fraud", json=payload)
            res = response.json()
            
            with col2:
                st.subheader("Risk Assessment")
                
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = res['probability'] * 100,
                    title = {'text': "Fraud Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if res['is_fraud'] else "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "salmon"}],
                    }))
                st.plotly_chart(fig)
                
                if res['is_fraud']:
                    st.error(f"❌ ACTION: {res['action']} (Risk: {res['risk_level']})")
                else:
                    st.success(f"✅ ACTION: {res['action']} (Risk: {res['risk_level']})")
                    
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# ==========================================
# PAGE 3: SEGMENTATION
# ==========================================
elif page == "👥 User Segmentation":
    st.header("Customer Clustering (PCA Visualization)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.info("Adjust Principal Components to find user segments.")
        pc1 = st.slider("PC1 (Variance A)", -5.0, 15.0, 0.0)
        pc2 = st.slider("PC2 (Variance B)", -5.0, 15.0, 0.0)
        
        if st.button("Classify User"):
            try:
                res = requests.post(f"{API_URL}/predict/segment", json={"pc1": pc1, "pc2": pc2}).json()
                st.session_state['segment_res'] = res
            except:
                st.error("API Error")

    with col2:
        # Visualize the "Map"
        # We generate a static background scatter to show where the clusters usually are
        # (This mimics the training data shape)
        
        # Mock background data for visualization
        x_dummy = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10, 2, 100)])
        y_dummy = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10, 2, 100)])
        
        fig = px.scatter(x=x_dummy, y=y_dummy, opacity=0.3, labels={'x': 'PC1', 'y': 'PC2'}, title="User Cluster Map")
        
        # Add the User's point
        fig.add_trace(go.Scatter(x=[pc1], y=[pc2], mode='markers', marker=dict(color='red', size=15), name="Current User"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        if 'segment_res' in st.session_state:
            res = st.session_state['segment_res']
            st.success(f"User belongs to: **{res['segment_name']}**")
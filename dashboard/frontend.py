import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os  # <--- NEW IMPORT

# --- CONFIGURATION ---
# FIX: Read from Environment Variable (set in Dockerfile) or default to localhost
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="FinTech AI Platform", layout="wide")

# --- TITLE ---
st.title("🏦 FinTech Intelligence Command Center")

# --- SIDEBAR STATUS ---
st.sidebar.header("System Status")
try:
    health = requests.get(API_URL)
    if health.status_code == 200:
        st.sidebar.success(f"API: Online")
    else:
        st.sidebar.error("API: Error")
except:
    st.sidebar.error("API: Offline (Run uvicorn!)")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["💳 Fraud Monitor", "📈 Market Forecast", "👥 Customer Strategy"])

# ==========================================
# TAB 1: FRAUD MONITOR
# ==========================================
with tab1:
    st.header("Real-time Fraud Analysis")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Input Transaction Data")
        amt = st.number_input("Transaction Amount ($)", value=150.0)
        
        # Slider to simulate fraud intensity (0 = Normal, 10 = High Risk)
        anomaly_score = st.slider("Simulated Fraud Intensity", 0.0, 10.0, 0.0)
        
        # --- SMART FEATURE CONSTRUCTION ---
        # 1. Generate standard background noise (normal transactions)
        features = list(np.random.normal(0, 1, 28))
        
        # 2. Inject specific fraud patterns if slider > 0
        if anomaly_score > 0:
            # Fraud usually has negative V12, V14, V17 and positive V4, V11
            features[11] = -anomaly_score * 2  # V12
            features[13] = -anomaly_score * 3  # V14
            features[16] = -anomaly_score * 3  # V17
            features[3]  = anomaly_score * 1.5 # V4
            features[10] = anomaly_score * 1.5 # V11
            
        # 3. Combine features + Amount for the API
        final_features = features + [amt]

        if st.button("Scan Transaction", key="btn_fraud"):
            try:
                # Send data to API
                res = requests.post(f"{API_URL}/predict/fraud", json={"features": final_features})
                
                if res.status_code == 200:
                    data = res.json()
                    
                    with col2:
                        # Display Risk Score
                        st.metric("Risk Probability", f"{data['risk_score']*100:.2f}%")
                        
                        if data['is_fraud']:
                            st.error("🚨 FRAUD DETECTED")
                        else:
                            st.success("✅ Transaction Verified")
                            
                        # Visual Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number", 
                            value = data['risk_score']*100,
                            title = {'text': "AI Risk Score"},
                            gauge = {
                                'axis': {'range': [0, 100]}, 
                                'bar': {'color': "red" if data['is_fraud'] else "green"},
                                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 80}
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Debug info (optional)
                        with st.expander("View Raw Feature Vector"):
                            st.write(final_features)
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# ==========================================
# TAB 2: MARKET FORECAST
# ==========================================
with tab2:
    st.header("Bitcoin Price Intelligence")
    st.write("Using XGBoost model with Lag Features (Yesterday vs Last Week).")
    
    if st.button("Analyze Market Trends", key="btn_crypto"):
        with st.spinner("Fetching live data & running AI models..."):
            try:
                res = requests.post(f"{API_URL}/predict/btc")
                if res.status_code == 200:
                    data = res.json()
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Current Price", f"${data['current_price']:,.2f}")
                    c2.metric("Predicted (Next Day)", f"${data['predicted_next_day']:,.2f}", 
                              delta=f"{data['predicted_next_day'] - data['current_price']:.2f}")
                    
                    if data['trend'] == "UP":
                        st.success("📈 AI Recommendation: BUY/HOLD")
                    else:
                        st.warning("📉 AI Recommendation: SELL")
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ==========================================
# TAB 3: CUSTOMER STRATEGY
# ==========================================
with tab3:
    st.header("User Classification System")
    st.write("Classifies users into **Poor**, **Standard**, or **VIP** tiers based on financial activity.")
    
    c1, c2 = st.columns(2)
    with c1:
        # High range to allow testing the VIP threshold ($50k+)
        vol = st.slider("Monthly Volume ($)", 0, 100000, 500)
        freq = st.slider("Transaction Count", 0, 100, 5)
        
        if st.button("Classify User", key="btn_segment"):
            try:
                res = requests.post(f"{API_URL}/segment", json={"volume": vol, "frequency": freq})
                if res.status_code == 200:
                    data = res.json()
                    
                    st.divider()
                    st.subheader(f"Tier: {data['segment_name']}")
                    st.info(data['description'])
                    
                    # Visual Feedback based on Tier
                    if data['segment_name'] == "VIP":
                        st.balloons()
                        st.success("🌟 STATUS: ELITE / VIP CLIENT")
                    elif data['segment_name'] == "Standard":
                        st.warning("🔵 STATUS: STANDARD CLIENT")
                    else:
                        st.error("🔻 STATUS: BASIC / LOW BALANCE")
                        
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")
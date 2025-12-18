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
page = st.sidebar.selectbox("Choose Module", [
    "💰 Bitcoin Forecaster", 
    "📊 Market Correlation",
    "👥 Customer Segmentation" # Renamed for clarity
])

# ==========================================
# PAGE 1: BITCOIN FORECASTER
# ==========================================
if page == "💰 Bitcoin Forecaster":
    st.header("Bitcoin Price Direction & Forecasting")
    
    # 1. Fetch Live Data for Context
    ticker = "BTC-USD"
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    
    col1, col2, col3 = st.columns(3)
    current_price = last_row['Close']
    change = current_price - prev_row['Close']
    
    col1.metric("Current Price", f"${current_price:,.2f}", f"{change:,.2f}")
    col2.metric("Volume", f"{last_row['Volume']:,}")
    col3.metric("Volatility (7d)", f"{data['Close'].pct_change().rolling(7).std().iloc[-1]:.4f}")

    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'])])
    fig.update_layout(title="BTC-USD Price Action (3 Months)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤖 AI Analysis")
    if st.button("Generate Forecast"):
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
            if response.status_code == 200:
                res_json = response.json()
                c1, c2 = st.columns(2)
                signal_color = "green" if res_json['signal'] == "Buy" else "red" if res_json['signal'] == "Sell" else "gray"
                c1.markdown(f"#### AI Signal: :{signal_color}[{res_json['signal'].upper()}]")
                c1.info(f"Confidence: {res_json['confidence']}")
                c2.metric("Predicted Next Price", f"${res_json['predicted_next_price']:,.2f}")
            else:
                st.error("⚠️ Model Server Error")
        except Exception as e:
            st.error(f"API Error: {e}")

# ==========================================
# PAGE 2: MARKET CORRELATION
# ==========================================
elif page == "📊 Market Correlation":
    st.header("Global Asset Risk Radar")
    st.markdown("""
    This tool analyzes the hidden relationships between Bitcoin and traditional markets.
    * **Positive Correlation (Blue):** Moving Together.
    * **Negative Correlation (Red):** Moving Apart.
    """)
    
    tickers = ['BTC-USD', 'ETH-USD', '^GSPC', 'GC=F', 'DX-Y.NYB']
    labels = ['Bitcoin', 'Ethereum', 'S&P 500', 'Gold', 'US Dollar']
    
    if st.button("Analyze & Explain"):
        with st.spinner("Analyzing global markets..."):
            raw_data = yf.download(tickers, period="6mo", progress=False)['Close']
            
            if len(raw_data.columns) == len(labels):
               raw_data.columns = labels
            
            returns = raw_data.pct_change().dropna()
            corr_matrix = returns.corr()
            
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale='RdBu_r', 
                zmin=-1, zmax=1,
                title="6-Month Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🧠 AI Market Diagnosis")
            try:
                btc_sp500 = corr_matrix.loc['Bitcoin', 'S&P 500']
                btc_gold = corr_matrix.loc['Bitcoin', 'Gold']
                
                if btc_sp500 > 0.5:
                    st.error(f"🚨 **Current State: RISK-ON (Correlation: {btc_sp500:.2f})**")
                    st.write("Bitcoin is following the stock market closely.")
                    with st.expander("📚 What does 'Risk-On' mean?"):
                        st.markdown("""
                        **Risk-On** means investors are feeling confident. They are buying "risky" assets like Tech Stocks (Apple, Nvidia) and Bitcoin.
                        * **Implication:** If the S&P 500 crashes, Bitcoin will likely crash too.
                        * **Strategy:** Watch the stock market news (Interest rates, Earnings) to predict Bitcoin.
                        """)
                
                elif btc_gold > 0.4:
                    st.success(f"🛡️ **Current State: DIGITAL GOLD (Correlation: {btc_gold:.2f})**")
                    st.write("Bitcoin is acting as a hedge against inflation/fear.")
                    with st.expander("📚 What does 'Digital Gold' mean?"):
                        st.markdown("""
                        Investors are scared of the economy (Inflation, War, Bank Failures). They are selling cash and buying things that hold value, like Gold and Bitcoin.
                        * **Implication:** Bitcoin might rise even if the stock market falls.
                        """)

                else:
                    st.info(f"🛸 **Current State: DECOUPLED (Correlation: {btc_sp500:.2f})**")
                    st.write("Bitcoin is ignoring traditional markets. It is moving based on Crypto-specific news.")
                    
                    st.markdown("#### 🔍 Key Drivers When Decoupled:")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        with st.expander("⛏️ The Halving"):
                            st.caption("Supply Shock")
                            st.write("Every 4 years, the amount of new Bitcoin created is cut in half. Less supply + same demand = Higher Price.")
                    
                    with c2:
                        with st.expander("🏦 ETFs"):
                            st.caption("Institutional Money")
                            st.write("Exchange Traded Funds (ETFs) allow big stock market investors (like BlackRock) to buy Bitcoin easily. This brings massive new money.")
                    
                    with c3:
                        with st.expander("⚖️ Regulation"):
                            st.caption("Government Rules")
                            st.write("Laws about whether crypto is legal or illegal. Positive laws (clarity) boost price; bans crash price.")

            except Exception as e:
                st.warning("Could not generate text insights due to data alignment issues.")

# ==========================================
# PAGE 3: USER SEGMENTATION (REFACTORED)
# ==========================================
elif page == "👥 Customer Segmentation":
    st.header("Customer Tier Classification")
    st.markdown("Identify user personas based on their **Spending** and **Activity** habits.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🕵️ User Simulator")
        st.info("Adjust the sliders to simulate a user's behavior.")
        
        # Renamed sliders to Business Terms
        spending = st.slider("💰 Spending Volume (PC1)", -5.0, 15.0, 0.0, help="Higher = Higher Transaction Amounts")
        frequency = st.slider("🔄 Activity Frequency (PC2)", -5.0, 15.0, 0.0, help="Higher = More Logins/Trades per day")
        
        if st.button("Classify User Tier"):
            try:
                # We still send 'pc1' and 'pc2' to the API, but the user sees "Spending" and "Frequency"
                response = requests.post(f"{API_URL}/predict/segment", json={"pc1": spending, "pc2": frequency})
                if response.status_code == 200:
                    st.session_state['segment_res'] = response.json()
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Connection Error: {e}")

        # EDUCATIONAL PANEL
        with st.expander("📚 How to interpret this?"):
            st.markdown("""
            **1. Spending Volume (X-Axis):**
            * **Left:** Small retail traders (micro-transactions).
            * **Right:** Whales / Institutional Investors (large volume).

            **2. Activity Frequency (Y-Axis):**
            * **Bottom:** "HODLers" (Buy once, come back in a year).
            * **Top:** Day Traders (Trading every hour).
            """)

    with col2:
        # Mock Visualization with Business Labels
        # Standard Users (Cluster 0) = Lower Spend, Variable Frequency
        x_std = np.random.normal(0, 2, 100)
        y_std = np.random.normal(0, 2, 100)
        
        # VIP Users (Cluster 1) = High Spend, High Frequency
        x_vip = np.random.normal(8, 2, 50)
        y_vip = np.random.normal(5, 2, 50)
        
        fig = go.Figure()
        
        # Add Cluster 0 (Standard)
        fig.add_trace(go.Scatter(
            x=x_std, y=y_std, 
            mode='markers', name='Standard Tier',
            marker=dict(color='lightblue', opacity=0.6)
        ))
        
        # Add Cluster 1 (VIP)
        fig.add_trace(go.Scatter(
            x=x_vip, y=y_vip, 
            mode='markers', name='VIP / High-Net-Worth',
            marker=dict(color='gold', opacity=0.8)
        ))

        # Add the "Current User" Dot
        fig.add_trace(go.Scatter(
            x=[spending], y=[frequency], 
            mode='markers', name='THIS USER',
            marker=dict(color='red', size=20, symbol='star')
        ))

        fig.update_layout(
            title="Customer Persona Map",
            xaxis_title="Spending Volume (Low → High)",
            yaxis_title="Activity Frequency (Low → High)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Result Display with Business Logic
        if 'segment_res' in st.session_state:
            res = st.session_state['segment_res']
            cluster_id = res.get('cluster', -1) # Default to -1 if missing
            
            # Map Cluster IDs to Names
            # Note: This assumes your K-Means assigned 0 to Standard and 1 to VIP.
            # If your model training flipped them, swap these names.
            cluster_name = "VIP / High-Net-Worth" if cluster_id == 1 else "Standard Tier"
            
            st.divider()
            # Result Display with Business Logic
        if 'segment_res' in st.session_state:
            res = st.session_state['segment_res']
            cluster_id = res.get('cluster', -1) 
            
            # --- THE FIX: SWAP THE MAPPING ---
            # If your model predicts 0 for high values, then 0 is VIP.
            # We treat 0 as VIP and 1 as Standard (or vice versa based on observation).
            
            if cluster_id == 0:  # <--- CHANGED FROM 1 TO 0 (Try this flip)
                cluster_name = "VIP / High-Net-Worth"
                recommendation = "**Recommendation:** Assign dedicated account manager. Offer zero-fee OTC desk."
                box_color = "success" # Green
            else:
                cluster_name = "Standard Tier"
                recommendation = "**Recommendation:** Send 'Crypto 101' educational emails. Encourage recurring buy setup."
                box_color = "info" # Blue
            
            st.divider()
            
            # Display Dynamic Result
            if box_color == "success":
                st.success(f"### 🏆 Classification: {cluster_name}")
            else:
                st.info(f"### 👤 Classification: {cluster_name}")
                
            st.markdown(recommendation)
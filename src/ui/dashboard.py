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
    "👥 User Segmentation"
])

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
# PAGE 2: MARKET CORRELATION (EDUCATIONAL)
# ==========================================
elif page == "📊 Market Correlation":
    st.header("Global Asset Risk Radar")
    st.markdown("""
    This tool analyzes the hidden relationships between Bitcoin and traditional markets.
    * **Positive Correlation (Blue):** Moving Together.
    * **Negative Correlation (Red):** Moving Apart.
    """)
    
    # 1. Define Tickers
    tickers = ['BTC-USD', 'ETH-USD', '^GSPC', 'GC=F', 'DX-Y.NYB']
    labels = ['Bitcoin', 'Ethereum', 'S&P 500', 'Gold', 'US Dollar']
    
    if st.button("Analyze & Explain"):
        with st.spinner("Analyzing global markets..."):
            # Download Data
            raw_data = yf.download(tickers, period="6mo", progress=False)['Close']
            
            # Clean Columns
            if len(raw_data.columns) == len(labels):
               raw_data.columns = labels
            
            # Calculate Correlation
            returns = raw_data.pct_change().dropna()
            corr_matrix = returns.corr()
            
            # Draw Heatmap
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale='RdBu_r', 
                zmin=-1, zmax=1,
                title="6-Month Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- EDUCATIONAL INSIGHTS ENGINE ---
            st.markdown("### 🧠 AI Market Diagnosis")
            
            try:
                btc_sp500 = corr_matrix.loc['Bitcoin', 'S&P 500']
                btc_gold = corr_matrix.loc['Bitcoin', 'Gold']
                
                # SCENARIO 1: RISK-ON (Follows Stocks)
                if btc_sp500 > 0.5:
                    st.error(f"🚨 **Current State: RISK-ON (Correlation: {btc_sp500:.2f})**")
                    st.write("Bitcoin is following the stock market closely.")
                    with st.expander("📚 What does 'Risk-On' mean?"):
                        st.markdown("""
                        **Risk-On** means investors are feeling confident. They are buying "risky" assets like Tech Stocks (Apple, Nvidia) and Bitcoin.
                        * **Implication:** If the S&P 500 crashes, Bitcoin will likely crash too.
                        * **Strategy:** Watch the stock market news (Interest rates, Earnings) to predict Bitcoin.
                        """)
                
                # SCENARIO 2: SAFE HAVEN (Follows Gold)
                elif btc_gold > 0.4:
                    st.success(f"🛡️ **Current State: DIGITAL GOLD (Correlation: {btc_gold:.2f})**")
                    st.write("Bitcoin is acting as a hedge against inflation/fear.")
                    with st.expander("📚 What does 'Digital Gold' mean?"):
                        st.markdown("""
                        Investors are scared of the economy (Inflation, War, Bank Failures). They are selling cash and buying things that hold value, like Gold and Bitcoin.
                        * **Implication:** Bitcoin might rise even if the stock market falls.
                        """)

                # SCENARIO 3: DECOUPLED (Independent)
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
                response = requests.post(f"{API_URL}/predict/segment", json={"pc1": pc1, "pc2": pc2})
                if response.status_code == 200:
                    st.session_state['segment_res'] = response.json()
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Connection Error: {e}")

    with col2:
        x_dummy = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10, 2, 100)])
        y_dummy = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10, 2, 100)])
        
        fig = px.scatter(x=x_dummy, y=y_dummy, opacity=0.3, labels={'x': 'PC1', 'y': 'PC2'}, title="User Cluster Map")
        fig.add_trace(go.Scatter(x=[pc1], y=[pc2], mode='markers', marker=dict(color='red', size=15), name="Current User"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        if 'segment_res' in st.session_state:
            res = st.session_state['segment_res']
            st.success(f"User belongs to: **{res['segment_name']}**")
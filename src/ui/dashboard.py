import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# API URL - Use environment variable or default to localhost
# In containerized deployments, use localhost since both services run in same container
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="CryptoGuard AI", layout="wide")

st.title("🛡️ CryptoGuard AI Dashboard")
st.markdown("### Enterprise-Grade Financial Intelligence System")

# Sidebar Navigation
page = st.sidebar.selectbox("Choose Module", [
    "💰 Bitcoin Forecaster", 
    "📊 Market Correlation",
    "👥 Customer Segmentation",
    "📈 Data Explorer",
    "⚠️ Data Drift Monitor",
    "📊 Model Performance"
])

# ==========================================
# PAGE 1: BITCOIN FORECASTER
# ==========================================
if page == "💰 Bitcoin Forecaster":
    st.header("Bitcoin Price Direction & Forecasting")
    
    # 1. Fetch Live Data for Context
    ticker = "BTC-USD"
    data = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    
    col1, col2, col3 = st.columns(3)
    current_price = last_row['Close']
    change = current_price - prev_row['Close']
    
    col1.metric("Current Price", f"${current_price:,.2f}", f"{change:,.2f}")
    col2.metric("Volume", f"{last_row['Volume']:,}")
    col3.metric("Volatility (7d)", f"{data['Close'].pct_change(fill_method=None).rolling(7).std().iloc[-1]:.4f}")

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'])])
    
    # Add prediction interval if available in session state
    if 'btc_prediction' in st.session_state:
        pred_data = st.session_state['btc_prediction']
        if 'prediction_interval' in pred_data:
            interval = pred_data['prediction_interval']
            # Add prediction point and interval
            last_date = data.index[-1]
            next_date = pd.date_range(start=last_date, periods=2, freq='D')[1]
            
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[pred_data['predicted_next_price']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Predicted Price'
            ))
            
            # Add confidence interval as error bars
            fig.add_trace(go.Scatter(
                x=[next_date, next_date],
                y=[interval['lower'], interval['upper']],
                mode='lines',
                line=dict(width=4, color='rgba(255,0,0,0.3)'),
                name='95% Confidence Interval',
                showlegend=True
            ))
    
    fig.update_layout(title="BTC-USD Price Action (3 Months)", height=400)
    st.plotly_chart(fig, width='stretch')

    st.subheader("🤖 AI Analysis")
    forecast_clicked = st.button("Generate Forecast")
    
    if forecast_clicked:
        closes = data['Close']
        payload = {
            "close": float(closes.iloc[-1]),
            "volume": float(data['Volume'].iloc[-1]),
            "lag_1": float(closes.iloc[-2]),
            "daily_return": float(closes.pct_change(fill_method=None).iloc[-1]),
            "ma_7": float(closes.rolling(7).mean().iloc[-1]),
            "ma_30": float(closes.rolling(30).mean().iloc[-1]),
            "volatility": float(closes.pct_change(fill_method=None).rolling(7).std().iloc[-1])
        }
        
        try:
            response = requests.post(f"{API_URL}/predict/btc", json=payload, timeout=5)
            if response.status_code == 200:
                res_json = response.json()
                c1, c2, c3 = st.columns(3)
                signal_color = "green" if res_json['signal'] == "Buy" else "red" if res_json['signal'] == "Sell" else "gray"
                c1.markdown(f"#### AI Signal: :{signal_color}[{res_json['signal'].upper()}]")
                c1.info(f"Confidence: {res_json['confidence']}")
                c2.metric("Predicted Next Price", f"${res_json['predicted_next_price']:,.2f}")
                
                # Show prediction intervals if available
                if 'prediction_interval' in res_json:
                    interval = res_json['prediction_interval']
                    interval_width = interval['upper'] - interval['lower']
                    c3.metric("Prediction Interval", f"${interval['lower']:,.2f} - ${interval['upper']:,.2f}")
                    c3.caption(f"Width: ${interval_width:,.2f}")
                
                # Store payload for explanation
                st.session_state['btc_payload'] = payload
                st.session_state['btc_prediction'] = res_json
            else:
                st.error("⚠️ Model Server Error")
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ **API Backend Not Connected** - Forecast generation requires the FastAPI service to be running.")
        except requests.exceptions.Timeout:
            st.warning("⏱️ API request timed out. Please try again.")
        except Exception as e:
            st.warning(f"Service temporarily unavailable: {str(e)[:80]}...")
    
    # Show explanation button if prediction exists
    if 'btc_payload' in st.session_state and st.button("🔍 Explain Forecast"):
        payload = st.session_state['btc_payload']
        with st.spinner("Generating explanations..."):
            try:
                # Get SHAP explanation
                explain_response = requests.post(f"{API_URL}/explain/btc", json=payload, timeout=5)
                if explain_response.status_code == 200:
                    exp_data = explain_response.json()
                    
                    st.subheader("📊 Model Interpretability")
                    tab1, tab2 = st.tabs(["SHAP Analysis", "Feature Importance"])
                    
                    with tab1:
                        if exp_data.get('shap_explanation', {}).get('feature_importance'):
                            df_importance = pd.DataFrame(exp_data['shap_explanation']['feature_importance'])
                            fig = px.bar(df_importance.head(10), x='importance', y='feature', 
                                        orientation='h', title="Top 10 Features Driving Prediction (SHAP)")
                            st.plotly_chart(fig, width='stretch')
                    
                    with tab2:
                        if exp_data.get('feature_importance'):
                            df_feat = pd.DataFrame(exp_data['feature_importance'])
                            fig = px.bar(df_feat.head(10), x='importance', y='feature',
                                        orientation='h', title="Model Feature Importance")
                            st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Could not generate explanations. Model may not support explanations.")
            except Exception as e:
                st.error(f"Explanation error: {e}")
    
    # Add feature importance visualization (always available - prominently displayed)
    st.subheader("📊 Model Feature Importance")
    try:
        feat_response = requests.get(f"{API_URL}/features/importance/btc_price", timeout=2)
        if feat_response.status_code == 200:
            feat_data = feat_response.json()
            df_feat = pd.DataFrame({
                'feature': feat_data['features'],
                'importance': feat_data['importance']
            }).sort_values('importance', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(df_feat.head(10), x='importance', y='feature',
                            orientation='h', title="Top 10 Most Important Features for BTC Price Prediction",
                            labels={'importance': 'Importance Score', 'feature': 'Feature'})
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("### Insights")
                top_feature = df_feat.iloc[0]
                st.info(f"**{top_feature['feature']}** is the most important feature (score: {top_feature['importance']:.4f})")
                st.caption("Higher importance scores indicate features that contribute more to price predictions.")
        else:
            st.info("Feature importance not available. The model may need to be retrained.")
    except requests.exceptions.ConnectionError:
        st.info("💡 **Feature importance unavailable** - API backend not connected. This feature requires the FastAPI service to be running.")
    except requests.exceptions.Timeout:
        st.info("⏱️ Feature importance request timed out. The API may be starting up.")
    except Exception as e:
        # Hide technical error details from users
        st.info("Feature importance not available at this time.")

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
            raw_data = yf.download(tickers, period="6mo", progress=False, auto_adjust=True)['Close']
            
            if len(raw_data.columns) == len(labels):
               raw_data.columns = labels
            
            returns = raw_data.pct_change(fill_method=None).dropna()
            corr_matrix = returns.corr()
            
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale='RdBu_r', 
                zmin=-1, zmax=1,
                title="6-Month Correlation Matrix"
            )
            st.plotly_chart(fig, width='stretch')
            
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
# PAGE 4: CUSTOMER SEGMENTATION
# ==========================================
elif page == "👥 Customer Segmentation":
    st.header("Customer Tier Classification")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🕵️ User Simulator")
        spending = st.slider("💰 Spending Volume (PC1)", -5.0, 15.0, 0.0)
        frequency = st.slider("🔄 Activity Frequency (PC2)", -5.0, 15.0, 0.0)
        
        # Initialize result variables to avoid NameError later
        cluster_name = "Unknown"
        box_color = "gray"
        
        # AUTOMATIC PREDICTION (Inside Try Block)
        try:
            response = requests.post(f"{API_URL}/predict/segment", json={"pc1": spending, "pc2": frequency}, timeout=3)
            
            # ✅ CHECK INSIDE THE BLOCK
            if response.status_code == 200:
                res = response.json()
                cluster_name = res.get('segment_name', 'Unknown')
                
                # Set box color based on segment name
                if "VIP" in cluster_name or "High-Net-Worth" in cluster_name:
                    box_color = "success"
                else:
                    box_color = "info"
            else:
                st.warning("⚠️ API is starting up...")
                cluster_name = "Unknown (API unavailable)"
                
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ **API Backend Not Connected** - Customer segmentation requires the FastAPI service.")
            cluster_name = "Unknown (API unavailable)"
        except requests.exceptions.Timeout:
            st.warning("⚠️ API request timed out. Please try again.")
            cluster_name = "Unknown (API timeout)"
        except Exception as e:
            st.warning(f"⚠️ Service temporarily unavailable. Error: {str(e)[:50]}...")
            cluster_name = "Unknown (Service error)"

        # EDUCATIONAL PANEL
        with st.expander("📚 Interpretation Guide"):
            st.markdown("""
            * **High Spending + High Freq** = VIP / Institution
            * **Low Spending + Low Freq** = Retail / HODLer
            """)

    with col2:
        # Visualization
        x_std = np.random.normal(2, 1.5, 100)
        y_std = np.random.normal(2, 1.5, 100)
        x_vip = np.random.normal(12, 1.5, 50)
        y_vip = np.random.normal(12, 1.5, 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_std, y=y_std, mode='markers', name='Standard Tier', marker=dict(color='lightblue')))
        fig.add_trace(go.Scatter(x=x_vip, y=y_vip, mode='markers', name='VIP Tier', marker=dict(color='gold')))
        fig.add_trace(go.Scatter(x=[spending], y=[frequency], mode='markers', name='THIS USER', marker=dict(color='red', size=20, symbol='star')))
        
        fig.update_layout(title="Persona Map", xaxis_title="Spending", yaxis_title="Frequency", height=500)
        st.plotly_chart(fig, width='stretch')
        
        # Result Display (Safe because variables are initialized)
        st.divider()
        if box_color == "success":
            st.success(f"### 🏆 Classification: {cluster_name}")
            st.markdown("**Recommendation:** Assign Dedicated Account Manager.")
        elif box_color == "info":
            st.info(f"### 👤 Classification: {cluster_name}")
            st.markdown("**Recommendation:** Send Standard Promo Emails.")

# ==========================================
# PAGE 5: DATA EXPLORER (EDA)
# ==========================================
elif page == "📈 Data Explorer":
    st.header("Exploratory Data Analysis")
    
    analysis_type = st.selectbox("Choose Dataset", ["Fraud Detection", "Bitcoin Market"])
    
    if analysis_type == "Fraud Detection":
        try:
            # Use relative imports or direct implementation
            import sys
            import os
            # Add project root to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            try:
                from src.utils.eda_helpers import load_sample_data, calculate_summary_stats, detect_outliers
            except ImportError:
                # Fallback: implement directly with robust path finding
                def load_sample_data(dataset_name, sample_size=5000):
                    if dataset_name.lower() == 'fraud':
                        # Try multiple possible paths
                        possible_paths = [
                            os.path.join(project_root, "data", "creditcard.csv"),
                            os.path.join(os.getcwd(), "data", "creditcard.csv"),
                            "data/creditcard.csv",
                        ]
                        for filepath in possible_paths:
                            if os.path.exists(filepath):
                                df = pd.read_csv(filepath)
                                if len(df) > sample_size:
                                    df = df.sample(n=sample_size, random_state=42)
                                return df
                        # File not found - generate mock data
                        st.info("💡 **Using generated sample data** - Original file not available (file >25MB, not in Git). This is normal in deployed environments.")
                        return _generate_mock_fraud_data_dashboard(sample_size)
                    return None
                
                def _generate_mock_fraud_data_dashboard(n_samples=5000):
                    """Generate mock fraud data for dashboard when file is not available"""
                    np.random.seed(42)
                    n_features = 29
                    data = np.random.randn(n_samples, n_features)
                    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                    df = pd.DataFrame(data, columns=cols[:-1])  # All except Class
                    df['Amount'] = np.abs(np.random.lognormal(mean=3, sigma=1.5, size=n_samples))
                    fraud_rate = 0.0017
                    n_fraud = int(n_samples * fraud_rate)
                    df['Class'] = 0
                    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
                    df.loc[fraud_indices, 'Class'] = 1
                    if n_fraud > 0:
                        df.loc[fraud_indices, 'V14'] = df.loc[fraud_indices, 'V14'] - 2
                        df.loc[fraud_indices, 'V12'] = df.loc[fraud_indices, 'V12'] - 2
                        df.loc[fraud_indices, 'V10'] = df.loc[fraud_indices, 'V10'] - 1.5
                    df['Time'] = np.sort(np.random.uniform(0, 48*3600, n_samples))
                    return df
            
            def _generate_mock_fraud_data_dashboard(n_samples=5000):
                """Generate mock fraud data for dashboard when file is not available (module-level fallback)"""
                np.random.seed(42)
                n_features = 29
                data = np.random.randn(n_samples, n_features)
                cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                df = pd.DataFrame(data, columns=cols[:-1])  # All except Class
                df['Amount'] = np.abs(np.random.lognormal(mean=3, sigma=1.5, size=n_samples))
                fraud_rate = 0.0017
                n_fraud = int(n_samples * fraud_rate)
                df['Class'] = 0
                fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
                df.loc[fraud_indices, 'Class'] = 1
                if n_fraud > 0:
                    df.loc[fraud_indices, 'V14'] = df.loc[fraud_indices, 'V14'] - 2
                    df.loc[fraud_indices, 'V12'] = df.loc[fraud_indices, 'V12'] - 2
                    df.loc[fraud_indices, 'V10'] = df.loc[fraud_indices, 'V10'] - 1.5
                df['Time'] = np.sort(np.random.uniform(0, 48*3600, n_samples))
                return df
            
            try:
                from src.utils.eda_helpers import calculate_summary_stats, detect_outliers
            except ImportError:
                def calculate_summary_stats(df):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    summary = {
                        'total_rows': len(df),
                        'total_columns': len(df.columns),
                        'numeric_columns': len(numeric_cols),
                        'statistics': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
                    }
                    if 'Class' in df.columns:
                        fraud_count = df['Class'].sum()
                        summary['fraud_count'] = int(fraud_count)
                        summary['fraud_rate'] = float(fraud_count / len(df))
                    return summary
                
                def detect_outliers(df, column, method='iqr'):
                    if column not in df.columns:
                        return pd.DataFrame()
                    values = df[column].dropna()
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    return pd.DataFrame({
                        'is_outlier': df[column].apply(lambda x: x < lower_bound or x > upper_bound),
                        'value': df[column],
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })
            
            df = load_sample_data("fraud", sample_size=5000)
            
            if df is not None and not df.empty and 'Class' in df.columns:
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Correlations", "Outliers"])
                
                with tab1:
                    st.subheader("Dataset Overview")
                    stats = calculate_summary_stats(df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Records", stats.get('total_rows', 0))
                    col2.metric("Total Columns", stats.get('total_columns', 0))
                    col3.metric("Numeric Columns", stats.get('numeric_columns', 0))
                    
                    if 'fraud_rate' in stats:
                        col4.metric("Fraud Rate", f"{stats['fraud_rate']*100:.2f}%")
                    
                    st.subheader("Summary Statistics")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe())
                
                with tab2:
                    st.subheader("Feature Distributions")
                    selected_col = st.selectbox("Select Feature", list(df.select_dtypes(include=[np.number]).columns[:15]))
                    
                    if 'Class' in df.columns:
                        fig = px.histogram(df, x=selected_col, color='Class', 
                                          title=f"Distribution of {selected_col}",
                                          nbins=50)
                    else:
                        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}", nbins=50)
                    st.plotly_chart(fig, width='stretch')
                
                with tab3:
                    st.subheader("Correlation Analysis")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]
                    if len(numeric_cols) > 1:
                        # Make sure 'Class' is included if it exists
                        if 'Class' in df.columns and 'Class' not in numeric_cols:
                            numeric_cols = list(numeric_cols) + ['Class']
                        corr = df[numeric_cols].corr()
                        
                        # For fraud data, focus on correlations with Class (fraud target)
                        if 'Class' in df.columns and 'Class' in corr.columns:
                            # Get correlations with Class
                            class_corr = corr['Class'].drop('Class').sort_values(key=abs, ascending=False)
                            
                            st.markdown("#### Top Features Correlated with Fraud (Class)")
                            top_corr_df = pd.DataFrame({
                                'Feature': class_corr.head(15).index,
                                'Correlation': class_corr.head(15).values
                            })
                            
                            fig = px.bar(top_corr_df, x='Correlation', y='Feature', 
                                        orientation='h', color='Correlation',
                                        color_continuous_scale='RdBu_r',
                                        title="Top 15 Features Correlated with Fraud",
                                        labels={'Correlation': 'Correlation Coefficient with Fraud'})
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, width='stretch')
                            
                            # Interpretation
                            st.markdown("**🔍 Research Insights:**")
                            positive_corr = class_corr[class_corr > 0.1]
                            negative_corr = class_corr[class_corr < -0.1]
                            if len(negative_corr) > 0:
                                st.info(f"Features with **strong negative correlation** (e.g., {', '.join(negative_corr.head(3).index.tolist())}) are key indicators of fraud. Lower values of these features increase fraud likelihood.")
                            if len(positive_corr) > 0:
                                st.warning(f"Features with **positive correlation** (e.g., {', '.join(positive_corr.head(3).index.tolist())}) increase with fraud cases, though they may be less predictive.")
                            
                            # Full correlation matrix (excluding self-correlations)
                            st.markdown("#### Full Feature Correlation Matrix")
                            # Create mask for upper triangle to remove duplicates
                            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                            corr_masked = corr.mask(mask)
                            
                            fig2 = px.imshow(corr_masked, text_auto='.2f', aspect="auto", 
                                           title="Feature Correlation Matrix (Upper Triangle Only)",
                                           color_continuous_scale='RdBu_r',
                                           zmin=-1, zmax=1)
                            st.plotly_chart(fig2, width='stretch')
                        else:
                            # For non-fraud data, show full correlation but mask diagonal
                            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                            corr_masked = corr.mask(mask)
                            
                            fig = px.imshow(corr_masked, text_auto='.2f', aspect="auto", 
                                           title="Feature Correlation Matrix",
                                           color_continuous_scale='RdBu_r',
                                           zmin=-1, zmax=1)
                            st.plotly_chart(fig, width='stretch')
                
                with tab4:
                    st.subheader("Outlier Detection")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]
                    selected_col = st.selectbox("Select Feature for Outlier Analysis", 
                                               list(numeric_cols), key="outlier_col")
                    
                    if 'Class' in df.columns:
                        fig = px.box(df, y=selected_col, color='Class', 
                                    title=f"Outlier Detection: {selected_col}")
                    else:
                        fig = px.box(df, y=selected_col, title=f"Outlier Detection: {selected_col}")
                    st.plotly_chart(fig, width='stretch')
                    
                    outliers_df = detect_outliers(df, selected_col)
                    if not outliers_df.empty:
                        outlier_count = outliers_df['is_outlier'].sum()
                        st.info(f"Found {outlier_count} outliers in {selected_col}")
            else:
                # This should not happen if load_sample_data returns mock data, but just in case
                if df is None:
                    st.info("💡 **Using generated sample data** - Original file not available (file >25MB, not in Git). This is normal in deployed environments.")
                    df = _generate_mock_fraud_data_dashboard(5000)
                elif df.empty:
                    st.error("File is empty.")
                elif 'Class' not in df.columns:
                    st.error(f"'Class' column not found. Columns: {list(df.columns)[:10]}...")
                else:
                    st.error("Could not load fraud detection data.")
        except Exception as e:
            import traceback
            st.error(f"Error loading data: {e}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    elif analysis_type == "Bitcoin Market":
        try:
            # Use relative imports or direct implementation
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            try:
                from src.utils.eda_helpers import load_sample_data, calculate_summary_stats
            except ImportError:
                # Fallback implementation
                def load_sample_data(dataset_name, sample_size=5000):
                    if dataset_name.lower() in ['btc', 'bitcoin']:
                        filepath = "data/raw/btc_usd.parquet"
                        if os.path.exists(filepath):
                            df = pd.read_parquet(filepath)
                            if len(df) > sample_size:
                                df = df.sample(n=sample_size, random_state=42)
                            return df
                    return None
                
                def calculate_summary_stats(df):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    return {
                        'total_rows': len(df),
                        'total_columns': len(df.columns),
                        'numeric_columns': len(numeric_cols),
                        'statistics': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
                    }
            
            df = load_sample_data("btc", sample_size=5000)
            
            if df is not None and not df.empty:
                tab1, tab2, tab3 = st.tabs(["Overview", "Time Series", "Correlations"])
                
                with tab1:
                    st.subheader("Dataset Overview")
                    stats = calculate_summary_stats(df)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Records", stats.get('total_rows', 0))
                    col2.metric("Total Columns", stats.get('total_columns', 0))
                    col3.metric("Numeric Columns", stats.get('numeric_columns', 0))
                    
                    st.subheader("Summary Statistics")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe())
                
                with tab2:
                    st.subheader("Price Time Series")
                    if 'Close' in df.columns:
                        fig = px.line(df, y='Close', title="Bitcoin Price Over Time")
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Close price data not available")
                
                with tab3:
                    st.subheader("Market Insights & Relationships")
                    
                    # Calculate returns and volatility for more meaningful analysis
                    if 'Close' in df.columns:
                        df_analysis = df.copy()
                        df_analysis['Returns'] = df_analysis['Close'].pct_change(fill_method=None)
                        df_analysis['Price_Change'] = df_analysis['Close'].diff()
                        df_analysis['Volatility'] = df_analysis['Returns'].rolling(window=7).std()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Price-Volume Relationship")
                            if 'Volume' in df_analysis.columns:
                                # Correlation between returns and volume
                                volume_corr = df_analysis[['Returns', 'Volume']].corr().iloc[0, 1] if len(df_analysis[['Returns', 'Volume']].dropna()) > 0 else 0
                                
                                fig = px.scatter(
                                    df_analysis.dropna(subset=['Returns', 'Volume']).tail(500),
                                    x='Volume',
                                    y='Close',
                                    color='Returns',
                                    title="Price vs Volume (colored by returns)",
                                    labels={'Close': 'Price (USD)', 'Volume': 'Trading Volume'},
                                    color_continuous_scale='RdYlGn',
                                    hover_data=['Returns']
                                )
                                st.plotly_chart(fig, width='stretch')
                                
                                st.info(f"""
                                **Insight:** Volume-Price correlation: {volume_corr:.3f}
                                - **Positive correlation** ({volume_corr:.3f}) suggests higher volume accompanies price movements
                                - Large price moves often occur with increased trading activity
                                - Watch for volume spikes during major price changes
                                """)
                        
                        with col2:
                            st.markdown("#### 📈 Returns Distribution Analysis")
                            returns_data = df_analysis['Returns'].dropna()
                            if len(returns_data) > 0:
                                fig = px.histogram(
                                    returns_data,
                                    nbins=50,
                                    title="Returns Distribution",
                                    labels={'value': 'Daily Returns (%)', 'count': 'Frequency'}
                                )
                                fig.update_traces(marker_color='steelblue')
                                st.plotly_chart(fig, width='stretch')
                                
                                # Calculate key statistics
                                mean_ret = returns_data.mean() * 100
                                std_ret = returns_data.std() * 100
                                skew = returns_data.skew()
                                kurt = returns_data.kurtosis()
                                
                                st.metric("Average Daily Return", f"{mean_ret:.3f}%")
                                st.metric("Daily Volatility (Std Dev)", f"{std_ret:.3f}%")
                                st.caption(f"Skewness: {skew:.2f} | Kurtosis: {kurt:.2f}")
                                
                                skew_desc = 'Positive = more large gains' if skew > 0 else 'Negative = more large losses'
                                kurt_desc = 'High = fat tails, extreme events more common' if abs(kurt) > 3 else 'Normal distribution'
                                risk_level = 'high' if std_ret > 5 else 'moderate' if std_ret > 2 else 'low'
                                
                                st.info(f"""
                                **Statistical Insights:**
                                - **Skewness ({skew:.2f}):** {skew_desc}
                                - **Kurtosis ({kurt:.2f}):** {kurt_desc}
                                - **Volatility:** {std_ret:.2f}% daily std dev indicates {risk_level} risk
                                """)
                        
                        st.divider()
                        
                        # Volatility clustering analysis
                        st.markdown("#### ⚠️ Volatility Clustering Detection")
                        if 'Volatility' in df_analysis.columns:
                            vol_data = df_analysis[['Close', 'Volatility']].dropna()
                            if len(vol_data) > 0:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=vol_data.tail(200).index,
                                    y=vol_data.tail(200)['Close'],
                                    name='Price',
                                    yaxis='y',
                                    line=dict(color='blue')
                                ))
                                fig.add_trace(go.Scatter(
                                    x=vol_data.tail(200).index,
                                    y=vol_data.tail(200)['Volatility'] * 100000,  # Scale for visibility
                                    name='Volatility (scaled)',
                                    yaxis='y2',
                                    line=dict(color='red')
                                ))
                                fig.update_layout(
                                    title="Price vs Rolling Volatility (Last 200 days)",
                                    xaxis_title="Time",
                                    yaxis=dict(title="Price (USD)", side='left'),
                                    yaxis2=dict(title="Volatility (scaled)", side='right', overlaying='y'),
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, width='stretch')
                                
                                st.warning("""
                                **Research Insight: Volatility Clustering**
                                - Volatility tends to cluster in time (high volatility followed by high volatility)
                                - This is a key characteristic of financial markets
                                - Periods of calm are followed by periods of calm, periods of turbulence by more turbulence
                                - This pattern challenges the assumption of constant volatility in many models
                                """)
                    else:
                        st.info("Close price data not available for advanced analysis.")
            else:
                st.error("Could not load Bitcoin data. Please ensure data/raw/btc_usd.parquet exists.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

# ==========================================
# PAGE 6: DATA DRIFT MONITOR
# ==========================================
elif page == "⚠️ Data Drift Monitor":
    st.header("Data Drift Detection")
    st.markdown("Monitor if incoming data has drifted from training distribution.")
    
    st.subheader("Test Transaction")
    col1, col2 = st.columns(2)
    
    with col1:
        amt = st.number_input("Transaction Amount ($)", value=150.0)
        anomaly_score = st.slider("Simulated Anomaly Score", 0.0, 10.0, 0.0)
        
        # Generate features similar to fraud page
        features = list(np.random.normal(0, 1, 28))
        if anomaly_score > 0:
            features[11] = -anomaly_score * 2
            features[13] = -anomaly_score * 3
            features[16] = -anomaly_score * 3
            features[3] = anomaly_score * 1.5
            features[10] = anomaly_score * 1.5
        
        final_features = features + [amt]
        final_features = [0.0] + final_features  # Add Time feature
    
    with col2:
        if st.button("Check for Drift"):
            try:
                response = requests.post(f"{API_URL}/monitor/drift", json={"features": final_features}, timeout=5)
                if response.status_code == 200:
                    drift_data = response.json()
                    
                    if drift_data['drift_detected']:
                        st.error(f"⚠️ Drift Detected! {len(drift_data['alerts'])} features drifted.")
                    else:
                        st.success("✅ No drift detected. Data is consistent with training distribution.")
                    
                    # Display drift scores
                    if drift_data.get('drift_scores'):
                        st.subheader("Drift Scores by Feature")
                        # Build drift dataframe with all available info
                        drift_rows = []
                        for feat, scores in drift_data['drift_scores'].items():
                            row = {
                                'Feature': feat,
                                'Drift Score': scores.get('drift_score', 0),
                                'Drifted': scores.get('drift_detected', False)
                            }
                            # Add z-score if available (single-row detection)
                            if 'z_score' in scores:
                                row['Z-Score'] = f"{scores.get('z_score', 0):.2f}"
                                row['Percentile'] = f"{scores.get('percentile', 0)*100:.1f}%"
                            # Add p-value if available (multi-row detection)
                            if scores.get('p_value') is not None:
                                row['P-Value'] = f"{scores.get('p_value', 1.0):.4f}"
                            else:
                                row['P-Value'] = 'N/A (single value)'
                            drift_rows.append(row)
                        
                        if drift_rows:
                            drift_df = pd.DataFrame(drift_rows)
                            
                            # Sort by drift score
                            drift_df = drift_df.sort_values('Drift Score', ascending=False)
                            
                            # Display table with color coding
                            st.dataframe(drift_df.style.apply(
                                lambda x: ['background-color: #ffcccc' if v else '' for v in x == True], 
                                axis=1, subset=['Drifted']
                            ))
                            
                            # Visualize top drifted features
                            top_drifted = drift_df.head(10)
                            if len(top_drifted) > 0:
                                fig = px.bar(top_drifted, x='Feature', y='Drift Score', 
                                            color='Drifted',
                                            title="Top 10 Features by Drift Score",
                                            color_discrete_map={True: 'red', False: 'green'})
                                st.plotly_chart(fig, width='stretch')
                        else:
                            st.warning("No drift data available for display.")
                    elif drift_scores and '_error' in drift_scores:
                        st.error(f"Error in drift detection: {drift_scores['_error']}")
                else:
                    st.error(f"API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.warning("⚠️ **API Backend Not Connected** - Drift detection requires the FastAPI service to be running.")
                st.info("💡 **Note:** In deployed environments, the API backend may need to be started separately or may not be available.")
            except requests.exceptions.Timeout:
                st.warning("⏱️ API request timed out. Please try again.")
            except Exception as e:
                error_msg = str(e)
                if "Connection refused" in error_msg or "ConnectionError" in error_msg:
                    st.warning("⚠️ **API Backend Not Connected** - Drift detection requires the FastAPI service.")
                else:
                    st.error(f"Error: {error_msg[:100]}...")

# ==========================================
# PAGE 7: MODEL PERFORMANCE
# ==========================================
elif page == "📊 Model Performance":
    st.header("Model Training Metrics")
    st.markdown("""
    Metrics from model training (accuracy, RMSE, F1-score, etc.)
    """)
    
    # Fallback training metrics from last successful training run
    fallback_metrics = {
                'fraud': {
                    'pr_auc': 0.8822,
                    'f1_score': 0.8743,
                    'precision': 0.9412,
                    'recall': 0.8163,
                    'best_threshold': 0.8254,
                    'last_trained': '2025-12-19 02:56:41'
                },
                'btc_price': {
                    'rmse': 4203.15,
                    'mae': 3516.38,
                    'mape_percent': 3.30,
                    'directional_accuracy_percent': 54.61,
                    'last_trained': '2025-12-19 02:57:02'
                },
                'segmentation': {
                    'silhouette_score': 0.8759,
                    'davies_bouldin': 0.1745,
                    'last_trained': '2025-12-19 02:57:06'
                }
    }
    
    training_metrics = None
    try:
        training_response = requests.get(f"{API_URL}/metrics/training", timeout=5)
        if training_response.status_code == 200:
            training_data = training_response.json()
            training_metrics = training_data.get('training_metrics', {})
    except requests.exceptions.ConnectionError:
        # Use fallback if API fails
        pass
    except requests.exceptions.Timeout:
        # Use fallback if API times out
        pass
    except Exception as e:
        # Use fallback on any other error
        pass
    
    # Use API data if available, otherwise use fallback
    if not training_metrics or len(training_metrics) == 0:
        training_metrics = fallback_metrics
        st.info("⚠️ Using cached training metrics (API unavailable). Run training script to update.")
    
    if training_metrics and len(training_metrics) > 0:
        # Display metrics for each model
        for model_name, metrics in training_metrics.items():
            with st.expander(f"**{model_name.upper()}** Model Training Metrics", expanded=True):
                # Format model name for display
                display_name = model_name.replace('_', ' ').title()
                
                # Create columns for metrics
                cols = st.columns(min(len(metrics), 4))
                
                metric_idx = 0
                for metric_name, metric_value in metrics.items():
                    if metric_name != 'last_trained' and isinstance(metric_value, (int, float)):
                        with cols[metric_idx % len(cols)]:
                            # Format metric name and value
                            formatted_name = metric_name.replace('_', ' ').title()
                            if 'percent' in metric_name.lower() or ('directional' in metric_name.lower() and 'accuracy' in metric_name.lower()):
                                formatted_value = f"{metric_value:.2f}%"
                            elif metric_name in ['pr_auc', 'silhouette_score', 'davies_bouldin', 'best_threshold']:
                                formatted_value = f"{metric_value:.4f}"
                            elif metric_name == 'rmse' or metric_name == 'mae':
                                formatted_value = f"${metric_value:.2f}"
                            elif metric_name in ['precision', 'recall', 'f1_score']:
                                formatted_value = f"{metric_value:.4f}"
                            else:
                                formatted_value = f"{metric_value:.4f}"
                            st.metric(formatted_name, formatted_value)
                        metric_idx += 1
                
                # Show last training date if available
                if 'last_trained' in metrics:
                    st.caption(f"Last trained: {metrics['last_trained']}")
    else:
        st.info("""
        **No training metrics available yet.**
        
        To populate training metrics:
        1. Run the training script: `python scripts/run_training_and_store_metrics.py`
        2. Or run individual training scripts:
           - `python src/training/train_fraud.py`
           - `python src/training/train_btc.py`
           - `python src/training/train_segmentation.py`
        """)
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
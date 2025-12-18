import yfinance as yf
import pandas as pd
import numpy as np
import os

def process_btc_data(ticker="BTC-USD", save_path="data/raw/btc_usd.parquet"):
    """
    Fetches LIVE BTC data from Yahoo Finance, engineers features, 
    and prepares targets for training.
    """
    print(f"⏳ Fetching live data for {ticker}...")
    
    try:
        # 1. Fetch Live Data (2 Years)
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        # 2. Fix MultiIndex Issues (Common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            raise ValueError("Yahoo Finance returned empty data.")

        # 3. Cache Data (MLOps Best Practice)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
        print(f"✅ Data cached to {save_path}")

    except Exception as e:
        print(f"⚠️ API Failed: {e}. Attempting to load from cache...")
        if os.path.exists(save_path):
            df = pd.read_parquet(save_path)
        else:
            print("❌ No cache found. Cannot proceed.")
            return None, None, None

    # 4. Feature Engineering
    df = df.copy()
    
    # Lag Features (Yesterday's Context)
    df['Lag_1'] = df['Close'].shift(1)
    
    # Returns & Volatility (Risk)
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=7).std()
    
    # Trends (Moving Averages)
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # 5. Create Targets (For Training)
    # Regression: What is the price tomorrow?
    df['Target_Price'] = df['Close'].shift(-1)
    # Classification: Will it go UP (1) or DOWN (0)?
    df['Target_Direction'] = (df['Target_Price'] > df['Close']).astype(int)

    # 6. Drop NaN (Must drop the LAST row because it has no target)
    # Note: We save the 'last_row' separately if we needed to forecast *tomorrow*
    # but for training, we must drop it.
    valid_data = df.dropna()

    # 7. Select Final Features
    features = ['Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility']
    
    X = valid_data[features]
    y_price = valid_data['Target_Price']
    y_dir = valid_data['Target_Direction']

    print(f"✅ BTC Data Processed: {len(valid_data)} training samples.")
    return X, y_price, y_dir
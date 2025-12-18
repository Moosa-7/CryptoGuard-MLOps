from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

app = FastAPI(title="FinTech Intelligence API", version="2.0")

# --- Global Model Storage ---
models = {}

@app.on_event("startup")
def load_artifacts():
    try:
        # Load all models
        models['fraud'] = joblib.load("models/fraud_model.pkl")
        models['fraud_pca'] = joblib.load("models/fraud_pca.pkl") 
        models['crypto'] = joblib.load("models/crypto_model.pkl")
        models['cluster'] = joblib.load("models/cluster_model.pkl")
        models['scaler'] = joblib.load("models/scaler.pkl")
        
        # Load Cluster Map (safe loading)
        try:
            models['cluster_map'] = joblib.load("models/cluster_map.pkl")
        except:
            models['cluster_map'] = {0: "Poor", 1: "Standard", 2: "VIP"}
            
        print("✅ Models Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# --- Input Schemas ---
class FraudInput(BaseModel):
    features: list[float] 

class UserInput(BaseModel):
    volume: float
    frequency: float

# --- Endpoints ---
@app.get("/")
def home():
    return {"status": "Online"}

@app.post("/predict/fraud")
def predict_fraud(data: FraudInput):
    # Fraud Model expects 29 features (28 PCA + Amount)
    if len(data.features) != 29:
        raise HTTPException(status_code=400, detail=f"Expected 29 features, got {len(data.features)}")
    
    try:
        # Reshape
        X = np.array(data.features).reshape(1, -1)
        # Apply PCA
        X_pca = models['fraud_pca'].transform(X)
        # Predict
        pred = models['fraud'].predict(X_pca)[0]
        # SMOTE models don't always support predict_proba well, but Random Forest does
        # We try/except to be safe
        try:
            prob = models['fraud'].predict_proba(X_pca)[0][1]
        except:
            prob = 1.0 if pred else 0.0

        return {"is_fraud": bool(pred), "risk_score": float(prob)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud Model Error: {str(e)}")

@app.post("/predict/btc")
def predict_btc():
    try:
        # Fetch Data (Fetch slightly more to ensure MA_7 calculation works)
        df = yf.download("BTC-USD", period="1mo", interval="1d", progress=False)
        
        # --- FIX: Check for Empty Data ---
        if df.empty or len(df) < 8:
            raise HTTPException(status_code=503, detail="Yahoo Finance Unavailable or Data too short.")
        # ---------------------------------

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Build Features matches the TRAINING data exactly: ['Lag_1', 'MA_7']
        lag_1 = float(df['Close'].iloc[-2])
        ma_7 = float(df['Close'].rolling(window=7).mean().iloc[-1])
        
        # Predict
        input_data = pd.DataFrame([[lag_1, ma_7]], columns=['Lag_1', 'MA_7'])
        pred = models['crypto'].predict(input_data)[0]
        
        return {
            "current_price": float(df['Close'].iloc[-1]),
            "predicted_next_day": float(pred),
            "trend": "UP" if pred > df['Close'].iloc[-1] else "DOWN"
        }
    except Exception as e:
        print(f"Crypto Error: {e}") # Print to terminal for debugging
        raise HTTPException(status_code=500, detail=f"Crypto Error: {str(e)}")

@app.post("/segment")
def segment_user(data: UserInput):
    try:
        # Scale Input
        X_scaled = models['scaler'].transform([[data.volume, data.frequency]])
        
        # Predict ID
        cluster_id = int(models['cluster'].predict(X_scaled)[0])
        
        # Get Name
        segment_name = models['cluster_map'].get(cluster_id, "Unknown")
        
        # Description
        if segment_name == "VIP":
            desc = "High Net Worth Individual. Priority Support."
        elif segment_name == "Standard":
            desc = "Regular Customer. Standard Services."
        else:
            desc = "Low Balance User. Basic Access."

        return {
            "cluster_id": cluster_id,
            "segment_name": segment_name, 
            "description": desc,
            "recommendations": [] # Kept empty as requested
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segment Error: {str(e)}")
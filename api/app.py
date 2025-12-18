from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

app = FastAPI(
    title="FinTech Intelligence API",
    description="End-to-End MLOps Pipeline for Fraud, Crypto, and Segmentation",
    version="2.0"
)

# --- Global Model Storage ---
models = {}

@app.on_event("startup")
def load_artifacts():
    try:
        # Load all the new "Pro" models
        models['fraud'] = joblib.load("models/fraud_model.pkl")
        models['fraud_pca'] = joblib.load("models/fraud_pca.pkl")  # NEW
        models['crypto'] = joblib.load("models/crypto_model.pkl")
        models['cluster'] = joblib.load("models/cluster_model.pkl")
        models['scaler'] = joblib.load("models/scaler.pkl")        # NEW
        # Try loading rules, handle error if file missing (optional)
        try:
            models['rules'] = joblib.load("models/association_rules.pkl") # NEW
        except:
            models['rules'] = None
            
        print("✅ All Advanced Models Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# --- Input Schemas ---
class FraudInput(BaseModel):
    # Expecting 29 raw features (V1-V28 + Amount)
    features: list[float] 

class UserInput(BaseModel):
    volume: float
    frequency: float

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "System Operational", "mode": "Production"}

@app.post("/predict/fraud")
def predict_fraud(data: FraudInput):
    if len(data.features) != 29:
        raise HTTPException(status_code=400, detail="Expected 29 features")
    
    # 1. Reshape
    X = np.array(data.features).reshape(1, -1)
    
    # 2. Apply PCA (Crucial step added)
    # The model expects 10 PCA components, not 29 raw features
    X_pca = models['fraud_pca'].transform(X)
    
    # 3. Predict
    pred = models['fraud'].predict(X_pca)[0]
    prob = models['fraud'].predict_proba(X_pca)[0][1]
    
    return {
        "is_fraud": bool(pred),
        "risk_score": float(prob),
        "method": "PCA + Random Forest"
    }

@app.post("/predict/btc")
def predict_btc():
    # Fetch live data to construct "Lag Features"
    try:
        df = yf.download("BTC-USD", period="10d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Create features: [Lag_1, Lag_7, MA_7]
        last_close = float(df['Close'].iloc[-1])
        lag_1 = float(df['Close'].iloc[-2]) # Yesterday
        lag_7 = float(df['Close'].iloc[-8]) # Week ago
        ma_7 = float(df['Close'].rolling(window=7).mean().iloc[-1])
        
        # Prepare input for XGBoost
        # Note: The model expects specific feature names or order
        input_data = pd.DataFrame([[lag_1, lag_7, ma_7]], columns=['Lag_1', 'Lag_7', 'MA_7'])
        
        pred = models['crypto'].predict(input_data)[0]
        
        return {
            "current_price": last_close,
            "predicted_next_day": float(pred),
            "trend": "UP" if pred > last_close else "DOWN"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
def segment_user(data: UserInput):
    try:
        # 1. Scale Input
        X_scaled = models['scaler'].transform([[data.volume, data.frequency]])
        
        # 2. Predict ID
        cluster_id = int(models['cluster'].predict(X_scaled)[0])
        
        # 3. Get Name (Poor / Standard / VIP)
        segment_name = models['cluster_map'].get(cluster_id, "Unknown")
        
        # 4. Simple Description
        if segment_name == "VIP":
            description = "High Net Worth Individual. Priority Support."
        elif segment_name == "Standard":
            description = "Regular Customer. Standard Banking Services."
        else:
            description = "Low Balance User. Basic Access Only."

        return {
            "cluster_id": cluster_id,
            "segment_name": segment_name, 
            "description": description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
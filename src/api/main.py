from fastapi import FastAPI, HTTPException
import pandas as pd
from src.inference.model_loader import ModelLoader
from src.api.schemas import FraudInput, BTCInput, SegmentationInput

# 1. Initialize App & Loader
app = FastAPI(title="CryptoGuard AI API", version="1.0.0")
loader = ModelLoader()

@app.get("/")
def home():
    return {"message": "CryptoGuard AI is Online 🟢", "docs": "/docs"}

@app.get("/health")
def health_check():
    """Checks if models are loaded and ready."""
    status = {
        "fraud_model": loader.fraud_model is not None,
        "btc_model": loader.btc_price_model is not None,
        "segmentation_model": loader.segmentation_model is not None
    }
    return status

# --- FRAUD ENDPOINT ---
@app.post("/predict/fraud")
def predict_fraud(data: FraudInput):
    """
    Detects if a transaction is fraudulent.
    Input: List of 30 floats [Time, V1-V28, Amount]
    """
    if not loader.fraud_model:
        raise HTTPException(status_code=503, detail="Fraud model not loaded")
    
    # Convert list to DataFrame (Model expects 2D array)
    # We use a dummy DataFrame to silence feature name warnings
    df_input = pd.DataFrame([data.features])
    
    is_fraud, prob, risk = loader.predict_fraud(df_input)
    
    return {
        "is_fraud": is_fraud,
        "probability": round(prob, 4),
        "risk_level": risk,
        "action": "BLOCK" if is_fraud else "APPROVE"
    }

# --- BTC ENDPOINT ---
@app.post("/predict/btc")
def predict_btc(data: BTCInput):
    """
    Forecasts Bitcoin Price and Direction.
    """
    if not loader.btc_price_model:
        raise HTTPException(status_code=503, detail="BTC model not loaded")
    
    # Use the helper method from schemas.py
    df_input = pd.DataFrame(data.to_df_list(), columns=[
        'Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility'
    ])
    
    price, signal = loader.predict_btc(df_input)
    
    return {
        "current_price": data.close,
        "predicted_next_price": round(price, 2),
        "signal": signal,
        "confidence": "High" if signal != "Hold" else "Neutral"
    }

# --- SEGMENTATION ENDPOINT ---
@app.post("/predict/segment")
def predict_segment(data: SegmentationInput):
    """
    Classifies a user into a cluster based on PCA features.
    """
    if not loader.segmentation_model:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    df_input = pd.DataFrame([[data.pc1, data.pc2]], columns=['PC1', 'PC2'])
    
    cluster_id = loader.get_segment(df_input)
    
    # Map ID to human name (based on our training logic)
    # Note: K-Means labels change, but usually 0=Casual, 1=Regular, etc.
    # For now, we return the raw ID.
    return {
        "cluster_id": cluster_id,
        "segment_name": f"Group {cluster_id}"
    }

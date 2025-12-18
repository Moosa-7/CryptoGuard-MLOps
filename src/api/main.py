from fastapi import FastAPI, HTTPException
from typing import Optional
import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.inference.model_loader import ModelLoader
from src.api.schemas import FraudInput, BTCInput, SegmentationInput, PerformanceLogInput
from src.interpretability.explainers import (
    explain_fraud_with_shap, explain_fraud_with_lime,
    explain_btc_with_shap, get_feature_importance
)
from src.monitoring.drift_detection import detect_data_drift
from src.monitoring.performance_tracker import (
    init_performance_db, log_prediction, get_performance_metrics,
    get_prediction_history, get_all_models_metrics, get_training_metrics,
    get_db_path
)
from src.utils.prediction_intervals import get_prediction_intervals

# 1. Initialize App & Loader
app = FastAPI(title="CryptoGuard AI API", version="1.0.0")
loader = ModelLoader()

# 2. Ensure training samples are loaded on startup
# (This is already called in ModelLoader.__init__, but we verify)
if loader.fraud_training_sample is None:
    loader._load_training_samples()

# 3. Initialize Performance Tracking Database (uses consistent path helper)
perf_db = init_performance_db()  # Uses get_db_path() internally

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
    
    # Convert to DataFrame - features are [Time, V1-V28, Amount] = 30 features
    # We need to drop Time (first feature) and keep V1-V28 + Amount
    X_raw = pd.DataFrame([data.features[1:30]], columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
    
    # Apply PCA if available (fraud model expects PCA features)
    if loader.fraud_pca is not None:
        X_pca = loader.fraud_pca.transform(X_raw)
        df_input = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    else:
        # If no PCA, use raw features
        df_input = X_raw
    
    is_fraud, prob, risk = loader.predict_fraud(df_input)
    
    # Log prediction for monitoring
    try:
        log_prediction(perf_db, "fraud", float(is_fraud), prob)
    except Exception as e:
        print(f"Error logging prediction: {e}")
    
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
    
    # Get prediction intervals
    lower_bound, upper_bound = None, None
    try:
        lower, upper = get_prediction_intervals(loader.btc_price_model, df_input)
        lower_bound = float(lower[0])
        upper_bound = float(upper[0])
    except Exception as e:
        print(f"Error calculating prediction intervals: {e}")
    
    # Log prediction for monitoring
    try:
        confidence_score = 0.8 if signal != "Hold" else 0.5
        log_prediction(perf_db, "btc_price", float(price), confidence_score)
    except Exception as e:
        print(f"Error logging prediction: {e}")
    
    result = {
        "current_price": data.close,
        "predicted_next_price": round(price, 2),
        "signal": signal,
        "confidence": "High" if signal != "Hold" else "Neutral"
    }
    
    if lower_bound is not None and upper_bound is not None:
        result["prediction_interval"] = {
            "lower": round(lower_bound, 2),
            "upper": round(upper_bound, 2)
        }
    
    return result

# --- SEGMENTATION ENDPOINT ---
@app.post("/predict/segment")
def predict_segment(data: SegmentationInput):
    """
    Classifies a user into a cluster based on PCA features.
    VIP users have high PC1 and PC2 values (typically > 7).
    """
    if not loader.segmentation_model:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")
    
    df_input = pd.DataFrame([[data.pc1, data.pc2]], columns=['PC1', 'PC2'])
    
    cluster_id = loader.get_segment(df_input)
    
    # Determine segment based on PC1 and PC2 values
    # High values (typically > 7) indicate VIP users
    # Visualization shows VIP around (12, 12) and Standard around (2, 2)
    threshold = 7.0
    if data.pc1 > threshold or data.pc2 > threshold:
        segment_name = "VIP / High-Net-Worth"
        confidence_score = 0.9  # High confidence for VIP
    else:
        segment_name = "Standard Tier"
        confidence_score = 0.7  # Moderate confidence for Standard
    
    # Log prediction for monitoring
    try:
        # Use cluster_id as prediction value (0, 1, or 2)
        log_prediction(perf_db, "segmentation", float(cluster_id), confidence_score)
    except Exception as e:
        print(f"Error logging segmentation prediction: {e}")
    
    return {
        "cluster_id": cluster_id,
        "segment_name": segment_name
    }

# --- EXPLANATION ENDPOINTS ---

@app.post("/explain/fraud")
def explain_fraud(data: FraudInput):
    """
    Returns SHAP and LIME explanations for fraud predictions.
    """
    if not loader.fraud_model:
        raise HTTPException(status_code=503, detail="Fraud model not loaded")
    
    if loader.fraud_training_sample is None:
        raise HTTPException(status_code=503, detail="Training samples not loaded")
    
    # Convert to DataFrame and apply PCA
    X_raw = pd.DataFrame([data.features[1:30]], columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
    
    if loader.fraud_pca is not None:
        X_pca = loader.fraud_pca.transform(X_raw)
        df_input = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    else:
        df_input = X_raw
        feature_names = list(X_raw.columns)
    
    # Get prediction
    is_fraud, prob, risk = loader.predict_fraud(df_input)
    
    # Get SHAP explanation
    shap_values, shap_importance = explain_fraud_with_shap(
        loader.fraud_model,
        df_input,
        loader.fraud_training_sample
    )
    
    # Get LIME explanation (use original feature names for interpretability)
    lime_explanation = None
    if loader.fraud_pca is not None and loader.fraud_training_sample is not None:
        # For LIME, use PCA space
        lime_explanation = explain_fraud_with_lime(
            loader.fraud_model,
            df_input,
            feature_names,
            loader.fraud_training_sample
        )
    
    return {
        "prediction": {
            "is_fraud": is_fraud,
            "probability": round(prob, 4),
            "risk_level": risk
        },
        "shap_explanation": {
            "feature_importance": shap_importance.to_dict('records') if shap_importance is not None else None,
            "shap_values": shap_values.tolist() if shap_values is not None else None
        },
        "lime_explanation": lime_explanation
    }

@app.post("/explain/btc")
def explain_btc(data: BTCInput):
    """
    Returns SHAP explanations for BTC price predictions.
    """
    if not loader.btc_price_model:
        raise HTTPException(status_code=503, detail="BTC model not loaded")
    
    if loader.btc_training_sample is None:
        raise HTTPException(status_code=503, detail="BTC training samples not loaded")
    
    # Convert to DataFrame
    df_input = pd.DataFrame(data.to_df_list(), columns=[
        'Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility'
    ])
    
    # Get prediction
    price, signal = loader.predict_btc(df_input)
    
    # Get SHAP explanation
    shap_values, shap_importance = explain_btc_with_shap(
        loader.btc_price_model,
        df_input,
        loader.btc_training_sample
    )
    
    # Get feature importance
    feature_names = ['Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility']
    feature_importance = get_feature_importance(loader.btc_price_model, feature_names)
    
    return {
        "prediction": {
            "predicted_price": round(price, 2),
            "signal": signal
        },
        "shap_explanation": {
            "feature_importance": shap_importance.to_dict('records') if shap_importance is not None else None,
            "shap_values": shap_values.tolist() if shap_values is not None else None
        },
        "feature_importance": feature_importance.to_dict('records') if feature_importance is not None else None
    }

# --- MONITORING ENDPOINTS ---

@app.post("/monitor/drift")
def check_drift(data: FraudInput):
    """
    Check if incoming data has drifted from training distribution.
    """
    # Try to load training sample if not already loaded
    if loader.fraud_training_sample is None:
        loader._load_training_samples()
    
    if loader.fraud_training_sample is None:
        raise HTTPException(
            status_code=503, 
            detail="Reference training data not loaded. Please ensure data/creditcard.csv exists and models are trained."
        )
    
    # Convert to DataFrame and apply PCA
    X_raw = pd.DataFrame([data.features[1:30]], columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
    
    if loader.fraud_pca is not None:
        X_pca = loader.fraud_pca.transform(X_raw)
        df_input = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    else:
        df_input = X_raw
    
    # Detect drift
    try:
        drift_results = detect_data_drift(loader.fraud_training_sample, df_input)
        
        # Find drifted features
        alerts = [k for k, v in drift_results.items() if v.get('drift_detected', False)]
        
        return {
            "drift_detected": len(alerts) > 0,
            "alerts": alerts,
            "drift_scores": drift_results
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in drift detection: {e}")
        print(error_details)
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting drift: {str(e)}"
        )

@app.get("/metrics/performance")
def get_performance():
    """
    Get model performance metrics for all models.
    """
    try:
        all_metrics = get_all_models_metrics(perf_db)
        
        # Get prediction history for all models
        fraud_history = get_prediction_history(perf_db, "fraud", limit=100)
        btc_history = get_prediction_history(perf_db, "btc_price", limit=100)
        seg_history = get_prediction_history(perf_db, "segmentation", limit=100)
        
        # Combine histories for overall view
        all_history = fraud_history + btc_history + seg_history
        
        # Aggregate metrics
        total_predictions = sum(m.get('total_predictions', 0) for m in all_metrics.values())
        frauds_detected = sum(1 for h in fraud_history if h.get('prediction', 0) >= 0.5)
        
        # Calculate average confidence from all predictions
        confidences = [h.get('confidence', 0) for h in all_history if h.get('confidence') is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Count last 24h predictions
        last_24h = 0
        try:
            fraud_24h = get_performance_metrics(perf_db, "fraud", "24h")
            btc_24h = get_performance_metrics(perf_db, "btc_price", "24h")
            seg_24h = get_performance_metrics(perf_db, "segmentation", "24h")
            last_24h = fraud_24h.get('total_predictions', 0) + btc_24h.get('total_predictions', 0) + seg_24h.get('total_predictions', 0)
        except Exception as e:
            print(f"Error getting 24h metrics: {e}")
        
        return {
            "total_predictions": total_predictions,
            "frauds_detected": frauds_detected,
            "avg_confidence": round(avg_confidence, 4),
            "last_24h": last_24h,
            "prediction_history": all_history[:50],  # Last 50 for chart
            "models": all_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@app.post("/metrics/log")
def log_metric(data: PerformanceLogInput):
    """
    Log a prediction for monitoring (internal use).
    """
    try:
        log_prediction(perf_db, data.model_name, data.prediction, data.confidence)
        return {"status": "logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging metric: {str(e)}")

# --- FEATURE IMPORTANCE ENDPOINT ---

@app.get("/features/importance/{model_name}")
def get_feature_importance_endpoint(model_name: str):
    """
    Get feature importance for tree-based models.
    """
    model = None
    feature_names = []
    
    if model_name == "fraud":
        model = loader.fraud_model
        if loader.fraud_pca is not None:
            feature_names = [f'PC{i+1}' for i in range(loader.fraud_pca.n_components_)]
        else:
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
    elif model_name == "btc_price":
        model = loader.btc_price_model
        feature_names = ['Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility']
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    if model is None:
        raise HTTPException(status_code=503, detail=f"{model_name} model not loaded")
    
    importance_df = get_feature_importance(model, feature_names)
    
    if importance_df is None:
        raise HTTPException(status_code=400, detail="Feature importance not available for this model type")
    
    return {
        "features": importance_df['feature'].tolist(),
        "importance": importance_df['importance'].tolist()
    }

# --- TRAINING METRICS ENDPOINT ---

@app.get("/metrics/training")
def get_training_metrics_endpoint(model_name: Optional[str] = None):
    """
    Get training metrics for all models or a specific model.
    These are metrics from when the model was trained (e.g., accuracy, RMSE, F1).
    """
    try:
        training_metrics = get_training_metrics(perf_db, model_name)
        return {
            "training_metrics": training_metrics,
            "note": "These are metrics from model training, not live predictions"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error getting training metrics: {e}")
        print(error_details)
        raise HTTPException(status_code=500, detail=f"Error getting training metrics: {str(e)}")

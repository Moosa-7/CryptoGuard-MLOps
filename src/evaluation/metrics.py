import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score,
    davies_bouldin_score,
    precision_score,
    recall_score
)

# ==========================================
# TASK 1: FRAUD DETECTION (Imbalanced Class)
# ==========================================

def get_fraud_metrics(y_true, y_prob):
    """
    Calculates metrics suitable for extreme imbalance.
    HARDENED: Falls back to 0.5 threshold if model has no skill.
    """
    # 1. Calculate PR-AUC (The Gold Standard)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # 2. Find Optimal Threshold (Maximize F1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores) # Handle division by zero
    
    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[best_idx]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        max_f1 = 0.0
        best_threshold = 0.5

    # --- SAFETY GUARDRAIL ---
    # If the best F1 is terrible (model is guessing), default to standard 0.5
    # This prevents the "Predict All 1s" strategy from winning.
    if max_f1 < 0.1:
        best_threshold = 0.5

    # 3. Apply Threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    return {
        "pr_auc": float(pr_auc),
        "best_threshold": float(best_threshold),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

# ==========================================
# TASK 2: BITCOIN FORECASTING (Time Series)
# ==========================================

def get_btc_metrics(y_true, y_pred, y_prev):
    """
    Evaluates price prediction AND trading signal quality.
    y_prev: The closing price of the previous day (needed for direction).
    """
    # 1. Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 2. Directional Accuracy (The Money Maker)
    # Did we correctly predict Up vs Down?
    true_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)
    direction_acc = np.mean(true_direction == pred_direction) * 100
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape_percent": float(mape),
        "directional_accuracy_percent": float(direction_acc) # PRIMARY METRIC
    }

# ==========================================
# TASK 3: USER SEGMENTATION (Clustering)
# ==========================================

def get_segmentation_metrics(X, labels):
    """
    Evaluates cluster separation and compactness.
    """
    # Silhouette: Higher is better (-1 to 1)
    # Davies-Bouldin: Lower is better (0 to inf)
    try:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
    except ValueError:
        # Fails if only 1 cluster found
        sil = -1.0
        db = 999.0
        
    return {
        "silhouette_score": float(sil),   # PRIMARY METRIC
        "davies_bouldin": float(db)
    }
import numpy as np
import pytest
from src.evaluation.metrics import get_fraud_metrics, get_btc_metrics, get_segmentation_metrics

# ==========================================
# 1. Fraud Metric Tests (Imbalance Handling)
# ==========================================

def test_fraud_metrics_imbalance():
    """
    Scenario: 95 Legit transactions (0), 5 Fraud (1).
    We test a 'Lazy Model' that predicts 0.0 probability for everyone.
    """
    y_true = np.array([0]*95 + [1]*5)
    
    # Model 1: Terrible model (predicts 0.0 probability for everything)
    y_prob_bad = np.array([0.0]*100)
    metrics_bad = get_fraud_metrics(y_true, y_prob_bad)
    
    # Validation:
    # 1. Recall must be 0.0 (It caught nothing)
    assert metrics_bad['recall'] == 0.0
    # 2. PR-AUC should be calculable (even if poor)
    assert metrics_bad['pr_auc'] >= 0.0
    # 3. Guardrail Check: Best threshold should default to 0.5 because F1 was 0
    assert metrics_bad['best_threshold'] == 0.5

def test_fraud_metrics_perfect():
    """
    Scenario: Perfect predictions.
    """
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.95]) # Clear separation
    
    metrics = get_fraud_metrics(y_true, y_prob)
    
    assert metrics['recall'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['pr_auc'] == 1.0

# ==========================================
# 2. Bitcoin Metric Tests (Directional Logic)
# ==========================================

def test_btc_directional_logic():
    """
    Verifies that we measure the 'Move', not just the 'Price'.
    """
    # History: Price was 100 yesterday.
    y_prev = np.array([100, 100])
    
    # Reality: Today it went to 110 (UP) and 90 (DOWN).
    y_true = np.array([110, 90])
    
    # Prediction: Model predicted 105 (UP) and 95 (DOWN).
    # Note: The *prices* are wrong (off by 5), but the *direction* is correct!
    y_pred = np.array([105, 95])
    
    metrics = get_btc_metrics(y_true, y_pred, y_prev)
    
    # Directional Accuracy should be 100% (2/2 correct moves)
    assert metrics['directional_accuracy_percent'] == 100.0
    # RMSE should calculate the error (5^2 = 25, sqrt = 5)
    assert metrics['rmse'] == 5.0

# ==========================================
# 3. Segmentation Metric Tests
# ==========================================

def test_segmentation_metrics():
    """
    Tests clustering quality scores.
    """
    # 2D Data points: Two distinct clusters
    X = np.array([
        [1, 1], [1, 2], [2, 1],  # Cluster A (Bottom Left)
        [10, 10], [10, 11], [11, 10] # Cluster B (Top Right)
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    
    metrics = get_segmentation_metrics(X, labels)
    
    # Silhouette score ranges from -1 to 1. 
    # Distinct clusters should be very high (> 0.5).
    assert metrics['silhouette_score'] > 0.8 
    # Davies-Bouldin should be low (good separation)
    assert metrics['davies_bouldin'] < 1.0
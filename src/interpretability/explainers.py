import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from lime.lime_tabular import LimeTabularExplainer

def explain_fraud_with_shap(model, X_sample: pd.DataFrame, X_train_background: pd.DataFrame):
    """
    Generate SHAP explanations for fraud detection model.
    
    Args:
        model: Trained fraud detection model
        X_sample: Single prediction instance (DataFrame with 1 row)
        X_train_background: Training data sample for background (DataFrame)
    
    Returns:
        tuple: (shap_values, feature_importance_df) or (None, None) if error
    """
    try:
        # Limit background size for performance (SHAP can be slow)
        background_size = min(100, len(X_train_background))
        X_background = X_train_background.sample(n=background_size, random_state=42)
        
        # For XGBoost models, use TreeExplainer (fast)
        if hasattr(model, 'get_booster'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # For other models, use KernelExplainer with small background
            def model_predict_proba_wrapper(X):
                return model.predict_proba(X)
            
            explainer = shap.KernelExplainer(model_predict_proba_wrapper, X_background)
            shap_values = explainer.shap_values(X_sample)
        
        # Handle list of arrays (multi-class) - take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get fraud class (positive)
        
        # Get feature importance (mean absolute SHAP values)
        if len(shap_values.shape) > 1:
            importance_array = np.abs(shap_values).mean(axis=0)
        else:
            importance_array = np.abs(shap_values)
        
        # Create feature importance DataFrame
        feature_names = [f'PC{i+1}' for i in range(len(importance_array))]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_array
        }).sort_values('importance', ascending=False)
        
        return shap_values, feature_importance
        
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        return None, None

def explain_fraud_with_lime(model, X_sample: pd.DataFrame, feature_names: List[str], X_train: pd.DataFrame):
    """
    Generate LIME explanations for fraud detection model.
    
    Args:
        model: Trained fraud detection model
        X_sample: Single prediction instance (DataFrame with 1 row)
        feature_names: List of feature names
        X_train: Training data for explainer
    
    Returns:
        dict: Feature contributions or None if error
    """
    try:
        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=['Legitimate', 'Fraud'],
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            X_sample.values[0],
            model.predict_proba,
            num_features=min(10, len(feature_names))
        )
        
        # Extract explanation as dictionary
        explanation_dict = dict(exp.as_list())
        return explanation_dict
        
    except Exception as e:
        print(f"LIME explanation error: {e}")
        return None

def explain_btc_with_shap(model, X_sample: pd.DataFrame, X_train_background: pd.DataFrame):
    """
    Generate SHAP explanations for BTC price prediction model.
    
    Args:
        model: Trained BTC price model
        X_sample: Single prediction instance (DataFrame with 1 row)
        X_train_background: Training data sample for background
    
    Returns:
        tuple: (shap_values, feature_importance_df) or (None, None) if error
    """
    try:
        # Limit background size
        background_size = min(100, len(X_train_background))
        X_background = X_train_background.sample(n=background_size, random_state=42)
        
        # For tree models, use TreeExplainer
        if hasattr(model, 'get_booster') or hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # For other models, use KernelExplainer
            explainer = shap.KernelExplainer(model.predict, X_background)
            shap_values = explainer.shap_values(X_sample)
        
        # Get feature importance
        if len(shap_values.shape) > 1:
            importance_array = np.abs(shap_values).mean(axis=0)
        else:
            importance_array = np.abs(shap_values)
        
        feature_names = ['Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_array
        }).sort_values('importance', ascending=False)
        
        return shap_values, feature_importance
        
    except Exception as e:
        print(f"BTC SHAP explanation error: {e}")
        return None, None

def get_feature_importance(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance or None if not available
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_array = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importance_array = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        else:
            return None
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_array
        }).sort_values('importance', ascending=False)
        
        return feature_importance
        
    except Exception as e:
        print(f"Feature importance extraction error: {e}")
        return None


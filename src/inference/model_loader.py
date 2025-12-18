import joblib
import os
import pandas as pd
import numpy as np

class ModelLoader:
    def __init__(self, models_dir="models"):
        # Use absolute path to avoid "file not found" errors
        self.models_dir = os.path.abspath(models_dir)
        self.fraud_model = None
        self.fraud_threshold = 0.5
        self.btc_price_model = None
        self.btc_direction_model = None
        self.segmentation_model = None
        
        self._load_models()

    def _load_models(self):
        """
        Loads all models from disk. Fails gracefully if files are missing.
        """
        print(f"🔄 Loading Models from: {self.models_dir}")
        
        # 1. Load Fraud Model & Threshold
        try:
            model_path = os.path.join(self.models_dir, "fraud/xgboost_best.pkl")
            if os.path.exists(model_path):
                self.fraud_model = joblib.load(model_path)
                
                # Load threshold if exists
                thresh_path = os.path.join(self.models_dir, "fraud/threshold.pkl")
                if os.path.exists(thresh_path):
                    self.fraud_threshold = joblib.load(thresh_path)
                print(f"   ✅ Fraud Model Loaded (Threshold: {self.fraud_threshold:.4f})")
            else:
                print("   ⚠️ Fraud Model file not found.")
        except Exception as e:
            print(f"   ⚠️ Fraud Model Error: {e}")

        # 2. Load BTC Models
        try:
            # Check for Price Model
            price_path = os.path.join(self.models_dir, "btc/xgboost_price.pkl")
            # Fallback to MLflow generic name if specific one missing
            if not os.path.exists(price_path):
                 price_path = os.path.join(self.models_dir, "btc/model.pkl")
            
            if os.path.exists(price_path):
                self.btc_price_model = joblib.load(price_path)
                print("   ✅ BTC Price Model Loaded")
                
            # Check for Direction Model
            dir_path = os.path.join(self.models_dir, "btc/xgboost_direction.pkl")
            if os.path.exists(dir_path):
                self.btc_direction_model = joblib.load(dir_path)
                print("   ✅ BTC Direction Model Loaded")
        except Exception as e:
            print(f"   ⚠️ BTC Models Error: {e}")

        # 3. Load Segmentation
        try:
            seg_path = os.path.join(self.models_dir, "segmentation/best_clustering.pkl")
            if os.path.exists(seg_path):
                self.segmentation_model = joblib.load(seg_path)
                print("   ✅ Segmentation Model Loaded")
            else:
                print("   ⚠️ Segmentation Model file not found.")
        except Exception as e:
            print(f"   ⚠️ Segmentation Model Error: {e}")

    def predict_fraud(self, features: pd.DataFrame):
        if not self.fraud_model:
            return False, 0.0, "System Error"
            
        prob = self.fraud_model.predict_proba(features)[:, 1][0]
        is_fraud = prob >= self.fraud_threshold
        
        risk = "Critical" if prob > 0.9 else "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        return bool(is_fraud), float(prob), risk

    def predict_btc(self, features: pd.DataFrame):
        price = 0.0
        signal = "Hold"
        
        if self.btc_price_model:
            price = float(self.btc_price_model.predict(features)[0])
            
        if self.btc_direction_model:
            # Check if model has predict_proba (classifiers)
            if hasattr(self.btc_direction_model, "predict_proba"):
                prob = self.btc_direction_model.predict_proba(features)[:, 1][0]
                if prob > 0.6: signal = "Buy"
                elif prob < 0.4: signal = "Sell"
            else:
                # Fallback for regressors
                pred = self.btc_direction_model.predict(features)[0]
                signal = "Buy" if pred == 1 else "Sell"
                
        return price, signal

    def get_segment(self, features: pd.DataFrame):
        if not self.segmentation_model:
            return -1
        return int(self.segmentation_model.predict(features)[0])
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
        self.fraud_pca = None
        self.btc_price_model = None
        self.btc_direction_model = None
        self.segmentation_model = None
        
        # Training samples for explanations and drift detection
        self.fraud_training_sample = None
        self.btc_training_sample = None
        self.fraud_reference_stats = None
        
        self._load_models()
        self._load_training_samples()

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
                
                # Load PCA transformer if exists
                pca_path = os.path.join(self.models_dir, "fraud_pca.pkl")
                if os.path.exists(pca_path):
                    self.fraud_pca = joblib.load(pca_path)
                    print(f"   ✅ Fraud Model & PCA Loaded (Threshold: {self.fraud_threshold:.4f})")
                else:
                    print(f"   ✅ Fraud Model Loaded (Threshold: {self.fraud_threshold:.4f})")
                    print("   ⚠️ PCA transformer not found (may need for explanations)")
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
    
    def _find_data_file(self, filename):
        """
        Find data file in multiple possible locations.
        """
        possible_paths = [
            os.path.join(os.getcwd(), filename),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), filename),  # Project root
            os.path.join(self.models_dir.replace('models', ''), filename),  # Relative to models dir
            os.path.join(os.path.abspath('.'), filename),
            filename  # Try relative path as last resort
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _load_training_samples(self):
        """
        Load training data samples for explanations and drift detection.
        """
        try:
            # Load fraud training data - use helper to find file
            fraud_data_path = self._find_data_file("data/creditcard.csv")
            if fraud_data_path and os.path.exists(fraud_data_path):
                df_fraud = pd.read_csv(fraud_data_path)
                # Sample 1000 rows for performance
                df_fraud_sample = df_fraud.sample(n=min(1000, len(df_fraud)), random_state=42)
                
                # Apply PCA if available
                if self.fraud_pca is not None:
                    X_fraud = df_fraud_sample.drop(['Class', 'Time'], axis=1, errors='ignore')
                    X_fraud_pca = self.fraud_pca.transform(X_fraud)
                    self.fraud_training_sample = pd.DataFrame(
                        X_fraud_pca, 
                        columns=[f'PC{i+1}' for i in range(X_fraud_pca.shape[1])]
                    )
                    print("   ✅ Fraud training sample loaded (PCA transformed)")
                else:
                    # Store raw features sample
                    X_fraud = df_fraud_sample.drop(['Class', 'Time'], axis=1, errors='ignore')
                    self.fraud_training_sample = X_fraud.iloc[:1000]
                    print("   ✅ Fraud training sample loaded (raw features)")
                
                # Calculate reference statistics for drift detection (we'll do this lazily if needed)
                self.fraud_reference_stats = None  # Will be calculated on demand
            else:
                checked_paths = [
                    os.path.join(os.getcwd(), "data/creditcard.csv"),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data/creditcard.csv"),
                    "data/creditcard.csv"
                ]
                print(f"   ⚠️ Fraud training data not found. Checked paths: {checked_paths}")
        except Exception as e:
            print(f"   ⚠️ Error loading training samples: {e}")
        
        try:
            # Load BTC training data (if available) - use helper
            btc_data_path = self._find_data_file("data/raw/btc_usd.parquet")
            if btc_data_path and os.path.exists(btc_data_path):
                df_btc = pd.read_parquet(btc_data_path)
                # Extract feature columns if they exist
                feature_cols = ['Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility']
                available_cols = [col for col in feature_cols if col in df_btc.columns]
                if available_cols:
                    self.btc_training_sample = df_btc[available_cols].dropna().iloc[:1000]
                    print("   ✅ BTC training sample loaded")
        except Exception as e:
            print(f"   ⚠️ Error loading BTC training sample: {e}")
    
    def get_reference_statistics(self):
        """
        Get reference statistics for drift detection.
        """
        if self.fraud_reference_stats is None and self.fraud_training_sample is not None:
            try:
                from src.monitoring.drift_detection import calculate_reference_statistics
                self.fraud_reference_stats = calculate_reference_statistics(self.fraud_training_sample)
            except Exception as e:
                print(f"Error calculating reference statistics: {e}")
        return self.fraud_reference_stats
"""
Script to run all training scripts and store metrics in the database.
This should be run whenever models are retrained to update metrics in the dashboard.

Handles file paths correctly based on project hierarchy:
- Script is in scripts/ (root level)
- Training scripts are in src/training/
- Data files are in data/ or data/raw/
- Models are saved to models/ (root level)
- Database is in data/monitoring.db (root level)
"""
import os
import sys

# Add project root to path - script is in scripts/, so go up one level
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.monitoring.performance_tracker import init_performance_db, save_training_metrics, get_db_path

def train_and_store_fraud():
    """Train fraud model and store metrics."""
    print("\n" + "="*60)
    print("🔹 TRAINING FRAUD DETECTION MODEL")
    print("="*60)
    
    try:
        from src.training.train_fraud import train_challenger
        from src.features.fraud_features import load_fraud_data
        from src.evaluation.metrics import get_fraud_metrics
        import joblib
        
        # Check multiple possible locations for data file
        fraud_data_path = None
        for path in [
            os.path.join(project_root, "data", "creditcard.csv"),
            os.path.join(project_root, "data", "raw", "creditcard.csv"),
            "data/creditcard.csv",
            "data/raw/creditcard.csv"
        ]:
            if os.path.exists(path):
                fraud_data_path = path
                break
        
        if fraud_data_path is None:
            print("❌ Creditcard.csv not found in data/ or data/raw/")
            return False
        
        print(f"📂 Using data file: {fraud_data_path}")
        X_train, X_test, y_train, y_test = load_fraud_data(filepath=fraud_data_path)
        
        # Train challenger (best model)
        print("\nTraining XGBoost Challenger model...")
        train_challenger(X_train, y_train, X_test, y_test)
        
        # Load the trained model and get metrics - use absolute path
        model_path = os.path.join(project_root, "models", "fraud", "xgboost_best.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            probs = model.predict_proba(X_test)[:, 1]
            metrics = get_fraud_metrics(y_test, probs)
            
            # Store metrics - uses consistent path helper
            db = init_performance_db()  # Uses get_db_path() internally
            save_training_metrics(db, "fraud", {
                "pr_auc": metrics['pr_auc'],
                "f1_score": metrics['f1_score'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "best_threshold": metrics['best_threshold']
            })
            db.close()
            print(f"✅ Fraud metrics stored: PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1_score']:.4f}")
            return True
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
    except Exception as e:
        print(f"❌ Error training fraud model: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_and_store_btc():
    """Train BTC model and store metrics."""
    print("\n" + "="*60)
    print("🔹 TRAINING BITCOIN FORECASTING MODEL")
    print("="*60)
    
    try:
        from src.training.train_btc import train_challenger_price
        from src.features.btc_features import process_btc_data
        
        # Load data - process_btc_data handles its own paths
        X, y_price, y_dir = process_btc_data()
        if X is None:
            print("❌ Failed to load BTC data")
            return False
        
        # Time-series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_price_train, y_price_test = y_price.iloc[:split_idx], y_price.iloc[split_idx:]
        y_prev_test = X_test['Lag_1']
        
        # Train challenger - this already returns metrics
        print("\nTraining XGBoost Price model...")
        metrics = train_challenger_price(X_train, y_price_train, X_test, y_price_test, y_prev_test)
        
        # Store metrics - uses consistent path helper
        db = init_performance_db()  # Uses get_db_path() internally
        save_training_metrics(db, "btc_price", {
            "rmse": metrics['rmse'],
            "mae": metrics['mae'],
            "mape_percent": metrics['mape_percent'],
            "directional_accuracy_percent": metrics['directional_accuracy_percent']
        })
        db.close()
        print(f"✅ BTC metrics stored: RMSE={metrics['rmse']:.2f}, Directional Acc={metrics['directional_accuracy_percent']:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Error training BTC model: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_and_store_segmentation():
    """Train segmentation model and store metrics."""
    print("\n" + "="*60)
    print("🔹 TRAINING SEGMENTATION MODEL")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from src.evaluation.metrics import get_segmentation_metrics
        
        # Generate data (same as train_segmentation.py)
        np.random.seed(42)
        data = {
            'A': np.concatenate([np.random.normal(0, 1, 500), np.random.normal(10, 1, 500)]),
            'B': np.concatenate([np.random.normal(0, 1, 500), np.random.normal(10, 1, 500)])
        }
        df = pd.DataFrame(data)
        X_scaled = StandardScaler().fit_transform(df)
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        
        # Train model
        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)
        
        # Get metrics
        metrics = get_segmentation_metrics(X_pca, labels)
        
        # Save model - use absolute paths
        models_dir = os.path.join(project_root, "models", "segmentation")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "best_clustering.pkl")
        joblib.dump(model, model_path)
        
        # Store metrics - uses consistent path helper
        db = init_performance_db()  # Uses get_db_path() internally
        save_training_metrics(db, "segmentation", {
            "silhouette_score": metrics['silhouette_score'],
            "davies_bouldin": metrics['davies_bouldin']
        })
        db.close()
        print(f"✅ Segmentation metrics stored: Silhouette={metrics['silhouette_score']:.4f}, DB={metrics['davies_bouldin']:.4f}")
        return True
    except Exception as e:
        print(f"❌ Error training segmentation model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all training scripts and store metrics."""
    from datetime import datetime
    
    print("\n" + "="*60)
    print("🚀 CRYPTOGUARD MODEL TRAINING & METRICS STORAGE")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Start time: {datetime.now()}\n")
    
    results = {}
    
    # Train all models
    results['fraud'] = train_and_store_fraud()
    results['btc_price'] = train_and_store_btc()
    results['segmentation'] = train_and_store_segmentation()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model:20s}: {status}")
    
    all_success = all(results.values())
    if all_success:
        print("\n✅ All models trained successfully! Metrics stored in database.")
        print("   You can now view training metrics in the Model Performance dashboard.")
    else:
        print("\n⚠️ Some models failed to train. Check errors above.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    exit(main())

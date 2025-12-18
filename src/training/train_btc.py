import mlflow
import mlflow.sklearn
import optuna
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from src.features.btc_features import process_btc_data
from src.evaluation.metrics import get_btc_metrics

# ==========================================
# 🛠️ FIXED: PATH LOGIC FOR RAILWAY
# ==========================================
# Get the absolute path of the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Move up to the project root (from src/training/ to root)
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
# Define the absolute path for the models folder
model_base_path = os.path.join(project_root, "models", "btc")
os.makedirs(model_base_path, exist_ok=True)
# ==========================================

def train_regression_baseline(X_train, y_train, X_test, y_test, y_prev_test):
    """
    Baseline: Simple Linear Regression for Price.
    """
    print("\n🔹 Training Regression Baseline...")
    with mlflow.start_run(run_name="BTC_Baseline_Linear"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = get_btc_metrics(y_test, preds, y_prev_test)
        
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   Directional Acc: {metrics['directional_accuracy_percent']:.2f}%")
        return metrics['rmse']

def optimize_xgboost_price(trial, X_train, y_train, X_test, y_test, y_prev_test):
    """
    Optuna Objective: Minimize RMSE for Price Prediction.
    """
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = XGBRegressor(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

def train_challenger_price(X_train, y_train, X_test, y_test, y_prev_test):
    """
    Tunes XGBoost Regressor for Price and SAVES REAL BINARY.
    """
    print("\n🔸 Tuning Price Model (XGBoost Regressor)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: optimize_xgboost_price(t, X_train, y_train, X_test, y_test, y_prev_test), n_trials=10)
    
    with mlflow.start_run(run_name="BTC_Challenger_Price"):
        best_model = XGBRegressor(**study.best_params)
        best_model.fit(X_train, y_train)
        
        preds = best_model.predict(X_test)
        metrics = get_btc_metrics(y_test, preds, y_prev_test)
        
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        # FIXED: Use absolute path for saving
        save_path = os.path.join(model_base_path, "xgboost_price.pkl")
        joblib.dump(best_model, save_path)
        print(f"✅ Price Model Saved: {save_path}")
        return metrics

def train_challenger_direction(X_train, y_train_dir, X_test, y_test_dir):
    """
    Tunes XGBoost Classifier for Direction (Up/Down).
    """
    print("\n🔸 Tuning Direction Model (XGBoost Classifier)...")
    
    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2)
        }
        model = XGBClassifier(**param)
        model.fit(X_train, y_train_dir)
        preds = model.predict(X_test)
        return 1.0 - np.mean(preds == y_test_dir)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    with mlflow.start_run(run_name="BTC_Challenger_Direction"):
        best_model = XGBClassifier(**study.best_params)
        best_model.fit(X_train, y_train_dir)
        
        acc = best_model.score(X_test, y_test_dir)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(best_model, "model")
        
        # FIXED: Use absolute path for saving
        save_path = os.path.join(model_base_path, "xgboost_direction.pkl")
        joblib.dump(best_model, save_path)
        print(f"✅ Direction Model Saved: {save_path}")
        print(f"   Best Directional Acc: {acc*100:.2f}%")

if __name__ == "__main__":
    mlflow.set_experiment("BTC_Forecasting")
    
    # 1. Load Data
    X, y_price, y_dir = process_btc_data()
    
    if X is None:
        print("❌ Failed to load data.")
        exit(1)

    # 2. Time-Series Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_price_train, y_price_test = y_price.iloc[:split_idx], y_price.iloc[split_idx:]
    y_dir_train, y_dir_test = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
    
    y_prev_test = X_test['Lag_1']
    
    # 3. Run Experiments
    train_regression_baseline(X_train, y_price_train, X_test, y_price_test, y_prev_test)
    price_metrics = train_challenger_price(X_train, y_price_train, X_test, y_price_test, y_prev_test)
    train_challenger_direction(X_train, y_dir_train, X_test, y_dir_test)
    
    print("\n✅ BTC Training Complete.")
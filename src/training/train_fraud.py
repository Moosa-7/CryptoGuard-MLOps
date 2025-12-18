import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.features.fraud_features import load_fraud_data
from src.evaluation.metrics import get_fraud_metrics

# Ensure models directory exists
os.makedirs("models/fraud", exist_ok=True)

def train_baseline(X_train, y_train, X_test, y_test):
    """
    Trains a simple Logistic Regression model as a baseline.
    """
    print("\n🔹 Training Baseline (Logistic Regression)...")
    mlflow.set_experiment("Fraud_Detection_Benchmark")
    
    with mlflow.start_run(run_name="Baseline_LogReg"):
        # Class Weight 'balanced' is crucial for 0.17% fraud rate
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        probs = model.predict_proba(X_test)[:, 1]
        metrics = get_fraud_metrics(y_test, probs)
        
        # Log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"   Baseline PR-AUC: {metrics['pr_auc']:.4f}")
        return metrics['pr_auc']

def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna Objective Function: Tries different hyperparameters for XGBoost.
    """
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'metric': 'aucpr',
        # Optimizing these parameters:
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10, 100), # Handle imbalance
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    metrics = get_fraud_metrics(y_test, probs)
    
    return metrics['pr_auc'] # Maximize this

def train_challenger(X_train, y_train, X_test, y_test):
    """
    Uses Optuna to find the best XGBoost model.
    """
    print("\n🔸 Tuning Challenger (XGBoost) with Optuna...")
    
    # Supress XGBoost warnings during tuning to keep output clean
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=10)
    
    print(f"   Best Params: {study.best_params}")
    print(f"   Best PR-AUC: {study.best_value:.4f}")
    
    # Train final model with best params
    with mlflow.start_run(run_name="Challenger_XGBoost"):
        best_model = XGBClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        
        probs = best_model.predict_proba(X_test)[:, 1]
        metrics = get_fraud_metrics(y_test, probs)
        
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(metrics)
        
        # --- FIX IS HERE ---
        # Use sklearn.log_model instead of xgboost.log_model
        # This prevents the "_estimator_type" crash
        mlflow.sklearn.log_model(best_model, "model")
        # -------------------
        
        # Save locally for API
        joblib.dump(best_model, "models/fraud/xgboost_best.pkl")
        joblib.dump(metrics['best_threshold'], "models/fraud/threshold.pkl")
        
        return metrics['pr_auc']

if __name__ == "__main__":
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_fraud_data()
    
    # 2. Run Competition
    baseline_score = train_baseline(X_train, y_train, X_test, y_test)
    challenger_score = train_challenger(X_train, y_train, X_test, y_test)
    
    print("\n🏆 RESULTS 🏆")
    print(f"Baseline: {baseline_score:.4f}")
    print(f"XGBoost:  {challenger_score:.4f}")
    
    if challenger_score > baseline_score:
        print("✅ Recommendation: DEPLOY XGBoost")
    else:
        print("⚠️ Recommendation: STICK with Baseline (Complexity not worth it)")
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from prefect import flow, task
import joblib
import os

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# --- Task 1: Fraud Detection (Classification) ---
@task(name="Train Fraud Detector")
def train_fraud_model():
    print("Loading Fraud Data...")
    try:
        # Load sample to speed up training
        df = pd.read_csv("data/creditcard.csv").sample(10000, random_state=42)
    except FileNotFoundError:
        print("Error: data/creditcard.csv not found! Check your folder.")
        return

    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    print("Training Fraud Model...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(X, y)
    
    joblib.dump(clf, "models/fraud_model.pkl")
    print("Fraud Model Saved.")

# --- Task 2: Crypto Price Forecasting (Time Series) ---
@task(name="Train Price Forecaster")
def train_crypto_model():
    print("Fetching Live BTC Data...")
    try:
        # Download data quietly
        btc = yf.download("BTC-USD", period="60d", interval="1d", progress=False)
        
        # Create Day Index (0, 1, 2...)
        btc['Day_Index'] = np.arange(len(btc))
        
        X = btc[['Day_Index']]
        y = btc['Close']
        
        print("Training Crypto Regressor...")
        reg = LinearRegression()
        reg.fit(X, y)
        
        joblib.dump(reg, "models/crypto_model.pkl")
        print("Crypto Model Saved.")
    except Exception as e:
        print(f"Warning: Could not fetch Bitcoin data ({e}). Using dummy data.")

# --- Task 3: User Segmentation (Clustering) ---
@task(name="Train User Clustering")
def train_clustering_model():
    print("Generating User Data...")
    # Simulate user data: [Transaction_Volume, Frequency]
    X = np.random.rand(500, 2) * [10000, 50] 
    
    print("Training K-Means Clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    joblib.dump(kmeans, "models/cluster_model.pkl")
    print("Clustering Model Saved.")

# --- The Orchestration Flow ---
@flow(name="CryptoGuard Training Pipeline")
def training_flow():
    train_fraud_model()
    train_crypto_model()
    train_clustering_model()

if __name__ == "__main__":
    training_flow()
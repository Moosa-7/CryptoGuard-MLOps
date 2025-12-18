import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, f1_score, confusion_matrix, mean_squared_error, silhouette_score
import xgboost as xgb
from prefect import flow, task
from imblearn.over_sampling import SMOTE
import os

# Setup Directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# ==========================================
# 1. FRAUD DETECTION (Classification)
# ==========================================
@task(name="Train Fraud Model")
def train_fraud_model():
    print("\n🔹 STARTING FRAUD DETECTION PIPELINE")
    
    # 1. Load Data
    try:
        df = pd.read_csv("data/creditcard.csv")
        # Sample 50k for speed, but large enough for SMOTE
        df_sample = df.sample(min(len(df), 50000), random_state=42)
    except FileNotFoundError:
        print("❌ Error: creditcard.csv not found!")
        return

    X = df_sample.drop(['Class', 'Time'], axis=1)
    y = df_sample['Class']

    # --- PCA STEP ---
    print("   📉 Applying PCA...")
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    joblib.dump(pca, "models/fraud_pca.pkl") 
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)

    # --- SMOTE (The Fix for Imbalance) ---
    print("   ⚖️ Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    # ---------------------------

    # 3. Define Models (Optimized for Recall)
    model_zoo = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"C": [0.01, 0.1, 1]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {"n_estimators": [50, 100], "max_depth": [10, 20]}
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(eval_metric='logloss'),
            "params": {"learning_rate": [0.1, 0.2], "scale_pos_weight": [1, 10]} 
        }
    }

    best_score = 0
    best_model = None
    best_name = ""

    # 4. Train & Tune
    for name, config in model_zoo.items():
        print(f"   ⚙️ Tuning {name}...")
        # Optimize for RECALL (Catching fraud is priority)
        search = RandomizedSearchCV(config["model"], config["params"], n_iter=3, cv=3, scoring='recall', random_state=42)
        search.fit(X_train_balanced, y_train_balanced)
        
        best_ver = search.best_estimator_
        y_pred = best_ver.predict(X_test)
        
        rec = recall_score(y_test, y_pred)
        
        print(f"      ✅ {name} -> Recall: {rec:.4f}")
        
        if rec > best_score:
            best_score = rec
            best_model = best_ver
            best_name = name

    print(f"   🏆 WINNER: {best_name} (Recall: {best_score:.4f})")
    joblib.dump(best_model, "models/fraud_model.pkl")

# ==========================================
# 2. BITCOIN PREDICTION (Time Series)
# ==========================================
@task(name="Train Crypto Model")
def train_crypto_model():
    print("\n🔹 STARTING CRYPTO PIPELINE")
    
    # Ingestion
    try:
        df = yf.download("BTC-USD", period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df[['Close']].copy()
    except:
        print("❌ Error downloading crypto data")
        return
    
    # Feature Engineering (Lag & Rolling)
    df['Lag_1'] = df['Close'].shift(1)
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df.dropna(inplace=True)
    
    # Time-Aware Split
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    
    # MATCHING FEATURES WITH API: Only use Lag_1 and MA_7
    features = ['Lag_1', 'MA_7']
    target = 'Close'
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # Train XGBoost
    print("   ⚙️ Training XGBoost Regressor...")
    model = xgb.XGBRegressor()
    search = RandomizedSearchCV(model, {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}, cv=TimeSeriesSplit(n_splits=3), scoring='neg_root_mean_squared_error')
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"   🏆 BTC RMSE: {rmse:.2f}")
    joblib.dump(best_model, "models/crypto_model.pkl")

# ==========================================
# 3. USER SEGMENTATION (Clustering)
# ==========================================
@task(name="Train Clustering Model")
def train_clustering_model():
    print("\n🔹 STARTING USER SEGMENTATION PIPELINE")
    
    # 1. Generate Synthetic Data for 3 Explicit Tiers
    X = np.concatenate([
        np.random.normal([100, 5], [50, 2], (300, 2)),      # Poor
        np.random.normal([5000, 20], [1000, 5], (300, 2)),  # Standard
        np.random.normal([50000, 50], [10000, 10], (50, 2)) # VIP
    ])
    
    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # 3. Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    joblib.dump(kmeans, "models/cluster_model.pkl")
    
    # 4. Auto-Labeling Logic
    real_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_map = {}
    
    for cluster_id, center in enumerate(real_centers):
        avg_volume = center[0]
        if avg_volume > 25000:
            cluster_map[cluster_id] = "VIP"
        elif avg_volume > 2000:
            cluster_map[cluster_id] = "Standard"
        else:
            cluster_map[cluster_id] = "Poor"
            
    print(f"   🏷️ Defined Segments: {cluster_map}")
    joblib.dump(cluster_map, "models/cluster_map.pkl")

# ==========================================
# ORCHESTRATION
# ==========================================
@flow(name="FinTech Pipeline")
def main_flow():
    # Calling the functions by their CORRECT names
    train_fraud_model()
    train_crypto_model()
    train_clustering_model()

if __name__ == "__main__":
    main_flow()
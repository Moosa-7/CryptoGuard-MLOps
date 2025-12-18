import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_and_process_users(n_samples=1000):
    """
    Generates synthetic user transaction data and applies PCA.
    Returns: X_pca (2D array for clustering), df (Original data for interpretation)
    """
    np.random.seed(42)
    
    # 1. Generate Synthetic Data (Whales vs Minnows vs Churners)
    data = {
        'tx_count': np.concatenate([
            np.random.normal(5, 2, 500),    # Casuals
            np.random.normal(50, 10, 300),  # Regulars
            np.random.normal(500, 50, 200)  # Whales
        ]),
        'avg_amt': np.concatenate([
            np.random.normal(10, 5, 500),
            np.random.normal(100, 20, 300),
            np.random.normal(1000, 200, 200)
        ]),
        'failed_tx_ratio': np.random.beta(2, 10, n_samples) # Skewed low
    }
    
    df = pd.DataFrame(data)
    
    # 2. Scaling (Critical for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # 3. PCA Reduction (Reduce to 2D for Visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"✅ User Data Generated & Reduced: {X_pca.shape}")
    return X_pca, df  # Return original df for interpretation later
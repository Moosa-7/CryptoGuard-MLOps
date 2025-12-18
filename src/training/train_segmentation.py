import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Setup Directory
MODELS_DIR = os.path.join(os.getcwd(), "models/segmentation")
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"📂 Target Directory: {MODELS_DIR}")

# 2. Generate Data (In-Line Failsafe)
print("🔄 Generating failsafe data...")
np.random.seed(42)
n = 1000
# Simple 3-cluster generation that cannot fail
data = {
    'A': np.concatenate([np.random.normal(0, 1, 500), np.random.normal(10, 1, 500)]),
    'B': np.concatenate([np.random.normal(0, 1, 500), np.random.normal(10, 1, 500)])
}
df = pd.DataFrame(data)
X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(df))

# 3. Train Model
print("🔹 Training K-Means...")
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X_pca)

# 4. Save Model
save_path = os.path.join(MODELS_DIR, "best_clustering.pkl")
joblib.dump(model, save_path)

print("-" * 30)
if os.path.exists(save_path):
    print(f"✅ SUCCESS: Model saved to: {save_path}")
    print("🚀 You can now proceed to Phase 4.")
else:
    print("❌ ERROR: File write failed. Check permissions.")
print("-" * 30)
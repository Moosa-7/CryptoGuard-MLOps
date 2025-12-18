import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# 🛠️ FIXED: ROBUST PATH LOGIC FOR RAILWAY
# ==========================================
# Get the absolute path of the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up to the project root (from src/training/ to root)
# This ensures we save to /app/models/ and not a subfolder
project_root = os.path.abspath(os.path.join(script_dir, "../../"))

# Define the absolute path for the models folder
MODELS_DIR = os.path.join(project_root, "models", "segmentation")
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"📂 Resolved Target Directory: {MODELS_DIR}")
# ==========================================

# 2. Generate Data (In-Line Failsafe)
print("🔄 Generating failsafe data...")
np.random.seed(42)
n = 1000
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
# FIXED: Using absolute path join
save_path = os.path.join(MODELS_DIR, "best_clustering.pkl")

# Use a context manager to ensure the file is closed and flushed to disk
with open(save_path, "wb") as f:
    joblib.dump(model, f)

print("-" * 30)
if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
    print(f"✅ SUCCESS: Real model binary saved!")
    print(f"📍 Location: {save_path}")
    print(f"📦 Size: {os.path.getsize(save_path) / 1024:.2f} KB")
else:
    print("❌ ERROR: File is missing or too small (1-byte pointer error).")
print("-" * 30)
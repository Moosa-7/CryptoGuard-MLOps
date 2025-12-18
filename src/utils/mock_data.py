import pandas as pd
import numpy as np
import os

def create_mock_fraud_data(path="data/raw/creditcard.csv"):
    """Creates a tiny valid dataset so CI/CD training doesn't crash."""
    print(f"🛠️ Generating mock data at {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 30 columns (Time, V1-V28, Amount) + Class
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    
    # Generate 100 rows
    data = np.random.randn(100, 31)
    df = pd.DataFrame(data, columns=cols)
    
    # Fix binary target
    df['Class'] = np.random.randint(0, 2, 100)
    
    df.to_csv(path, index=False)
    print("✅ Mock data created.")

if __name__ == "__main__":
    create_mock_fraud_data()
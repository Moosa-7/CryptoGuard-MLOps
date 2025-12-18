import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_fraud_data(filepath="data/raw/creditcard.csv", test_size=0.2):
    """
    Loads and preprocesses the Kaggle Fraud Dataset.
    - Scales 'Time' and 'Amount' using RobustScaler.
    - Leaves V1-V28 alone (already PCA'd).
    - Stratifies split to maintain 0.17% fraud ratio in test set.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found: {filepath}. Please download the Kaggle dataset.")

    # 1. Scale Time and Amount (The only non-PCA features)
    # RobustScaler is less prone to outliers than StandardScaler
    scaler = RobustScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    # 2. Split Features & Target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Stratified Split (Crucial for Imbalanced Data)
    # This ensures both Train and Test have exactly 0.17% fraud
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"✅ Fraud Data Loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test
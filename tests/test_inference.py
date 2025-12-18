import pytest
import pandas as pd
import numpy as np
from src.inference.model_loader import ModelLoader

# Fixture to load models once for all tests
@pytest.fixture
def loader():
    return ModelLoader()

def test_loader_initialization(loader):
    """Verify all models are loaded successfully."""
    assert loader.fraud_model is not None, "Fraud model missing"
    assert loader.btc_price_model is not None, "BTC Price model missing"
    assert loader.segmentation_model is not None, "Segmentation model missing"

def test_fraud_prediction(loader):
    """Test Fraud Model with correct input shape (30 columns)."""
    # XGBoost expects 30 features: Time, V1...V28, Amount
    # We must match the training column names exactly if using DataFrames
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    dummy_input = pd.DataFrame(np.random.rand(1, 30), columns=cols)
    
    is_fraud, prob, risk = loader.predict_fraud(dummy_input)
    
    assert isinstance(is_fraud, bool)
    assert 0.0 <= prob <= 1.0
    assert risk in ["Low", "Medium", "High", "Critical"]

def test_btc_prediction(loader):
    """Test BTC Model with correct input shape (7 columns)."""
    # Features: Close, Volume, Lag_1, Returns, MA_7, MA_30, Volatility
    cols = ['Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility']
    dummy_input = pd.DataFrame(np.random.rand(1, 7), columns=cols)
    
    price, signal = loader.predict_btc(dummy_input)
    
    assert isinstance(price, float)
    assert signal in ["Buy", "Sell", "Hold"]

def test_segmentation_prediction(loader):
    """Test Segmentation Model with correct input shape (2 columns)."""
    # PCA reduced data has 2 columns
    dummy_input = pd.DataFrame(np.random.rand(1, 2), columns=['PC1', 'PC2'])
    
    cluster_id = loader.get_segment(dummy_input)
    
    assert isinstance(cluster_id, int)
    assert cluster_id >= 0
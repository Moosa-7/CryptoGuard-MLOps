from pydantic import BaseModel, Field, validator
from typing import List

# 1. Fraud Schema
class FraudInput(BaseModel):
    # We expect exactly 30 features: [Time, V1, V2, ..., V28, Amount]
    features: List[float]

    @validator('features')
    def check_length(cls, v):
        if len(v) != 30:
            raise ValueError(f'Fraud model expects exactly 30 features (Time + V1-V28 + Amount). Got {len(v)}.')
        return v

# 2. Bitcoin Schema
class BTCInput(BaseModel):
    # These match the columns: ['Close', 'Volume', 'Lag_1', 'Returns', 'MA_7', 'MA_30', 'Volatility']
    close: float
    volume: float
    lag_1: float
    daily_return: float
    ma_7: float
    ma_30: float
    volatility: float

    def to_df_list(self):
        """Helper to convert to list for the model"""
        return [[
            self.close, self.volume, self.lag_1, 
            self.daily_return, self.ma_7, self.ma_30, self.volatility
        ]]

# 3. Segmentation Schema
class SegmentationInput(BaseModel):
    # Expects the 2 Principal Components (PCA features)
    pc1: float
    pc2: float
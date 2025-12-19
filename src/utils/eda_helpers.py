import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_sample_data(dataset_name: str, sample_size: int = 5000) -> Optional[pd.DataFrame]:
    """
    Load and sample datasets for EDA.
    
    Args:
        dataset_name: Name of dataset ('fraud', 'btc')
        sample_size: Number of samples to load
    
    Returns:
        DataFrame or None if file not found
    """
    try:
        if dataset_name.lower() == 'fraud':
            # Try multiple possible paths
            possible_paths = [
                os.path.join(os.getcwd(), "data", "creditcard.csv"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "creditcard.csv"),  # Project root
                "data/creditcard.csv",
                os.path.join("data", "creditcard.csv")
            ]
            
            filepath = None
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath and os.path.exists(filepath):
                df = pd.read_csv(filepath)
                # Sample for performance
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                return df
            else:
                # File not found - generate mock data for deployment
                print(f"⚠️ {filepath} not found, generating mock fraud data for EDA...")
                return _generate_mock_fraud_data(sample_size)
        elif dataset_name.lower() in ['btc', 'bitcoin']:
            # Try multiple possible paths
            possible_paths = [
                os.path.join(os.getcwd(), "data", "raw", "btc_usd.parquet"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "btc_usd.parquet"),  # Project root
                "data/raw/btc_usd.parquet",
                os.path.join("data", "raw", "btc_usd.parquet")
            ]
            
            filepath = None
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath and os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                # Sample for performance
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                return df
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return None

def calculate_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with summary statistics
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
        }
        
        # Add target-specific stats if Class column exists
        if 'Class' in df.columns:
            fraud_count = df['Class'].sum()
            summary['fraud_count'] = int(fraud_count)
            summary['fraud_rate'] = float(fraud_count / len(df))
            summary['legitimate_count'] = int(len(df) - fraud_count)
        
        return summary
    except Exception as e:
        print(f"Error calculating summary stats: {e}")
        return {}

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in a column using specified method.
    
    Args:
        df: DataFrame
        column: Column name to analyze
        method: Method to use ('iqr' for Interquartile Range)
    
    Returns:
        DataFrame with outlier information
    """
    try:
        if column not in df.columns:
            return pd.DataFrame()
        
        values = df[column].dropna()
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            return pd.DataFrame({
                'is_outlier': df[column].apply(lambda x: x < lower_bound or x > upper_bound),
                'value': df[column],
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error detecting outliers: {e}")
        return pd.DataFrame()


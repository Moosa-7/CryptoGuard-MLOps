import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional

def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Dict]:
    """
    Detect data drift using Kolmogorov-Smirnov test for numerical features.
    
    Args:
        reference_data: Reference training data (DataFrame)
        current_data: Current/prediction data (DataFrame, can be single row)
        threshold: P-value threshold for drift detection (default 0.05)
    
    Returns:
        Dictionary with drift results per feature
    """
    drift_results = {}
    
    # Only check numerical features that exist in both dataframes
    numeric_cols_ref = reference_data.select_dtypes(include=[np.number]).columns
    numeric_cols_curr = current_data.select_dtypes(include=[np.number]).columns
    # Find intersection of columns - convert to list to avoid Index type issues
    numeric_cols = [col for col in numeric_cols_ref if col in numeric_cols_curr]
    
    if len(numeric_cols) == 0:
        # Return empty results with error message
        return {
            '_error': f'No matching numeric columns found. Reference: {list(numeric_cols_ref)[:5]}..., Current: {list(numeric_cols_curr)[:5]}...'
        }
    
    for col in numeric_cols:
            try:
                ref_values = reference_data[col].dropna()
                current_values = current_data[col].dropna()
                
                if len(ref_values) < 2:
                    # Not enough reference data
                    drift_results[col] = {
                        'p_value': None,
                        'drift_detected': False,
                        'drift_score': 0.0,
                        'statistic': 0.0,
                        'error': 'Insufficient reference data'
                    }
                elif len(current_values) >= 2:
                    # Multiple current values - use KS test
                    stat, p_value = stats.ks_2samp(ref_values, current_values)
                    drift_results[col] = {
                        'p_value': float(p_value),
                        'drift_detected': bool(p_value < threshold),
                        'drift_score': float(stat),
                        'statistic': float(stat)
                    }
                else:
                    # Single current value - use z-score/percentile method
                    current_val = float(current_values.iloc[0])
                    ref_mean = float(ref_values.mean())
                    ref_std = float(ref_values.std())
                    
                    if ref_std > 0:
                        # Calculate z-score
                        z_score = abs((current_val - ref_mean) / ref_std)
                        # Calculate percentile
                        percentile = stats.percentileofscore(ref_values, current_val) / 100.0
                        
                        # Drift if value is more than 3 standard deviations away
                        # or in extreme percentiles (< 1% or > 99%)
                        drift_detected = bool(z_score > 3.0 or percentile < 0.01 or percentile > 0.99)
                        drift_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1
                        
                        drift_results[col] = {
                            'p_value': None,  # Can't calculate p-value for single point
                            'drift_detected': drift_detected,
                            'drift_score': float(drift_score),
                            'statistic': float(z_score),
                            'z_score': float(z_score),
                            'percentile': float(percentile),
                            'current_value': float(current_val),
                            'reference_mean': float(ref_mean),
                            'reference_std': float(ref_std)
                        }
                    else:
                        # Zero variance in reference - compare directly
                        drift_detected = bool(abs(current_val - ref_mean) > 1e-6)
                        drift_results[col] = {
                            'p_value': None,
                            'drift_detected': drift_detected,
                            'drift_score': 1.0 if drift_detected else 0.0,
                            'statistic': abs(current_val - ref_mean),
                            'current_value': float(current_val),
                            'reference_mean': float(ref_mean)
                        }
            except Exception as e:
                # If test fails, mark as no drift
                drift_results[col] = {
                    'p_value': None,
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'statistic': 0.0,
                    'error': str(e)
                }
    
    # Convert all numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    return convert_to_native(drift_results)

def calculate_reference_statistics(training_data: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate reference statistics for training data.
    
    Args:
        training_data: Training DataFrame
    
    Returns:
        Dictionary with statistics per feature
    """
    stats_dict = {}
    
    numeric_cols = training_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        try:
            values = training_data[col].dropna()
            stats_dict[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75))
            }
        except Exception as e:
            stats_dict[col] = {'error': str(e)}
    
    return stats_dict


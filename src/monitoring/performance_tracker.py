import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

def get_project_root():
    """
    Get the project root directory.
    Works regardless of where the code is called from (scripts/, src/, etc.)
    """
    # Get the directory of this file (src/monitoring/performance_tracker.py)
    current_file = os.path.abspath(__file__)
    # Go up to src/, then to root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root

def get_db_path():
    """
    Get the absolute path to the monitoring database.
    Database is always at: project_root/data/monitoring.db
    """
    project_root = get_project_root()
    return os.path.join(project_root, "data", "monitoring.db")

def init_performance_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Initialize SQLite database for performance tracking.
    
    Args:
        db_path: Optional path to SQLite database file. If None, uses default location.
    
    Returns:
        Database connection
    """
    # Use default path if not provided
    if db_path is None:
        db_path = get_db_path()
    else:
        # Convert relative paths to absolute based on project root
        if not os.path.isabs(db_path):
            project_root = get_project_root()
            db_path = os.path.join(project_root, db_path)
    
    # Create data directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            prediction REAL,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_model_timestamp 
        ON predictions(model_name, timestamp)
    ''')
    
    # Create training_metrics table to store model training results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            training_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, metric_name, training_date)
        )
    ''')
    
    # Create index for training metrics queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_model_metric 
        ON training_metrics(model_name, metric_name)
    ''')
    
    conn.commit()
    return conn

def log_prediction(conn: sqlite3.Connection, model_name: str, prediction: Any, confidence: float, timestamp: Optional[datetime] = None):
    """
    Log a prediction to the database.
    
    Args:
        conn: Database connection
        model_name: Name of the model (e.g., 'fraud', 'btc_price')
        prediction: Prediction value (can be bool, float, int, str)
        confidence: Confidence/probability score
        timestamp: Optional timestamp (defaults to now)
    """
    try:
        # Convert prediction to a storable format
        if isinstance(prediction, bool):
            prediction_val = 1.0 if prediction else 0.0
        elif isinstance(prediction, (int, float)):
            prediction_val = float(prediction)
        else:
            prediction_val = None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (model_name, prediction, confidence, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (model_name, prediction_val, confidence, timestamp))
        conn.commit()
    except Exception as e:
        print(f"Error logging prediction: {e}")

def get_performance_metrics(conn: sqlite3.Connection, model_name: str, time_window: str = "24h") -> Dict[str, Any]:
    """
    Get aggregated performance metrics for a model.
    
    Args:
        conn: Database connection
        model_name: Name of the model
        time_window: Time window ('24h', '7d', '30d', 'all')
    
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Calculate cutoff time
        now = datetime.now()
        if time_window == "24h":
            cutoff = now - timedelta(hours=24)
        elif time_window == "7d":
            cutoff = now - timedelta(days=7)
        elif time_window == "30d":
            cutoff = now - timedelta(days=30)
        else:  # 'all'
            cutoff = datetime.min
        
        # Query metrics
        query = '''
            SELECT 
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN prediction >= 0.5 THEN 1 ELSE 0 END) as positive_predictions
            FROM predictions
            WHERE model_name = ? AND timestamp >= ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(model_name, cutoff))
        
        if len(df) > 0 and df['total_predictions'].iloc[0] > 0:
            total = int(df['total_predictions'].iloc[0])
            avg_conf = float(df['avg_confidence'].iloc[0]) if df['avg_confidence'].iloc[0] is not None else 0.0
            positives = int(df['positive_predictions'].iloc[0]) if df['positive_predictions'].iloc[0] is not None else 0
        else:
            total = 0
            avg_conf = 0.0
            positives = 0
        
        return {
            'total_predictions': total,
            'avg_confidence': avg_conf,
            'positive_predictions': positives,
            'time_window': time_window
        }
    except Exception as e:
        print(f"Error getting performance metrics: {e}")
        return {
            'total_predictions': 0,
            'avg_confidence': 0.0,
            'positive_predictions': 0,
            'time_window': time_window
        }

def get_prediction_history(conn: sqlite3.Connection, model_name: str, limit: int = 100) -> List[Dict]:
    """
    Get recent prediction history for visualization.
    
    Args:
        conn: Database connection
        model_name: Name of the model
        limit: Maximum number of records to return
    
    Returns:
        List of dictionaries with prediction data
    """
    try:
        query = '''
            SELECT timestamp, prediction, confidence
            FROM predictions
            WHERE model_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(model_name, limit))
        
        # Convert to list of dicts with proper formatting
        history = []
        for _, row in df.iterrows():
            history.append({
                'timestamp': row['timestamp'],
                'prediction': row['prediction'],
                'confidence': row['confidence']
            })
        
        return history
    except Exception as e:
        print(f"Error getting prediction history: {e}")
        return []

def save_training_metrics(conn: sqlite3.Connection, model_name: str, metrics: Dict[str, float], training_date: Optional[datetime] = None):
    """
    Save training metrics to the database.
    
    Args:
        conn: Database connection
        model_name: Name of the model (e.g., 'fraud', 'btc_price', 'segmentation')
        metrics: Dictionary of metric names and values (e.g., {'accuracy': 0.95, 'rmse': 0.5})
        training_date: Optional timestamp (defaults to now)
    """
    if training_date is None:
        training_date = datetime.now()
    
    cursor = conn.cursor()
    for metric_name, metric_value in metrics.items():
        try:
            # Use INSERT OR REPLACE to update if metric already exists for same training date
            cursor.execute('''
                INSERT OR REPLACE INTO training_metrics (model_name, metric_name, metric_value, training_date)
                VALUES (?, ?, ?, ?)
            ''', (model_name, metric_name, float(metric_value), training_date))
        except Exception as e:
            print(f"Error saving training metric {metric_name} for {model_name}: {e}")
    
    conn.commit()

def get_training_metrics(conn: sqlite3.Connection, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get training metrics for a specific model or all models.
    
    Args:
        conn: Database connection
        model_name: Optional model name to filter by. If None, returns all models.
    
    Returns:
        Dictionary of model_name -> {metric_name: value, ...}
    """
    try:
        if model_name:
            query = '''
                SELECT model_name, metric_name, metric_value, training_date
                FROM training_metrics
                WHERE model_name = ?
                ORDER BY training_date DESC
            '''
            df = pd.read_sql_query(query, conn, params=(model_name,))
        else:
            query = '''
                SELECT model_name, metric_name, metric_value, training_date
                FROM training_metrics
                ORDER BY model_name, training_date DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return {}
        
        # Group by model_name and get latest metrics
        result = {}
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            # Get the most recent training date for this model
            latest_date = model_df['training_date'].max()
            latest_metrics = model_df[model_df['training_date'] == latest_date]
            
            result[model] = {
                row['metric_name']: row['metric_value'] 
                for _, row in latest_metrics.iterrows()
            }
            result[model]['last_trained'] = latest_date
        
        return result
    except Exception as e:
        print(f"Error getting training metrics: {e}")
        return {}

def get_all_models_metrics(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get performance metrics for all models.
    
    Args:
        conn: Database connection
    
    Returns:
        Dictionary with metrics per model
    """
    try:
        query = '''
            SELECT 
                model_name,
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                MAX(timestamp) as last_prediction
            FROM predictions
            GROUP BY model_name
        '''
        
        df = pd.read_sql_query(query, conn)
        
        result = {}
        for _, row in df.iterrows():
            result[row['model_name']] = {
                'total_predictions': int(row['total_predictions']),
                'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] is not None else 0.0,
                'last_prediction': row['last_prediction']
            }
        
        return result
    except Exception as e:
        print(f"Error getting all models metrics: {e}")
        return {}


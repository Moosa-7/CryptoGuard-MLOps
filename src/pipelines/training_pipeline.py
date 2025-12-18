import subprocess
import sys
import os
from prefect import flow, task

# Define the python executable
PYTHON_EXEC = sys.executable

def get_env():
    """
    Creates an environment where Python forces UTF-8 encoding.
    Crucial for Windows to handle emojis (✅) in logs.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env

@task(name="Train Fraud Model", retries=3, retry_delay_seconds=60)
def train_fraud_task():
    print("🚀 Starting Fraud Model Training...")
    result = subprocess.run(
        [PYTHON_EXEC, "-m", "src.training.train_fraud"],
        capture_output=True,
        text=True,
        encoding='utf-8',  # Force read as UTF-8
        env=get_env()      # Force write as UTF-8
    )
    if result.returncode != 0:
        print(f"❌ Fraud Training Failed:\n{result.stderr}")
        raise Exception("Fraud Training Script Failed")
    
    print("✅ Fraud Training Success!")
    print(result.stdout)

@task(name="Train BTC Model", retries=3, retry_delay_seconds=60)
def train_btc_task():
    print("🚀 Starting BTC Forecasting Training...")
    result = subprocess.run(
        [PYTHON_EXEC, "-m", "src.training.train_btc"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=get_env()
    )
    if result.returncode != 0:
        print(f"❌ BTC Training Failed:\n{result.stderr}")
        raise Exception("BTC Training Script Failed")
        
    print("✅ BTC Training Success!")
    print(result.stdout)

@task(name="Train Segmentation", retries=2, retry_delay_seconds=30)
def train_segmentation_task():
    print("🚀 Starting User Segmentation...")
    result = subprocess.run(
        [PYTHON_EXEC, "-m", "src.training.train_segmentation"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=get_env()
    )
    if result.returncode != 0:
        print(f"❌ Segmentation Failed:\n{result.stderr}")
        raise Exception("Segmentation Script Failed")
        
    print("✅ Segmentation Success!")
    print(result.stdout)

@flow(name="CryptoGuard Retraining Cycle")
def main_pipeline():
    """
    Orchestrates the full ML lifecycle.
    """
    print("🕒 Starting Scheduled Retraining Cycle...")
    
    # 1. Fraud
    train_fraud_task()
    
    # 2. Bitcoin
    train_btc_task()
    
    # 3. Segmentation
    train_segmentation_task()
    
    print("🏁 Cycle Complete. All models updated.")

if __name__ == "__main__":
    main_pipeline()
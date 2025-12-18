from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
import pandas as pd
import joblib
import os

def validate_models():
    print("Starting DeepChecks Validation...")
    
    # 1. Load Data (same sample used for training)
    try:
        df = pd.read_csv("data/creditcard.csv").sample(5000, random_state=42)
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    # 2. Load the Trained Fraud Model
    try:
        model = joblib.load("models/fraud_model.pkl")
    except FileNotFoundError:
        print("Error: Model not found. Run train_pipeline.py first!")
        return
    
    # 3. Create DeepChecks Dataset
    # We tell it that 'Class' is the target (what we are predicting)
    ds = Dataset(df, label='Class', cat_features=[])
    
    # 4. Run the Full Validation Suite
    # This runs 30+ checks (Data Integrity, Drift, Performance)
    suite = full_suite()
    result = suite.run(train_dataset=ds, test_dataset=ds, model=model)
    
    # 5. Save the Report
    # This creates a cool HTML file you can show in your demo video
    if not os.path.exists("reports"):
        os.makedirs("reports")
        
    result.save_as_html("reports/validation_report.html")
    print("Success! Report saved to reports/validation_report.html")

if __name__ == "__main__":
    validate_models()
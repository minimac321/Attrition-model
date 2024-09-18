import os
from pathlib import Path
import joblib
import pandas as pd
import sys
from datetime import datetime

# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))


from utils.evaluation import evaluate_model

# Directories
raw_data_dir = os.path.join(current_location, "raw_data")
os.makedirs(raw_data_dir, exist_ok=True)
transformed_data_dir = os.path.join(current_location, "transformed_data")
os.makedirs(transformed_data_dir, exist_ok=True)
output_data_dir = os.path.join(current_location, "output_data")
os.makedirs(output_data_dir, exist_ok=True)
processed_dir = os.path.join(current_location, "processed_dir")
os.makedirs(processed_dir, exist_ok=True)
models_dir = os.path.join(current_location, "models")
os.makedirs(models_dir, exist_ok=True)
report_dir = os.path.join(current_location, "reports")
os.makedirs(report_dir, exist_ok=True)
transformed_data_dir = os.path.join(current_location, "transformed_data")
os.makedirs(transformed_data_dir, exist_ok=True)


if __name__ == "__main__":
    # Load the test data
    test_fname = os.path.join(processed_dir, "test.csv")
    test_data = pd.read_csv(test_fname)
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    # Load the best model (change the filename to your saved model name)
    best_model_fname = os.path.join(
        models_dir, 
        "RandomForest_best_model_18_09_24.joblib"
    )  # Adjust the filename
    best_model = joblib.load(best_model_fname)
    print("best_model", best_model)
    
    # Make predictions and evaluate on the test set
    eval_metrics_test = evaluate_model(
        model=best_model,
        X=X_test, y=y_test, 
        model_name="XGBoost", 
        dataset_name="test"
    )
    print(f"Test Accuracy: {eval_metrics_test['accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {eval_metrics_test['balanced_accuracy']:.4f}")
    print("Classification Report (Test Data):")
    print(eval_metrics_test["conf_matrix_df"])
    
    # Convert evaluation metrics to DataFrame and display
    eval_metrics_df = pd.DataFrame([eval_metrics_test])
    print(eval_metrics_df)
    
    # Save inference results
    current_date = datetime.now().strftime("%d_%m_%y")
    inference_csv_fname = os.path.join(output_data_dir, f"inference_test_results_{best_model_fname.split('/')[-1].split('.')[0]}_{current_date}.csv")
    eval_metrics_test["inference_df"].to_csv(inference_csv_fname, index=False)
    print(f"Saved results too: {inference_csv_fname}")
    
    # Save evaluation metrics to a CSV
    current_date = datetime.now().strftime("%d_%m_%y")
    eval_metrics_csv = os.path.join(output_data_dir, f"eval_metrics_test_{best_model_fname.split('/')[-1].split('.')[0]}_{current_date}.csv")
    eval_metrics_df.to_csv(eval_metrics_csv, index=False)
    print(f"Evaluation metrics saved to {eval_metrics_csv}")

from datetime import datetime
import os
from pathlib import Path
import sys
import joblib
import pandas as pd


# Add my parent directory to path variables
current_location = Path(os.path.abspath('')).resolve()
print(current_location)
sys.path.append(str(current_location))


from utils.train import split_and_save_data, train_model
from utils.evaluation import evaluate_model




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

random_state_var = 42

# Example usage
if __name__ == "__main__":
    # Load the dataset and drop the 'Attrition' column
    cleaned_fname = os.path.join(transformed_data_dir, "cleaned_scaled_data_17_09_2024.csv")
    modelling_data = pd.read_csv(cleaned_fname)
    modelling_data = modelling_data.drop(columns=["Attrition"], errors="ignore")
    print(f"modelling_data.shape: {modelling_data.shape}")


    data_splits = split_and_save_data(
        input_df=modelling_data,
        target_col="target",
        processed_dir=processed_dir,
        random_state=random_state_var, 
    )
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_valid = data_splits['X_valid']
    y_valid = data_splits['y_valid']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']

    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_valid.shape: {X_valid.shape}")
    print(f"y_valid.shape: {y_valid.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")


    # Train both RandomForest and XGBoost, and find the best model on validation set
    models_to_train = ['RandomForest', 'XGBoost']
    best_model = None
    best_score = -float('inf')
    best_model_name = None
    
    for model_name in models_to_train:
        print(f"\nTraining {model_name}...")
        model = train_model(
            X_train, y_train, X_valid, y_valid,
            model_name=model_name,
            h_opt_max_evals=2  # Number of hyperopt evaluations
        )
        # Evaluate on validation set
        eval_metrics_valid = evaluate_model(model=model, X=X_valid, y=y_valid, model_name=model_name, dataset_name="valid")
        print(f"Validation Accuracy ({model_name}): {eval_metrics_valid['accuracy']:.4f}")
        
        # If this model has the best validation score, save it
        if eval_metrics_valid['accuracy'] > best_score:
            best_score = eval_metrics_valid['accuracy']
            best_model = model
            best_model_name = model_name
    
    print(f"\nBest model on validation set: {best_model_name} with accuracy: {best_score:.4f}")
    
    # Save the best model with the current date (dd_mm_yy)
    current_date = datetime.now().strftime("%d_%m_%y")
    best_model_fname = os.path.join(models_dir, f"{best_model_name}_best_model_{current_date}.joblib")
    joblib.dump(best_model, best_model_fname)
    print(f"Best model saved to {best_model_fname}")

    # Evaluate the best model on the test set
    eval_metrics_test = evaluate_model(model=best_model, X=X_test, y=y_test, model_name=best_model_name, dataset_name="test")
    print(f"Test Accuracy: {eval_metrics_test['accuracy']:.4f}")
    print(f"Test Balanced Accuracy: {eval_metrics_test['balanced_accuracy']:.4f}")
    print("Classification Report (Test Data):")
    print(eval_metrics_test["conf_matrix_df"])
        
